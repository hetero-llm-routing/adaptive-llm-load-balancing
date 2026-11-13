"""Simulate the routing strategies on clusters of 3, 8 and 16 GPUs (task #79, Section 6.5).

    python scripts/sim_scaling.py                                   # 3, 8, 16 nodes x rho 0.3, 0.6, 0.9 x 5 seeds
    python scripts/sim_scaling.py --check-saturation [--step-s 900]  # round-robin saturation of each cluster

Cluster: k nodes of the three measured GPU types (results/profiles). 3 nodes = the testbed, one
of each type. 8 and 16 cannot be split into three equal parts, so each is simulated with the
three rotations of its mix (RTX 4090 / RTX A6000 / RTX 2000 Ada = 3/3/2, 2/3/3, 3/2/3 for 8
nodes; 6/5/5, 5/6/5, 5/5/6 for 16) on the same requests, and a cell averages the three. Each
GPU type is then one third of the cluster on average, as on the testbed; with one mix only,
the share of the most efficient GPU (RTX 2000 Ada) would change with the cluster size and move
the energy per token for that reason alone.

Load: rho times the round-robin saturation rate of the k-node cluster. Round robin gives every
node 1/k of the requests, so it saturates when the slowest node does, and its saturation rate
grows with k: lambda_sat(k) = k/3 * lambda_sat(3), with lambda_sat(3) = 1.254 req/s measured on
the testbed for the mixed workload (results/calibration/calibration.json). The load per node is
therefore the same for every cluster size. --check-saturation repeats the calibration of
bench/calibrate.py in the simulator to check this scaling.

Workload: the mixed workload of the simulator (70% chat, 15% doc, 15% RAG; scripts/sim_a5.py),
Poisson arrivals for 300 s as in the real runs, the same requests for every strategy of a cell.

Simulator: scripts/sim_a5.py, with the measured speed and power by concurrency (A5) and the
calibration fitted on the pilot (results/sim_calibration.json).

Strategies: the seven of the real runs (marginal_cost with lambda = 0.002 s/J, its default there)
plus power_of_two and oracle, which run in the simulator only.

Metrics: as analysis/analyze.py computes them for the real runs. Requests that arrive in the
first 30 s are left out of the latency, throughput and utilization metrics; the energy covers
the whole run, from the first arrival to the last completion, on every node (Eq. 9).

Output (default results/scaling/):
  sim_runs.csv     one row per run, with the columns of results/phase5/runs.csv; experiment is
                   sim3, sim8 or sim16 and run_id names the mix (m332 = 3 RTX 4090, 3 RTX A6000,
                   2 RTX 2000 Ada). The per-GPU columns are per GPU type: U is the mean over the
                   nodes of that type, share the share of the requests sent to that type, jpt the
                   energy of those nodes divided by the tokens they generated. imbalance_L (Eq. 7)
                   is computed over all nodes.
  sim_summary.csv  mean over the runs of a cell (5 seeds, times 3 mixes for 8 and 16 nodes) and
                   95% confidence interval (half-width), with the columns of
                   results/phase5/summary.csv; the seeds column counts the runs.
"""
import argparse, csv, importlib, json, math, random, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import sim_a5 as S
from strategies.length_predictor import LengthPredictor

TYPES = ["rtx4090", "rtxa6000", "rtx2000ada"]          # node i is of type TYPES[(i + rotation) % 3]
STRATEGIES = ["round_robin", "weighted", "least_load", "least_work", "predictive", "rl", "marginal_cost",
              "power_of_two", "oracle"]
MC_LAMBDA = 0.002                                      # s/J, marginal_cost default in the real runs
DURATION_S = 300.0                                     # arrivals per run, as in the real runs
WARMUP_S = 30.0
METRICS = ["ttft_mean_ms", "ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms", "tpot_mean_ms", "tpot_p95_ms",
           "e2e_p50_ms", "e2e_p95_ms", "throughput_tps", "imbalance_L", "j_per_token", "failed_share"]
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
       11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
       20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
       29: 2.045}                                     # two-sided 95% t quantiles by degrees of freedom


def testbed_sat_rate():
    """Round-robin saturation rate of the mixed workload on the real 3-node testbed (req/s)."""
    return json.loads((ROOT / "results" / "calibration" / "calibration.json").read_text())["mixed"]["sat_rate"]


def sat_rate(k):
    return k / 3 * testbed_sat_rate()


def rotations(k):
    """The mixes simulated for k nodes: one if k splits into three equal parts, else all three."""
    return [0] if k % 3 == 0 else [0, 1, 2]


def cluster(k, rotation=0):
    """Node profiles of a k-node cluster and the GPU type of every node."""
    prof = S.node_profiles()
    if k == 3:                                         # same node names as the real testbed
        return {t: dict(prof[t]) for t in TYPES}, {t: t for t in TYPES}
    nodes, kind, count = {}, {}, {t: 0 for t in TYPES}
    for i in range(k):
        t = TYPES[(i + rotation) % 3]
        count[t] += 1
        name = f"{t}-{count[t]}"
        nodes[name], kind[name] = dict(prof[t]), t
    return nodes, kind


def mix_name(kind):
    """m332 = 3 RTX 4090, 3 RTX A6000, 2 RTX 2000 Ada (counts in the order of TYPES)."""
    return "m" + "".join(str(list(kind.values()).count(t)) for t in TYPES)


def workload(rate, seed, duration=DURATION_S, n=None):
    """Mixed-workload requests with Poisson arrivals at `rate` req/s, for `duration` seconds or,
    if n is given, until there are n requests."""
    rng = random.Random(seed)
    jobs, t = [], 0.0
    while True:
        t += rng.expovariate(rate)
        if (n is None and t > duration) or (n is not None and len(jobs) >= n):
            return jobs
        j = S.sample(rng, 1)[0]
        j["at"] = t
        jobs.append(j)


def length_predictor():
    """The predictor the simulator and the RL training use (scripts/sim_a5.py, scripts/train_rl.py)."""
    return LengthPredictor.fit(S.sample(random.Random(999), 3000))


def load_strategy(name, pred):
    mod = importlib.import_module("strategies." + name)
    if hasattr(mod, "PREDICTOR"):
        mod.PREDICTOR = pred
    return mod


def pct(v, q):
    """Percentile by linear interpolation (numpy's default), as analysis/analyze.py."""
    if not v:
        return float("nan")
    v = sorted(v)
    k = (len(v) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def union_length(intervals):
    total, end = 0.0, -math.inf
    for a, b in sorted(intervals):
        if b <= end:
            continue
        total += b - max(a, end)
        end = b
    return total


def run_row(columns, exp, rho, seed, strategy, res, kind, mix=""):
    """One row of runs.csv from a simulated run (definitions of analysis/analyze.py)."""
    done = res["jobs"]
    reqs = [r for r in done if r["at"] >= WARMUP_S]     # the run starts at t = 0 (start_ts of a real run)
    ttft = [(r["first"] - r["at"]) * 1000 for r in reqs]
    tpot = [(r["end"] - r["first"]) / (r["output_len"] - 1) * 1000 for r in reqs if r["output_len"] > 1]
    e2e = [(r["end"] - r["at"]) * 1000 for r in reqs]
    T = max(r["end"] for r in reqs) - min(r["at"] for r in reqs)
    U = {j: union_length([(r["at"], r["end"]) for r in reqs if r["gpu"] == j]) / T for j in kind}
    mean_u = statistics.mean(U.values())
    energy = res["energy_by_node"]                     # whole run, every node (Eq. 9)
    tokens = {j: sum(r["output_len"] for r in done if r["gpu"] == j) for j in kind}
    row = {"experiment": exp, "workload": "mixed", "rho": rho, "seed": seed, "strategy": strategy,
           "run_id": f"{exp}{'-' + mix if mix else ''}_mixed_r{round(rho * 100)}_s{seed}_{strategy}", "n": len(reqs),
           "ttft_mean_ms": statistics.mean(ttft), "ttft_p50_ms": pct(ttft, .5), "ttft_p95_ms": pct(ttft, .95),
           "ttft_p99_ms": pct(ttft, .99), "tpot_mean_ms": statistics.mean(tpot), "tpot_p95_ms": pct(tpot, .95),
           "e2e_p50_ms": pct(e2e, .5), "e2e_p95_ms": pct(e2e, .95),
           "throughput_tps": sum(r["output_len"] for r in reqs) / T,
           "imbalance_L": max(U.values()) / mean_u - 1 if mean_u > 0 else float("nan"),
           "j_per_token": sum(energy.values()) / sum(tokens.values()), "failed_share": 0.0}
    for t in TYPES:
        members = [j for j in kind if kind[j] == t]
        tok = sum(tokens[j] for j in members)
        row[f"U_{t}"] = statistics.mean(U[j] for j in members)
        row[f"share_{t}"] = sum(1 for r in reqs if kind[r["gpu"]] == t) / len(reqs)
        row[f"jpt_{t}"] = sum(energy[j] for j in members) / tok if tok else ""
    missing = [c for c in columns if c not in row]
    assert not missing, f"columns not computed: {missing}"
    return {c: row[c] for c in columns}


def ci95(v):
    v = [x for x in v if x not in ("", None) and not math.isnan(x)]
    if not v:
        return float("nan"), float("nan")
    m = statistics.mean(v)
    if len(v) < 2:
        return m, float("nan")
    return m, T95.get(len(v) - 1, 1.96) * statistics.stdev(v) / math.sqrt(len(v))


def summarize(rows):
    cells = {}
    for r in rows:
        cells.setdefault((r["experiment"], r["workload"], r["rho"], r["strategy"]), []).append(r)
    out = []
    for (exp, wl, rho, s), rs in cells.items():
        row = {"experiment": exp, "workload": wl, "rho": rho, "strategy": s, "seeds": len(rs)}
        for m in METRICS:
            row[m], row[m + "_ci95"] = ci95([r[m] for r in rs])
        out.append(row)
    return out


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def header(path):
    with open(path, newline="") as f:
        return next(csv.reader(f))


def run_matrix(a):
    columns = header(ROOT / "results" / "phase5" / "runs.csv")
    pred = length_predictor()
    rows = []
    for k in a.nodes:
        for rot in rotations(k):
            nodes, kind = cluster(k, rot)
            mix = mix_name(kind) if len(rotations(k)) > 1 else ""
            for rho in a.rho:
                for seed in a.seeds:
                    jobs = workload(rho * sat_rate(k), f"scaling-{k}-{rho}-{seed}")   # same requests for every mix
                    for s in a.strategies:
                        t_start = time.time()
                        res = S.simulate(load_strategy(s, pred), [dict(j) for j in jobs], nodes, keep=True,
                                         lam=MC_LAMBDA if s == "marginal_cost" else None)
                        rows.append(run_row(columns, f"sim{k}", rho, seed, s, res, kind, mix))
                        r = rows[-1]
                        print(f"{r['run_id']:40s} n={r['n']:5d}  E2E p95 {r['e2e_p95_ms'] / 1000:7.2f} s  "
                              f"J/token {r['j_per_token']:.3f}  ({time.time() - t_start:.1f} s)", flush=True)
    out = Path(a.out)
    write_csv(out / "sim_runs.csv", rows, columns)
    write_csv(out / "sim_summary.csv", summarize(rows), header(ROOT / "results" / "phase5" / "summary.csv"))
    print(f"{len(rows)} runs -> {out / 'sim_runs.csv'}, {out / 'sim_summary.csv'}")


def stable(res, warmup, drain_s):
    """Stability rule of bench/calibrate.py, applied to a simulated step."""
    done = res["jobs"]
    last_arrival = max(r["at"] for r in done)
    rows = sorted((r for r in done if r["at"] >= warmup), key=lambda r: r["at"])
    if len(rows) < 12:
        return False
    finished = sum(1 for r in rows if r["end"] <= last_arrival + drain_s) / len(rows)
    k = max(1, len(rows) // 3)
    p90 = lambda rs: sorted(r["end"] - r["at"] for r in rs)[int(0.9 * (len(rs) - 1))]
    return finished >= 0.95 and p90(rows[-k:]) <= 2.0 * p90(rows[:k])


def check_saturation(a, drain_s=30.0, repeats=5, max_steps=12):
    """Round-robin saturation rate of every cluster size in the simulator (doubling or halving,
    then bisection to within 5%, as bench/calibrate.py does on the real cluster). One step
    decides stability on few requests, so each step is repeated with `repeats` seeds and a rate
    counts as stable when the majority of them are."""
    step_s = a.step_s
    rr = load_strategy("round_robin", length_predictor())
    out = {}
    for k in a.nodes:
        nodes, _ = cluster(k)
        lo = hi = None
        rate = sat_rate(k)
        for i in range(max_steps):
            votes = sum(stable(S.simulate(rr, workload(rate, f"saturation-{k}-{i}-{r}", duration=step_s), nodes,
                                          keep=True), WARMUP_S, drain_s) for r in range(repeats))
            ok = votes > repeats / 2
            lo, hi = (rate, hi) if ok else (lo, rate)
            if lo is not None and hi is not None and (hi - lo) / hi <= 0.05:
                break
            rate = (lo + hi) / 2 if lo is not None and hi is not None else (rate * 2 if hi is None else rate / 2)
        out[k] = lo
        print(f"{k:2d} nodes, {step_s:.0f} s steps: simulated round-robin saturation {lo:.3f} req/s "
              f"({lo / k:.3f} per node);  k/3 x testbed = {sat_rate(k):.3f} req/s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, nargs="+", default=[3, 8, 16])
    ap.add_argument("--rho", type=float, nargs="+", default=[0.3, 0.6, 0.9])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--strategies", nargs="+", default=STRATEGIES)
    ap.add_argument("--out", default=str(ROOT / "results" / "scaling"))
    ap.add_argument("--check-saturation", action="store_true")
    ap.add_argument("--step-s", type=float, default=180.0, help="--check-saturation: seconds of arrivals per step")
    a = ap.parse_args()
    if a.check_saturation:
        check_saturation(a)
    else:
        run_matrix(a)


if __name__ == "__main__":
    main()
