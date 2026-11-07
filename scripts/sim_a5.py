"""Quick simulator of the real testbed with the measured profiles (speed and power by
concurrency, A5), for trying strategies before GPU runs. Not the paper's simulator (task #48/#61).

    python scripts/sim_a5.py --rho 0.6 --seeds 1 2 3
"""
import argparse, importlib, json, math, random, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strategies.base import Request, GpuState
from strategies.length_predictor import LengthPredictor

PROFILES = {"rtx4090": "geforce_rtx_4090", "rtxa6000": "rtx_a6000", "rtx2000ada": "rtx_2000_ada_generation"}
SAT_MIXED = 1.254          # round-robin saturation of the mixed workload, real cluster (req/s)
FAMS = {  # share, prompt (median, sigma, cap), output (median, sigma, lo, hi) from results/calibration
    "chat": (0.70, (40, 1.17, 2048), (252, 0.62, 16, 1024)),
    "doc": (0.15, (3269, 0.35, 6000), (128, 0.84, 32, 512)),
    "rag": (0.15, (664, 0.25, 4096), (137, 0.75, 16, 512)),
}


def node_profiles():
    out = {}
    for name, f in PROFILES.items():
        p = json.loads((ROOT / "results" / "profiles" / f"{f}.json").read_text())
        out[name] = {"speed_tps": p["speed_tps"], "prefill_tps": p["prefill_tps"],
                     "speed_by_n": [p["decode_by_concurrency"][str(n)]["total_tps"] for n in (1, 2, 3, 4)],
                     "power_by_n": [p["power"][f"decode_{n}"]["mean_w"] for n in (1, 2, 3, 4)],
                     "idle_w": p["power"]["idle"]["mean_w"],
                     "prefill_w": p["power"]["prefill_long"]["mean_w"]}
    return out


def sample(rng, n):
    out = []
    for i in range(n):
        x, acc = rng.random(), 0
        for fam, (share, pr, orr) in FAMS.items():
            acc += share
            if x <= acc:
                break
        P = int(min(pr[2], max(8, rng.lognormvariate(math.log(pr[0]), pr[1]))))
        O = int(min(orr[3], max(orr[2], rng.lognormvariate(math.log(orr[0]), orr[1]))))
        out.append({"req_type": fam, "prompt_len": P, "output_len": O})
    return out


def simulate(mod, jobs, nodes, dt=0.02, lam=None):
    if hasattr(mod, "reset"):
        mod.reset()
    if lam is not None and hasattr(mod, "LAMBDA"):
        mod.LAMBDA = lam
    st = {n: {"run": [], "queue": [], "energy": 0.0} for n in nodes}
    t, i, done = 0.0, 0, []
    hook = getattr(mod, "on_complete", None)
    while i < len(jobs) or any(s["run"] or s["queue"] for s in st.values()):
        while i < len(jobs) and jobs[i]["at"] <= t:
            j = jobs[i]; i += 1
            states = {n: GpuState(queue_len=len(s["queue"]), running=len(s["run"]), kv_usage=len(s["run"]) / 4,
                                  extra=dict(nodes[n], inflight=len(s["run"]) + len(s["queue"])))
                      for n, s in st.items()}
            req = Request(request_id=str(i), prompt_len=j["prompt_len"], req_type=j["req_type"], arrival_ts=t)
            g = mod.choose_gpu(req, states)
            j.update(gpu=g, pre=float(j["prompt_len"]), gen=0.0, first=None, id=str(i))
            st[g]["queue"].append(j)
        for n, s in st.items():
            pr = nodes[n]
            while s["queue"] and len(s["run"]) < 4:
                s["run"].append(s["queue"].pop(0))
            k = len(s["run"])
            s["energy"] += (pr["power_by_n"][k - 1] if k else pr["idle_w"]) * dt
            if not k:
                continue
            pre = [r for r in s["run"] if r["pre"] > 0]
            if pre:                      # a prompt in progress pauses decoding (one at a time)
                pre[0]["pre"] -= pr["prefill_tps"] * dt
                continue
            v = pr["speed_by_n"][k - 1] / k
            for r in list(s["run"]):
                r["gen"] += v * dt
                if r["first"] is None:
                    r["first"] = t + dt
                if r["gen"] >= r["output_len"]:
                    r["end"] = t + dt; s["run"].remove(r); done.append(r)
                    if hook:
                        hook(r["id"], n, (r["first"] - r["at"]) * 1000, (r["end"] - r["at"]) * 1000, r["output_len"])
        t += dt
    q = lambda v, x: sorted(v)[min(len(v) - 1, int(x * len(v)))]
    ttft = [r["first"] - r["at"] for r in done]; e2e = [r["end"] - r["at"] for r in done]
    toks = sum(r["output_len"] for r in done)
    E = sum(s["energy"] for s in st.values())
    share = {n: sum(1 for r in done if r["gpu"] == n) / len(done) for n in nodes}
    return {"ttft_p50": q(ttft, .5), "ttft_p95": q(ttft, .95), "e2e_p50": q(e2e, .5), "e2e_p95": q(e2e, .95),
            "j_per_token": E / toks, "share": share}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, nargs="+", default=[0.6])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--strategies", nargs="+", default=["round_robin", "weighted", "least_load", "predictive", "marginal_cost"])
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.002, 0.01])
    a = ap.parse_args()
    nodes = node_profiles()
    pred = LengthPredictor.fit(sample(random.Random(999), 3000))
    for rho in a.rho:
        res = {}
        for seed in a.seeds:
            rng = random.Random(seed)
            base, t = sample(rng, a.n), 0.0
            for j in base:
                t += rng.expovariate(rho * SAT_MIXED); j["at"] = t
            for s in a.strategies:
                mod = importlib.import_module("strategies." + s)
                if hasattr(mod, "PREDICTOR"):
                    mod.PREDICTOR = pred
                for lam in (a.lambdas if s == "marginal_cost" else [None]):
                    key = s if lam is None else f"{s}(l={lam})"
                    r = simulate(mod, [dict(j) for j in base], nodes, lam=lam)
                    res.setdefault(key, []).append(r)
        print(f"\nrho = {rho}  (mean over seeds {a.seeds})")
        print(f"{'strategy':26s} {'TTFT p50':>8s} {'TTFT p95':>8s} {'E2E p50':>8s} {'E2E p95':>8s} {'J/tok':>6s}  share 4090/A6000/2000Ada")
        for k, rs in res.items():
            m = lambda key: statistics.mean(r[key] for r in rs)
            sh = {n: statistics.mean(r["share"][n] for r in rs) for n in nodes}
            print(f"{k:26s} {m('ttft_p50'):8.2f} {m('ttft_p95'):8.2f} {m('e2e_p50'):8.2f} {m('e2e_p95'):8.2f} {m('j_per_token'):6.3f}  "
                  + " / ".join(f"{sh[n]:.2f}" for n in nodes))


if __name__ == "__main__":
    main()
