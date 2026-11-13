"""Simulator validation against the real pilot (task #61).

    python scripts/validate_sim.py --pilot results/pilot --calibrate     # fit on pilot 1, save
    python scripts/validate_sim.py --pilot results/pilot                 # check with the saved fit

Calibration (--calibrate) fits four parameters of the simulator on pilot 1 (scripts/sim_a5.py,
CAL): decode and prompt speed relative to the profile, a fixed client-side time per request, and
how long a GPU keeps its busy power after its last request. Hold-out check: pilot 2 ran the same
requests on other pods of the same GPU types (results/pilot2/README.md); its measured numbers are
compared with the calibrated simulator, which was not fitted on them.

For every strategy of the pilot, the exact requests of the real run (arrival time, prompt and
output length, type: from <run>_client.csv) are replayed in the simulator of the testbed
(scripts/sim_a5.py, measured profiles), and the simulated metrics are compared with the measured
ones. Requests that arrive in the first `warmup` seconds are left out of both. Target: every
compared metric within 15% (relative error), except TTFT, which the simulator does not model
in detail (network and HTTP time); TTFT is reported but not part of the target.
"""
import argparse, csv, glob, importlib, json, random, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import sim_a5 as S
from strategies.length_predictor import LengthPredictor

PROFILE_NAME = {"rtx4090": "rtx4090", "rtxa6000": "rtxa6000", "rtx2000ada": "rtx2000ada"}


def pct(v, q):
    v = sorted(v); k = (len(v) - 1) * q; lo = int(k); hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def real_metrics(meta, d, warmup):
    rows = list(csv.DictReader(open(d / meta["requests_csv"])))
    t0 = float(meta["start_ts"]) + warmup
    rows = [r for r in rows if float(r["arrival_ts"]) >= t0]
    e2e = [float(r["e2e_ms"]) / 1000 for r in rows]
    ttft = [float(r["ttft_ms"]) / 1000 for r in rows]
    energy = json.loads((d / f"{meta['run_id']}_energy.json").read_text())
    share = {n: sum(1 for r in rows if r["gpu"] == n) / len(rows) for n in S.PROFILES}
    return {"e2e_p50": pct(e2e, .5), "e2e_p95": pct(e2e, .95), "ttft_p95": pct(ttft, .95),
            "j_per_token": energy["cluster"]["j_per_token"], "share": share}


def sim_metrics(strategy, d, meta, warmup, nodes, pred, cal=None, lam=None):
    client = sorted(csv.DictReader(open(d / f"{meta['run_id']}_client.csv")), key=lambda r: float(r["scheduled_s"]))
    jobs = [{"at": float(r["scheduled_s"]), "req_type": r["req_type"], "prompt_len": int(r["prompt_len"]),
             "output_len": int(r["output_len"])} for r in client]
    mod = importlib.import_module("strategies." + strategy)
    if hasattr(mod, "PREDICTOR"):
        mod.PREDICTOR = pred
    res = S.simulate(mod, jobs, nodes, keep=True, cal=cal, lam=lam)
    done = [j for j in res["jobs"] if j["at"] >= warmup]
    e2e = [j["end"] - j["at"] for j in done]; ttft = [j["first"] - j["at"] for j in done]
    share = {n: sum(1 for j in done if j["gpu"] == n) / len(done) for n in nodes}
    return {"e2e_p50": pct(e2e, .5), "e2e_p95": pct(e2e, .95), "ttft_p95": pct(ttft, .95),
            "j_per_token": res["j_per_token"], "share": share}


PILOT2 = {   # measured on other pods, same requests (results/pilot2/README.md): e2e p50, e2e p95 (s), J/token
    "round_robin": (3.30, 25.78, 2.34), "weighted": (2.84, 10.61, 2.53), "least_load": (2.73, 10.80, 2.59),
    "predictive": (2.93, 6.83, 2.42), ("marginal_cost", 0.0): (2.68, 7.19, 2.27), ("marginal_cost", 0.01): (2.95, 7.87, 2.02)}


def error(runs, nodes, pred, cal, keys=("e2e_p50", "e2e_p95", "j_per_token")):
    errs = []
    for meta, d, real in runs:
        sim = sim_metrics(meta["strategy"], d, meta, 30, nodes, pred, cal)
        errs += [abs(sim[k] - real[k]) / real[k] for k in keys]
    return statistics.mean(errs), max(errs)


def calibrate(runs, nodes, pred):
    best = None
    for sp in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0):
        for ov in (0.0, 0.2, 0.4, 0.6, 0.8):
            for pf in (0.5, 0.75, 1.0):
                cal = {"speed_scale": sp, "overhead_s": ov, "prefill_scale": pf, "tail_s": 0.0}
                m, _ = error(runs, nodes, pred, cal, ("e2e_p50", "e2e_p95"))
                if best is None or m < best[0]:
                    best = (m, cal)
    cal = best[1]
    best_t = None
    for tail in (0, 1, 2, 3, 4, 6, 8, 10, 15):
        c = dict(cal, tail_s=float(tail))
        m, _ = error(runs, nodes, pred, c, ("j_per_token",))
        if best_t is None or m < best_t[0]:
            best_t = (m, c)
    return best_t[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default=str(ROOT / "results" / "pilot"))
    ap.add_argument("--warmup", type=float, default=30)
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()
    nodes = S.node_profiles()
    pred = LengthPredictor.fit(S.sample(random.Random(999), 3000))
    runs = []
    for mp in sorted(glob.glob(str(Path(a.pilot) / "*_meta.json"))):
        meta = json.loads(Path(mp).read_text()); d = Path(mp).parent
        runs.append((meta, d, real_metrics(meta, d, a.warmup)))
    if a.calibrate:
        cal = calibrate(runs, nodes, pred)
        S.CAL_FILE.write_text(json.dumps({"cal": cal, "fitted_on": str(a.pilot)}, indent=2))
        print("calibration:", cal)
        S.CAL.update(cal)
    rows = []
    for meta, d, real in runs:
        sim = sim_metrics(meta["strategy"], d, meta, a.warmup, nodes, pred)
        row = {"strategy": meta["strategy"]}
        for m in ("e2e_p50", "e2e_p95", "j_per_token", "ttft_p95"):
            row[f"{m}_real"], row[f"{m}_sim"] = round(real[m], 3), round(sim[m], 3)
            row[f"{m}_err"] = round((sim[m] - real[m]) / real[m], 3)
        row["share_real"] = " / ".join(f"{real['share'][n]:.2f}" for n in nodes)
        row["share_sim"] = " / ".join(f"{sim['share'][n]:.2f}" for n in nodes)
        rows.append(row)
    for r in rows:
        print(r)
    errs = [abs(r[f"{m}_err"]) for r in rows for m in ("e2e_p50", "e2e_p95", "j_per_token")]
    print(f"pilot 1 (fitted): mean |error| {statistics.mean(errs):.1%}, max {max(errs):.1%}")
    meta0, d0, _ = runs[0]
    hold = []
    for key, (p50, p95, jpt) in PILOT2.items():
        strat, lam = (key, None) if isinstance(key, str) else key
        sim = sim_metrics(strat, d0, meta0, a.warmup, nodes, pred, lam=lam)
        e = [(sim["e2e_p50"] - p50) / p50, (sim["e2e_p95"] - p95) / p95, (sim["j_per_token"] - jpt) / jpt]
        hold.append({"strategy": str(key), "e2e_p50_err": round(e[0], 3), "e2e_p95_err": round(e[1], 3), "j_per_token_err": round(e[2], 3)})
        print("pilot 2 (hold-out)", hold[-1])
    he = [abs(v) for h in hold for k, v in h.items() if k.endswith("_err")]
    print(f"pilot 2 (hold-out): mean |error| {statistics.mean(he):.1%}, max {max(he):.1%}  (target 15%)")
    with open(Path(a.pilot) / "sim_validation_holdout.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hold[0].keys())); w.writeheader(); w.writerows(hold)
    out = Path(a.pilot) / "sim_validation.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
