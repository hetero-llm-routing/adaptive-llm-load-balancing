"""Calibration (task #60): the round-robin saturation rate of each workload family, so that the
load levels rho = 0.3, 0.6, 0.9 mean the same on every workload (Section 5.2).

    python bench/calibrate.py --config config_run.yaml --workloads bench/workloads \
        --families chat doc rag mixed --out results/calibration.json [--step-s 180]

A rate is stable when, after the warm-up, (1) at least 95% of the requests that arrive finish
within --drain-s (30 s) after the last arrival and (2) the latency does not grow during the step (90th
percentile end-to-end latency of the last third <= 2 x that of the first third; the 90th
percentile, because under round robin only the slowest node's share of the requests queues). The
saturation rate is the highest stable rate, found by doubling/halving from the capacity
estimate sum_j s_j(4) / mean output length, then by bisection to within 5%.
"""
import argparse, csv, json, statistics, time
from pathlib import Path

import yaml

import replay as R


def stable(req_csv, client_log, t0, warmup):
    """client_log: every scheduled request, finished or cancelled after the drain time."""
    sched = [r for r in client_log if r["scheduled_s"] >= warmup]
    if len(sched) < 12:
        return False, {"n": len(sched)}
    done = sum(1 for r in sched if r["status"] == 200 and not r["error"]) / len(sched)
    rows = [r for r in csv.DictReader(open(req_csv)) if float(r["arrival_ts"]) >= t0 + warmup]
    rows.sort(key=lambda r: float(r["arrival_ts"]))
    k = max(1, len(rows) // 3)
    p90 = lambda rs: sorted(float(r["e2e_ms"]) for r in rs)[int(0.9 * (len(rs) - 1))]
    first, last = (p90(rows[:k]), p90(rows[-k:])) if len(rows) >= 6 else (1, 1e9)
    info = {"n": len(sched), "done_share": round(done, 3), "e2e_p90_first_ms": round(first), "e2e_p90_last_ms": round(last)}
    return done >= 0.95 and last <= 2.0 * first, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workloads", default="bench/workloads")
    ap.add_argument("--families", nargs="+", default=["chat", "doc", "rag", "mixed"])
    ap.add_argument("--out", default="results/calibration.json")
    ap.add_argument("--results-dir", default="results/calibration")
    ap.add_argument("--step-s", type=float, default=180)
    ap.add_argument("--warmup-s", type=float, default=30)
    ap.add_argument("--cooldown-s", type=float, default=20)
    ap.add_argument("--drain-s", type=float, default=30, help="requests not finished this long after the last arrival count as not done")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=10)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    cap_tps = sum(float((n.get("speed_by_n") or [n.get("speed_tps", 0)])[-1]) for n in cfg["nodes"])
    out = Path(a.out); res = json.loads(out.read_text()) if out.exists() else {}

    for fam in a.families:
        pool = R.load_workload(Path(a.workloads) / f"{fam}.jsonl")
        est = cap_tps / statistics.mean(r["output_len"] for r in pool)
        lo, hi, rate, steps = None, None, est, []
        for i in range(a.max_steps):
            sched, _ = R.schedule(pool, rate, a.step_s, a.seed + i)
            rid = f"cal_{fam}_{i}"
            m, req_csv, clog = R.run_once(a.config, "round_robin", rid, a.results_dir, sched, a.seed + i,
                                          meta={"family": fam, "rate": rate}, drain_s=a.drain_s)
            ok, info = stable(req_csv, clog, m["start_ts"], a.warmup_s)
            steps.append(dict(info, rate=round(rate, 4), stable=ok, run_id=rid))
            print(fam, steps[-1], flush=True)
            if ok:
                lo = rate
            else:
                hi = rate
            if lo is not None and hi is not None and (hi - lo) / hi <= 0.05:
                break
            rate = (lo + hi) / 2 if lo is not None and hi is not None else (rate * 2 if hi is None else rate / 2)
            time.sleep(a.cooldown_s)
        res[fam] = {"sat_rate": lo, "upper": hi, "estimate": round(est, 4), "step_s": a.step_s, "steps": steps}
        out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v["sat_rate"] for k, v in res.items()}, indent=2))


if __name__ == "__main__":
    main()
