"""Experiment runner (task #50): runs every cell of a matrix written in a YAML file.

    python bench/run_matrix.py --matrix bench/matrix_example.yaml [--only main] [--dry-run]

For each experiment, workload, load level rho and seed, the request schedule is built once
(seeded: same requests and arrival times for every strategy, a paired comparison), and the
strategies run on it one after another in a random order drawn from the seed. The rate is
rho x the round-robin saturation rate of the workload (bench/calibrate.py). Every run writes
into results_dir: <run>_*.csv (router), <run>_client.csv, <run>_power.csv, <run>_energy.json,
<run>_meta.json. A run whose meta file exists is skipped, so an interrupted matrix resumes.
"""
import argparse, itertools, json, os, random, time
from pathlib import Path

import yaml

import replay as R


def cells(m, only=None):
    for exp in m["experiments"]:
        if only and exp["name"] not in only:
            continue
        for wl, rho, seed in itertools.product(exp["workloads"], exp["rho"], exp.get("seeds", m["seeds"])):
            strategies = list(exp.get("strategies", m["strategies"]))
            random.Random(f"{seed}-{wl}-{rho}-{exp['name']}").shuffle(strategies)
            yield exp, wl, rho, seed, strategies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--only", nargs="*", help="experiment names to run")
    ap.add_argument("--dry-run", action="store_true", help="print the runs and their request counts only")
    a = ap.parse_args()
    m = yaml.safe_load(open(a.matrix))
    root = Path(m.get("results_dir", "results/runs"))
    cal = json.loads(Path(m["calibration"]).read_text()) if Path(m["calibration"]).exists() else {}
    wdir = Path(m.get("workloads_dir", "bench/workloads"))
    todo = list(cells(m, a.only))
    n_runs = sum(len(c[4]) for c in todo)
    print(f"{len(todo)} cells, {n_runs} runs, about {n_runs * (m['duration_s'] + m.get('cooldown_s', 30)) / 3600:.1f} h")
    done = 0
    for exp, wl, rho, seed, strategies in todo:
        sat = (cal.get(wl) or {}).get("sat_rate") or m.get("sat_rate_default", {}).get(wl)
        if not sat:
            raise SystemExit(f"no saturation rate for {wl}: run bench/calibrate.py first")
        rate = rho * sat
        pool = R.load_workload(wdir / f"{wl}.jsonl")
        arrival = exp.get("arrival", "poisson")
        sched, extra = R.schedule(pool, rate, m["duration_s"], seed, arrival, wdir / "azure_conv.csv")
        for s in strategies:
            rid = f"{exp['name']}_{wl}_r{int(rho * 100):02d}_s{seed}_{s}"
            done += 1
            out = root / exp["name"]
            if (out / f"{rid}_meta.json").exists():
                continue
            print(f"[{done}/{n_runs}] {rid}: {len(sched)} requests at {rate:.3f}/s", flush=True)
            if a.dry_run:
                continue
            meta = dict(experiment=exp["name"], workload=wl, rho=rho, rate=rate, sat_rate=sat, arrival=arrival,
                        duration_s=m["duration_s"], warmup_s=m.get("warmup_s", 30), config=m["config"], **extra)
            env_old = {k: os.environ.get(k) for k in exp.get("env", {})}
            os.environ.update({k: str(v) for k, v in exp.get("env", {}).items()})   # e.g. LB_MC_LAMBDA for the sweep
            mm, req_csv, _ = R.run_once(m["config"], s, rid, out, sched, seed, m.get("timeout_s", 600), meta,
                                         drain_s=m.get("drain_s", 300))
            for k, v in env_old.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
            print("   ", R.summarize(req_csv, m.get("warmup_s", 30), mm["start_ts"]),
                  f"failed={mm['n_failed']} power_rows={mm['power_rows']} max_lag_ms={mm['max_lag_ms']}", flush=True)
            time.sleep(m.get("cooldown_s", 30))


if __name__ == "__main__":
    main()
