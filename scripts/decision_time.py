"""Decision time of every routing strategy (task #77, Section 6.4, RQ3).

    python scripts/decision_time.py                  # 3, 8, 16 nodes; rho 0.6 and 0.9; 10,000 decisions
    python scripts/decision_time.py --nodes 16 --strategies rl --decisions 2000 --repeats 3

Each strategy routes a stream of mixed-workload requests in the simulator of the testbed
(scripts/sim_a5.py: measured speed and power by concurrency, A5) on a cluster of 3, 8 or 16
nodes (scripts/sim_scaling.py), and every call of choose_gpu is timed with
time.perf_counter_ns(). The node states and the strategy's own records (backlog, tracked
requests, completions reported through on_complete) are those of a run at that load, and the
decision is timed the way the router calls it. The first 500 decisions warm up the interpreter
and are not counted; the next --decisions are.

Other programs on the machine slow some calls down, at random and in busy periods. Every run is
therefore repeated --repeats times, in rounds: round 1 runs every strategy, cluster size and load
once, then round 2 runs them all again, and so on, so the repeats of one run are minutes apart.
The repeats are identical (same requests, same states, same choices, checked), so for every
decision the fastest of its repeats is kept, which removes most of the disturbance (the minimum of
repeated runs, as timeit recommends). median_us, p90_us, p99_us, mean_us and max_us are over these
per-decision minima; single_run_median_us and single_run_p99_us are those of round 1 alone,
disturbance included.

Only the decision is timed. Reading the node states is the router's work: on the testbed it
reads them in the background every 0.25 s (router.state_interval_s in config_testbed.yaml), off
the request path, and the cost is the same for every strategy that reads the state.

The garbage collector is paused during a run, as timeit does, so that its pauses, which do not
depend on the strategy, are not counted. Runs are sequential in one process.

Output (default results/overhead/decision_time.csv): one row per strategy, cluster size and
load, plus the CPU, the Python version, the timer resolution and the time of an empty call
through the same timing code (timer_floor_us, the measurement floor).
"""
import argparse, csv, gc, platform, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import sim_a5 as S
import sim_scaling as SC

STRATEGIES = ["round_robin", "weighted", "least_load", "least_work", "power_of_two", "predictive", "rl",
              "marginal_cost"]
WARMUP_DECISIONS = 500


class Timed:
    """Passes every call to the strategy module and times choose_gpu."""

    def __init__(self, mod):
        self.mod = mod
        self.ns = []
        self.choices = []

    def choose_gpu(self, request, gpu_states):
        choose = self.mod.choose_gpu
        t0 = time.perf_counter_ns()
        g = choose(request, gpu_states)
        self.ns.append(time.perf_counter_ns() - t0)
        self.choices.append(g)
        return g

    def __getattr__(self, name):                       # reset, on_complete, PREDICTOR, ...
        return getattr(self.mod, name)


class _Empty:
    @staticmethod
    def choose_gpu(request, gpu_states):
        return next(iter(gpu_states))


def cpu_name():
    try:
        if sys.platform == "win32":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def pct(v, q):
    return SC.pct(v, q)


def floor_us(k=3, n=20000):
    """Median time of an empty strategy call through the timing code (measurement floor)."""
    nodes, _ = SC.cluster(k)
    states = {name: None for name in nodes}
    t = Timed(_Empty())
    for _ in range(n):
        t.choose_gpu(None, states)
    return statistics.median(t.ns) / 1000


def run_once(name, k, rho, decisions, pred):
    """One simulated run; returns the choices and the time (ns) of every measured decision."""
    nodes, _ = SC.cluster(k)
    jobs = SC.workload(rho * SC.sat_rate(k), f"decision-{k}-{rho}", n=WARMUP_DECISIONS + decisions)
    mod = SC.load_strategy(name, pred)
    if name == "marginal_cost":
        mod.LAMBDA = SC.MC_LAMBDA                       # on the module: simulate() would set it on the wrapper
    timed = Timed(mod)
    gc.collect()
    gc.disable()
    try:
        S.simulate(timed, jobs, nodes)
    finally:
        gc.enable()
    ns = timed.ns[WARMUP_DECISIONS:WARMUP_DECISIONS + decisions]
    assert len(ns) == decisions, (name, k, len(ns))
    return timed.choices, ns


def measure(configs, decisions, pred, repeats=1, log=print):
    """Rounds of runs; per configuration the fastest repeat of every decision."""
    best, first, choices = {}, {}, {}
    for rnd in range(repeats):
        t0 = time.time()
        for cfg in configs:
            ch, ns = run_once(*cfg, decisions, pred)
            if rnd == 0:
                choices[cfg], first[cfg], best[cfg] = ch, ns, list(ns)
            else:
                assert ch == choices[cfg], f"{cfg}: repeated runs made different choices"
                best[cfg] = [min(a, b) for a, b in zip(best[cfg], ns)]
        log(f"round {rnd + 1}/{repeats} done ({time.time() - t0:.0f} s)")
    rows = []
    for (name, k, rho) in configs:
        us = [x / 1000 for x in best[(name, k, rho)]]
        one = [x / 1000 for x in first[(name, k, rho)]]
        rows.append({"strategy": name, "nodes": k, "rho": rho, "decisions": decisions, "repeats": repeats,
                     "median_us": statistics.median(us), "p90_us": pct(us, .90), "p99_us": pct(us, .99),
                     "mean_us": statistics.mean(us), "max_us": max(us),
                     "single_run_median_us": statistics.median(one), "single_run_p99_us": pct(one, .99)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, nargs="+", default=[3, 8, 16])
    ap.add_argument("--rho", type=float, nargs="+", default=[0.6, 0.9])
    ap.add_argument("--strategies", nargs="+", default=STRATEGIES)
    ap.add_argument("--decisions", type=int, default=10000)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "results" / "overhead" / "decision_time.csv"))
    a = ap.parse_args()
    info = {"cpu": cpu_name(), "python": platform.python_version(),
            "timer_resolution_us": time.get_clock_info("perf_counter").resolution * 1e6, "timer_floor_us": floor_us()}
    print(f"CPU {info['cpu']}, Python {info['python']}, timer resolution {info['timer_resolution_us']:.3f} us, "
          f"empty call {info['timer_floor_us']:.3f} us", flush=True)
    pred = SC.length_predictor()
    configs = [(name, k, rho) for name in a.strategies for k in a.nodes for rho in a.rho]
    rows = [dict(r, **info) for r in measure(configs, a.decisions, pred, a.repeats, log=lambda m: print(m, flush=True))]
    for r in rows:
        print(f"{r['strategy']:14s} {r['nodes']:2d} nodes  rho {r['rho']}  median {r['median_us']:7.2f} us  "
              f"p99 {r['p99_us']:7.2f} us  (single run: median {r['single_run_median_us']:7.2f}, "
              f"p99 {r['single_run_p99_us']:7.2f})", flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
