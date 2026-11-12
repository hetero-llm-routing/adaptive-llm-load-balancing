"""Tests for the strategies of the Phase 5 runs (least_work, power_of_two, oracle, rl, A5 in
predictive and mock_sim) and for the scaling and decision-time scripts.

Run from the repository root:  python tests/test_phase5_strategies.py   (or python -m pytest tests/ -q)
"""
import csv, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from strategies.base import Request, GpuState
from strategies import least_work, power_of_two, oracle, rl, predictive, round_robin, marginal_cost
from strategies.length_predictor import LengthPredictor
from strategies.mock_sim import Node
import sim_scaling as SC
import decision_time as DT

UNREACHABLE = 10**6


def req(i="r", prompt=50, kind="chat", t=1000.0, true_len=None):
    return Request(request_id=str(i), prompt_len=prompt, req_type=kind, arrival_ts=t, true_output_len=true_len)


def testbed(running=None, queue=None):
    """The three testbed nodes with their measured profiles (A5)."""
    nodes, _ = SC.cluster(3)
    running, queue = running or {}, queue or {}
    return {n: GpuState(queue_len=queue.get(n, 0), running=running.get(n, 0), extra=dict(p)) for n, p in nodes.items()}


def setup_function(_):
    for mod in (power_of_two, oracle, rl, predictive, round_robin):
        mod.reset()
    predictive.PREDICTOR = LengthPredictor({"global_mean": 100.0})
    oracle.PREDICTOR = LengthPredictor({"global_mean": 100.0})


# ---------- least work ----------

def test_least_work_idle_cluster_goes_to_fastest_node():
    assert least_work.choose_gpu(req(), testbed()) == "rtx4090"


def test_least_work_divides_load_by_speed():
    # RTX 4090 with 3 running: 4 / 450 tok/s; idle RTX A6000: 1 / 115 tok/s -> the A6000 is less work
    assert least_work.choose_gpu(req(), testbed(running={"rtx4090": 3})) == "rtxa6000"


def test_least_work_skips_unreachable_node():
    assert least_work.choose_gpu(req(), testbed(queue={"rtx4090": UNREACHABLE})) == "rtxa6000"


# ---------- power of two choices ----------

def test_power_of_two_picks_less_loaded_of_two():
    states = {"a": GpuState(running=4, extra={"speed_tps": 100}), "b": GpuState(running=1, extra={"speed_tps": 10})}
    assert all(power_of_two.choose_gpu(req(i), states) == "b" for i in range(20))


def test_power_of_two_is_repeatable_after_reset():
    states, _ = SC.cluster(16)
    states = {n: GpuState(running=i % 4, extra=p) for i, (n, p) in enumerate(states.items())}
    first = [power_of_two.choose_gpu(req(i), states) for i in range(50)]
    power_of_two.reset()
    assert [power_of_two.choose_gpu(req(i), states) for i in range(50)] == first


def test_power_of_two_never_picks_unreachable_node():
    states = testbed(queue={"rtx4090": UNREACHABLE, "rtxa6000": UNREACHABLE})
    assert all(power_of_two.choose_gpu(req(i), states) == "rtx2000ada" for i in range(20))


# ---------- oracle ----------

def test_oracle_uses_true_length():
    # RTX 4090 with 4 running: a long request waits for a slot there, so the idle RTX A6000 wins;
    # with the true length unknown the predictor's 100 tokens are used instead
    states = testbed(running={"rtx4090": 4})
    assert oracle.choose_gpu(req("long", true_len=2000), states) == "rtxa6000"
    assert oracle.completion_time(req("x", true_len=2000), "rtxa6000", states["rtxa6000"], 1000.0) > \
        oracle.completion_time(req("x"), "rtxa6000", states["rtxa6000"], 1000.0)


def test_oracle_forgets_completed_requests():
    g = oracle.choose_gpu(req("a", true_len=300), testbed())
    assert "a" in oracle._running[g]
    oracle.on_complete("a", g, 0, 0, 300)
    assert "a" not in oracle._running[g]


# ---------- RL ----------

def test_rl_policy_is_loaded():
    assert len(rl.THETA) == rl.N_FEAT and any(abs(x) > 0.1 for x in rl.THETA)


def test_rl_idle_cluster_goes_to_fastest_node():
    assert rl.choose_gpu(req(prompt=200), testbed()) == "rtx4090"


def test_rl_works_for_any_cluster_size():
    for k in (3, 8, 16):
        nodes, _ = SC.cluster(k)
        states = {n: GpuState(running=i % 5, extra=dict(p)) for i, (n, p) in enumerate(nodes.items())}
        assert rl.choose_gpu(req(), states) in nodes


def test_rl_skips_unreachable_node():
    assert rl.choose_gpu(req(), testbed(queue={"rtx4090": UNREACHABLE})) != "rtx4090"


# ---------- A5 in predictive and mock_sim ----------

def test_predictive_uses_speed_by_concurrency():
    # same single-request speed; node a gains a lot from batching, node b hardly at all
    a = GpuState(running=2, extra={"speed_tps": 100, "prefill_tps": 1e9, "speed_by_n": [100, 200, 400, 400]})
    b = GpuState(running=0, extra={"speed_tps": 100, "prefill_tps": 1e9, "speed_by_n": [100, 110, 115, 120]})
    # a: (200 + 100) / s_a(3) = 0.75 s;  b: 100 / s_b(1) = 1 s
    assert predictive.choose_gpu(req(), {"a": a, "b": b}) == "a"
    predictive.reset()
    for s in (a, b):
        del s.extra["speed_by_n"]                      # without A5: a takes 3 s, b 1 s
    assert predictive.choose_gpu(req(), {"a": a, "b": b}) == "b"


def test_mock_sim_rate_follows_speed_by_concurrency():
    n = Node(name="x", speed_tps=10, speed_by_n=[10, 16, 18, 20])
    n.running = [object(), object()]
    assert n.rate() == 8.0
    n.speed_by_n = None
    assert n.rate() == 5.0                             # without A5 the speed is shared


# ---------- defaults of the real runs ----------

def test_round_robin_reset_starts_from_first_node():
    states = testbed()
    round_robin.choose_gpu(req(), states)
    round_robin.reset()
    assert round_robin.choose_gpu(req(), states) == sorted(states)[0]


def test_marginal_cost_default_lambda_is_that_of_the_real_runs():
    if os.environ.get("LB_MC_LAMBDA") is None:
        assert marginal_cost.LAMBDA == 0.002


# ---------- scaling simulation and decision time ----------

def test_cluster_mixes():
    expected = {3: {"m111"}, 8: {"m332", "m233", "m323"}, 16: {"m655", "m565", "m556"}}
    for k, mixes in expected.items():
        assert {SC.mix_name(SC.cluster(k, r)[1]) for r in SC.rotations(k)} == mixes
        assert all(len(SC.cluster(k, r)[0]) == k for r in SC.rotations(k))


def test_saturation_rate_grows_with_cluster_size():
    assert abs(SC.sat_rate(3) - SC.testbed_sat_rate()) < 1e-12
    assert abs(SC.sat_rate(16) / SC.sat_rate(8) - 2.0) < 1e-12


def test_simulated_run_has_the_columns_of_the_real_runs():
    columns = SC.header(ROOT / "results" / "phase5" / "runs.csv")
    nodes, kind = SC.cluster(8)
    jobs = SC.workload(SC.sat_rate(8) * 0.6, "test", duration=60.0)
    res = SC.S.simulate(SC.load_strategy("least_load", SC.length_predictor()), jobs, nodes, keep=True)
    row = SC.run_row(columns, "sim8", 0.6, 1, "least_load", res, kind)
    assert list(row) == columns
    assert abs(sum(row[f"share_{t}"] for t in SC.TYPES) - 1.0) < 1e-9
    assert row["n"] > 0 and row["e2e_p95_ms"] >= row["e2e_p50_ms"] > 0


def test_decision_time_counts_the_requested_decisions():
    rows = DT.measure([("least_load", 3, 0.6), ("marginal_cost", 8, 0.6)], 200, SC.length_predictor(), repeats=2,
                      log=lambda m: None)
    for r in rows:
        assert r["decisions"] == 200 and r["repeats"] == 2
        assert 0 < r["median_us"] <= r["p99_us"] <= r["max_us"]
        assert r["median_us"] <= r["single_run_median_us"] + 1e-9      # the minimum of repeats is never slower


if __name__ == "__main__":          # also runs without pytest:  python tests/test_phase5_strategies.py
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for n, f in tests:
        setup_function(None)
        f()
        print("ok  ", n)
    print(f"{len(tests)} tests passed")
