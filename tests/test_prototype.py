"""Tests for the static weighted and least-load strategies and the energy attribution.

    python tests/test_prototype.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strategies.base import Request, GpuState
from strategies import weighted, least_load
from infra.attribute_energy import attribute
from router.router import parse_metrics


def st(q=0, n=0, **extra):
    return GpuState(queue_len=q, running=n, extra=extra)


R = Request("r", 10)


def test_weighted_follows_speed_ratio():
    weighted.reset()
    states = {"fast": st(speed_tps=30), "slow": st(speed_tps=10)}
    picks = [weighted.choose_gpu(R, states) for _ in range(400)]
    assert picks.count("fast") == 300 and picks.count("slow") == 100


def test_weighted_ignores_load():
    weighted.reset()
    states = {"a": st(q=50, speed_tps=1), "b": st(speed_tps=1)}
    picks = [weighted.choose_gpu(R, states) for _ in range(10)]
    assert picks.count("a") == 5


def test_weighted_without_speeds_is_round_robin():
    weighted.reset()
    states = {"a": st(), "b": st(), "c": st()}
    assert [weighted.choose_gpu(R, states) for _ in range(6)] == ["a", "b", "c", "a", "b", "c"]


def test_weighted_skips_unreachable():
    weighted.reset()
    states = {"a": st(q=10**6, speed_tps=50), "b": st(speed_tps=1)}
    assert {weighted.choose_gpu(R, states) for _ in range(5)} == {"b"}


def test_least_load_picks_fewest():
    assert least_load.choose_gpu(R, {"a": st(q=2, n=1), "b": st(n=2)}) == "b"


def test_least_load_tie_goes_to_faster():
    assert least_load.choose_gpu(R, {"a": st(n=1, speed_tps=10), "b": st(n=1, speed_tps=40)}) == "b"


def test_least_load_uses_router_inflight():
    # node reports 0 but the router has just sent it 3 requests
    assert least_load.choose_gpu(R, {"a": st(inflight=3), "b": st(n=1)}) == "b"


def test_least_load_avoids_unreachable():
    assert least_load.choose_gpu(R, {"a": st(q=10**6), "b": st(q=5, n=5)}) == "b"


def test_parse_metrics():
    text = "# HELP x\nllamacpp:requests_processing 2\nllamacpp:requests_deferred 3\nbad line here\n"
    m = parse_metrics(text)
    assert m["llamacpp:requests_processing"] == 2 and m["llamacpp:requests_deferred"] == 3


def samples(watts, t0=0.0, t1=10.0):
    return [(float(t), watts) for t in range(int(t0), int(t1) + 1)]


def test_energy_total_and_idle():
    reqs = [{"a": 0, "f": 1, "e": 10, "n": 90}]
    e, total, idle = attribute(reqs, samples(100), "tokens", 5)
    assert abs(total - 1000) < 1e-6 and idle == 0 and abs(e[0] - 1000) < 1e-6


def test_energy_equal_split_two_requests():
    reqs = [{"a": 0, "f": 0.5, "e": 10, "n": 10}, {"a": 0, "f": 0.5, "e": 10, "n": 90}]
    e, total, _ = attribute(reqs, samples(100), "equal", 5)
    assert abs(e[0] - 500) < 1e-6 and abs(e[1] - 500) < 1e-6


def test_energy_tokens_split_follows_tokens():
    reqs = [{"a": 0, "f": 0, "e": 10, "n": 10}, {"a": 0, "f": 0, "e": 10, "n": 90}]
    e, total, _ = attribute(reqs, samples(100), "tokens", 5)
    assert abs(e[0] - 100) < 1e-6 and abs(e[1] - 900) < 1e-6


def test_energy_idle_gap_not_attributed():
    reqs = [{"a": 0, "f": 1, "e": 2, "n": 5}, {"a": 8, "f": 9, "e": 10, "n": 5}]
    e, total, idle = attribute(reqs, samples(100), "tokens", 5)
    assert abs(total - 1000) < 1e-6 and abs(idle - 600) < 1e-6 and abs(sum(e) - 400) < 1e-6


def test_energy_conserved():
    reqs = [{"a": 0, "f": 2, "e": 7, "n": 40}, {"a": 1, "f": 1.5, "e": 9.5, "n": 120}, {"a": 3, "f": 6, "e": 10, "n": 3}]
    for m in ("tokens", "equal"):
        e, total, idle = attribute(reqs, samples(70), m, 5)
        assert abs(sum(e) + idle - total) < 1e-6


def test_energy_run_span_counts_idle_edges():
    # node busy only from 4 to 6 s of a 0 to 10 s run: the rest of the run is idle energy
    reqs = [{"a": 4, "f": 4, "e": 6, "n": 10}]
    e, total, idle = attribute(reqs, samples(100), "tokens", 5, span=(0, 10))
    assert abs(total - 1000) < 1e-6 and abs(idle - 800) < 1e-6 and abs(e[0] - 200) < 1e-6


def test_energy_node_without_requests_is_all_idle():
    e, total, idle = attribute([], samples(50), "tokens", 5, span=(0, 10))
    assert e == [] and abs(total - 500) < 1e-6 and abs(idle - 500) < 1e-6


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, fn in tests:
        fn()
        print("ok  ", name)
    print(len(tests), "tests passed")
