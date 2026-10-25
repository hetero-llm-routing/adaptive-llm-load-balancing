"""Tests for the predictive strategy and the output-length predictor.

Run from the repository root:  python -m pytest tests/ -q
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strategies.base import Request, GpuState
from strategies import predictive
from strategies.length_predictor import LengthPredictor, bucket


def nodes(fast=None, medium=None, slow=None):
    """Three nodes with the mock-cluster speeds (60, 30, 10 tokens/s)."""
    def st(speed, s):
        s = s or GpuState()
        s.extra = {"speed_tps": speed, "prefill_tps": 600}
        return s
    return {"fast": st(60, fast), "medium": st(30, medium), "slow": st(10, slow)}


def req(t=1000.0, prompt=50, kind="chat"):
    return Request(request_id="r", prompt_len=prompt, req_type=kind, arrival_ts=t)


def setup_function(_):
    predictive.reset()
    predictive.PREDICTOR = LengthPredictor({"global_mean": 100.0})


def test_idle_cluster_goes_to_fastest_node():
    assert predictive.choose_gpu(req(), nodes()) == "fast"


def test_busy_fast_node_sends_work_elsewhere():
    # fast node: 5 requests ahead (~500 tokens / 60 = 8.3 s); medium idle (100 / 30 = 3.3 s)
    assert predictive.choose_gpu(req(), nodes(fast=GpuState(queue_len=3, running=2))) == "medium"


def test_backlog_spreads_a_burst_by_speed():
    # 9 requests at the same moment: shares should follow speed 60 : 30 : 10
    counts = {"fast": 0, "medium": 0, "slow": 0}
    for _ in range(9):
        counts[predictive.choose_gpu(req(), nodes())] += 1
    assert counts["fast"] > counts["medium"] > counts["slow"]
    assert counts["fast"] == 6 and counts["medium"] == 3 and counts["slow"] == 0


def test_backlog_drains_over_time():
    for _ in range(3):
        predictive.choose_gpu(req(t=1000.0), nodes())
    # 60 s later every node has finished its work, so the fastest wins again
    assert predictive.choose_gpu(req(t=1060.0), nodes()) == "fast"


def test_unreachable_node_is_skipped():
    assert predictive.choose_gpu(req(), nodes(fast=GpuState(queue_len=10**6))) == "medium"


def test_speeds_fall_back_to_config_yaml():
    # no extra fields: speeds come from config.yaml (fast 60, medium 30, slow 10)
    states = {"fast": GpuState(), "medium": GpuState(), "slow": GpuState()}
    assert predictive.choose_gpu(req(), states) == "fast"


def test_long_prompt_counts_prefill_time():
    times, _ = predictive.estimate(req(prompt=6000), nodes())
    assert times["fast"] - times["medium"] < 0           # still faster overall
    assert times["fast"] >= 6000 / 600                    # includes 10 s of prefill


def test_decision_is_cheap():
    states = nodes()
    n, start = 2000, time.perf_counter()
    for i in range(n):
        predictive.choose_gpu(req(t=1000.0 + i * 0.05), states)
    per_call_us = (time.perf_counter() - start) / n * 1e6
    print(f"predictive decision: {per_call_us:.1f} us per request")
    assert per_call_us < 1000                             # well under 1 ms


# ---------- length predictor ----------

def test_buckets():
    assert [bucket(x) for x in (10, 32, 33, 500, 1729, 5000)] == [0, 0, 1, 2, 3, 4]


def test_fit_learns_per_bucket_means_with_shrinkage():
    rows = [{"prompt_len": 14, "output_len": 120}] * 20 + [{"prompt_len": 1729, "output_len": 60}] * 20
    p = LengthPredictor.fit(rows)
    assert abs(p.global_mean - 90) < 1e-9
    short, long_ = p.predict(14), p.predict(1729)
    assert 110 < short < 120 and 60 < long_ < 70          # pulled a little toward 90
    assert p.predict(300) == p.global_mean                # unseen bucket -> global mean


def test_request_type_is_used_when_present():
    rows = ([{"prompt_len": 1000, "output_len": 200, "req_type": "rag"}] * 30 +
            [{"prompt_len": 1000, "output_len": 50, "req_type": "chat"}] * 30)
    p = LengthPredictor.fit(rows)
    assert p.predict(1000, "rag") > p.predict(1000, "chat")


def test_untrained_predictor_uses_default():
    assert LengthPredictor().predict(50) == 128.0


if __name__ == "__main__":          # also runs without pytest:  python tests/test_predictive.py
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for n, f in tests:
        setup_function(None)
        f()
        print("ok  ", n)
    print(f"{len(tests)} tests passed")
