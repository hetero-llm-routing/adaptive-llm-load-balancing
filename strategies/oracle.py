"""Oracle: the predictive rule (shortest expected completion time) with the true output length
of every request instead of a prediction. It is a reference, not a deployable strategy: it
shows how much of the gap to the best possible routing comes from errors in the predicted length.

For node j with n running requests whose remaining output r_i is known exactly,
    T_j = W_j + P / p_j + L / v_j(min(n+1, 4)),   v_j(n) = s_j(n) / n   (A5)
where W_j is the time until a slot frees when all four are busy. The node with the smallest T_j
is chosen. The true length comes from Request.true_output_len (the router fills it from
max_tokens when the experiment forces the output length); without it the strategy falls back to
the length predictor.

Interface (strategies/base.py): choose_gpu(request, gpu_states) -> node name, on_complete(...)
"""
import os, time
from pathlib import Path

from strategies.length_predictor import LengthPredictor

ROOT = Path(__file__).resolve().parents[1]
UNREACHABLE = 10**6
SLOTS = 4
PREDICTOR = LengthPredictor.load(os.environ.get("LB_LENGTH_TABLE", ROOT / "strategies" / "length_table.json"))
_running = {}      # node -> {request_id: [start, prompt_len, true_len]}


def reset():
    _running.clear()


def _speed(ex, n):
    c = (ex or {}).get("speed_by_n")
    if c:
        return float(c[min(max(n, 1), len(c)) - 1])
    return float((ex or {}).get("speed_tps") or 30.0)


def _length(request):
    return float(request.true_output_len) if request.true_output_len else PREDICTOR.predict(request.prompt_len, request.req_type)


def completion_time(request, name, st, now):
    ex = st.extra or {}
    p = float(ex.get("prefill_tps") or 1000.0)
    reqs = _running.get(name, {})
    n_total = max(len(reqs), int(st.running) + int(st.queue_len))
    n = min(n_total, SLOTS)
    v = _speed(ex, max(n, 1)) / max(n, 1)
    rem = sorted(max(0.0, L - max(0.0, now - t0 - P / p) * v) for t0, P, L in reqs.values())
    L = _length(request)
    if n < SLOTS:
        return request.prompt_len / p + L / (_speed(ex, n + 1) / (n + 1))
    q = n_total - SLOTS
    v4 = _speed(ex, SLOTS) / SLOTS
    wait = (rem[min(q, len(rem) - 1)] / v4 if rem else L / v4) * (1 + q // SLOTS)
    return wait + request.prompt_len / p + L / v4


def choose_gpu(request, gpu_states):
    now = request.arrival_ts or time.time()
    t = {n: completion_time(request, n, s, now) for n, s in gpu_states.items() if s.queue_len < UNREACHABLE}
    if not t:
        return sorted(gpu_states)[0]
    best = min(sorted(t), key=t.get)
    _running.setdefault(best, {})[request.request_id] = [now, request.prompt_len, _length(request)]
    return best


def on_complete(request_id, gpu, ttft_ms, e2e_ms, output_len):
    _running.get(gpu, {}).pop(request_id, None)
