"""Marginal-cost routing (Eyad): send each request to the node where it adds the least to the
total latency of all requests in the cluster, plus lambda times the energy it adds.

Dispatching by marginal cost is known from queueing theory (Hyytiae et al., first policy
iteration for parallel servers); this strategy applies it to LLM serving on heterogeneous GPUs,
with costs computed from the measured profile of each node (assumption A5) instead of a model:

  s_j(n)   total decode speed of node j with n running requests (speed_by_n, n = 1..4),
           so each running request decodes at v_j(n) = s_j(n) / n
  p_j      prompt (prefill) speed
  P_j(n)   mean power with n running requests (power_by_n), P_j(0) = idle power

For a new request with prompt length P and predicted output length L, and node j with n
running requests (remaining tokens r_i, tracked by this strategy) and q waiting:

  own time      T_j = W_j + P / p_j + L / v_j(n+1)
                W_j = wait for a free slot when n >= slots (FIFO): the (q+1)-th earliest
                      finish among the running requests, approximated from r_i
  slowdown      D_j = sum_i [ P / p_j  +  min(r_i, L) * (1 / v_j(n+1) - 1 / v_j(n)) ]
                      (the prompt pauses the running requests; then they share the GPU)
  energy        E_j = (P_j(n+1) - P_j(n)) * min(T_j, R_j) + (P_j(1) - P_j(0)) * max(0, T_j - R_j)
                      R_j = time until the node would be idle without the request; the idle
                      power is spent anyway, so only the extra power counts
  cost          C_j = T_j + D_j + lambda * E_j                     (seconds; lambda in s/J)

lambda = 0 minimizes latency only; a larger lambda trades latency for energy (LB_MC_LAMBDA or
router.mc_lambda in config.yaml). With the measured profiles the energy term favors a node that
is already busy (a fourth request adds about 20 W on the RTX 4090) or the RTX 2000 Ada (about
60 W from idle) over waking an idle RTX 4090 (about 260 W).

Interface (strategies/base.py): choose_gpu(request, gpu_states) -> node name, plus
on_complete(request_id, gpu, ttft_ms, e2e_ms, output_len), called by the router.
"""
import os, time
from pathlib import Path

from strategies.length_predictor import LengthPredictor

ROOT = Path(__file__).resolve().parents[1]
UNREACHABLE = 10**6
SLOTS = int(os.environ.get("LB_SLOTS", "4"))   # slots per server; 3 in the case study (Qwen3-14B)
PREDICTOR = LengthPredictor.load(os.environ.get("LB_LENGTH_TABLE", ROOT / "strategies" / "length_table.json"))
LAMBDA = float(os.environ.get("LB_MC_LAMBDA", "0.002") or 0)   # fixed on the simulator (largest value within 10% of the fastest p95 at every load)

_running = {}        # node -> {request_id: [start_ts, prompt_len, predicted_len]}


def reset():
    _running.clear()


def _curve(extra, key, n, default=None):
    c = (extra or {}).get(key)
    if not c:
        return default
    return float(c[min(max(n, 1), len(c)) - 1])


def _speed(extra, n):
    """Total decode speed with n running requests; above the measured range: s(4) (A5)."""
    s = _curve(extra, "speed_by_n", n)
    if s is None:
        s = float((extra or {}).get("speed_tps") or 30.0) * max(n, 1) ** 0.5
    return s


def _power(extra, n):
    if n <= 0:
        return float((extra or {}).get("idle_w") or 0.0)
    return _curve(extra, "power_by_n", n, 0.0)


def _remaining(node, extra, now):
    """Remaining output tokens of the requests this router has running on the node."""
    reqs = _running.get(node, {})
    n = max(1, min(len(reqs), SLOTS))
    v = _speed(extra, n) / n
    p = float((extra or {}).get("prefill_tps") or 1000.0)
    out = []
    for start, plen, L in reqs.values():
        done = max(0.0, now - start - plen / p) * v
        out.append(max(0.1 * L, L - done))
    return sorted(out)


def cost(request, name, st, now, lam=None):
    lam = LAMBDA if lam is None else lam
    ex = st.extra or {}
    L = PREDICTOR.predict(request.prompt_len, request.req_type)
    P = float(request.prompt_len)
    p = float(ex.get("prefill_tps") or 1000.0)
    rem = _remaining(name, ex, now)
    n_total = max(len(rem), int(st.running) + int(st.queue_len))
    n = min(n_total, SLOTS)                       # running requests
    q = max(0, n_total - SLOTS)                   # waiting requests
    v_now = _speed(ex, n) / n if n else 0.0
    if n < SLOTS:                                 # a slot is free: start now, share the GPU
        v_new = _speed(ex, n + 1) / (n + 1)
        wait = 0.0
        run = rem[:n]
        slow = sum(P / p + min(r, L) * (1.0 / v_new - 1.0 / v_now) for r in run) if n else 0.0
        n_after = n + 1
    else:                                         # all slots busy: wait for the (q+1)-th finish
        v_full = _speed(ex, SLOTS) / SLOTS
        k = min(q, len(rem) - 1) if rem else 0
        wait = (rem[k] if rem else L) / v_full * (1 + q // SLOTS)
        v_new = v_full
        slow = 0.0                                # FIFO: requests ahead are not delayed
        n_after = SLOTS
    T = wait + P / p + L / v_new
    R = (max(rem) / v_now) if rem and v_now else 0.0    # time until the node would be idle
    dP_busy = _power(ex, n_after) - _power(ex, max(n_after - 1, 0)) if n else 0.0
    dP_wake = _power(ex, 1) - _power(ex, 0)
    energy = max(0.0, dP_busy) * min(T, R) + max(0.0, dP_wake) * max(0.0, T - R)
    return T + slow + lam * energy, {"T": T, "slow": slow, "energy_j": energy, "L": L}


def choose_gpu(request, gpu_states):
    now = request.arrival_ts or time.time()
    costs = {name: cost(request, name, st, now)[0]
             for name, st in gpu_states.items() if st.queue_len < UNREACHABLE}
    if not costs:
        return sorted(gpu_states)[0]
    best = min(sorted(costs), key=costs.get)
    _running.setdefault(best, {})[request.request_id] = [now, request.prompt_len,
                                                         PREDICTOR.predict(request.prompt_len, request.req_type)]
    return best


def on_complete(request_id, gpu, ttft_ms, e2e_ms, output_len):
    _running.get(gpu, {}).pop(request_id, None)
