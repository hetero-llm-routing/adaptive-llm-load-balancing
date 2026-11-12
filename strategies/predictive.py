"""Predictive strategy (Osama): send each request to the node expected to finish it first.

For every request the strategy
  1. predicts its output length L from the prompt length and request type
     (strategies/length_predictor.py),
  2. estimates, for every node j, the work already waiting there in tokens (B_j),
  3. picks the node with the smallest expected completion time

        T_j = (B_j + L) / s_j(n_j + 1)  +  P / p_j

     where s_j(n) is the node's total decode speed (tokens/s) with n running
     requests (speed_by_n, assumption A5; capped at n = 4; speed_tps if the
     curve is not configured), n_j the requests on the node, P the prompt
     length and p_j the node's prefill speed (tokens/s).

How B_j is kept up to date without any feedback from the router:
  * When a request is sent to node j, its predicted length L is added to B_j.
  * Between decisions, each node drains s_j tokens per second, so B_j shrinks
    by s_j * (elapsed seconds), never below zero.
  * The node's reported state corrects drift: B_j is at least
    (queue_len + running) * mean predicted length.

Node speeds come from GpuState.extra["speed_tps"] / ["prefill_tps"] if the router
fills them, otherwise from `speed_tps` / `prefill_tps` of the node in config.yaml
(measured once per GPU type), otherwise from the defaults below.

Interface (strategies/base.py):  choose_gpu(request, gpu_states) -> node name
"""
import os, time
from pathlib import Path

from strategies.length_predictor import LengthPredictor

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEED_TPS = 30.0      # decode tokens/s if nothing is configured
DEFAULT_PREFILL_TPS = 600.0   # prompt tokens/s if nothing is configured
UNREACHABLE = 10**6           # router sets queue_len to this when a node does not answer

PREDICTOR = LengthPredictor.load(os.environ.get("LB_LENGTH_TABLE", ROOT / "strategies" / "length_table.json"))

_backlog = {}                 # node -> predicted tokens still to generate
_last_ts = None               # time of the previous decision
_config_nodes = None


def _node_config():
    """speed_tps / prefill_tps per node from config.yaml (read once)."""
    global _config_nodes
    if _config_nodes is None:
        _config_nodes = {}
        try:
            import yaml
            cfg = yaml.safe_load(open(os.environ.get("LB_CONFIG", ROOT / "config.yaml")))
            _config_nodes = {n["name"]: n for n in cfg.get("nodes", [])}
        except Exception:
            pass
    return _config_nodes


def _speed(name, state, key, default):
    v = (state.extra or {}).get(key) or _node_config().get(name, {}).get(key)
    return float(v) if v else default


def _speed_n(name, state, n, fallback):
    """Total decode speed with n running requests (speed_by_n, A5), else the fixed speed."""
    c = (state.extra or {}).get("speed_by_n") or _node_config().get(name, {}).get("speed_by_n")
    if not c or n <= 0:
        return fallback
    return float(c[min(n, len(c)) - 1])


def reset():
    """Forget all backlog (call between experiment runs or in tests)."""
    global _last_ts
    _backlog.clear()
    _last_ts = None


def estimate(request, gpu_states):
    """Expected completion time (seconds) of the request on every reachable node."""
    global _last_ts
    now = request.arrival_ts or time.time()
    elapsed = 0.0 if _last_ts is None else max(0.0, now - _last_ts)
    _last_ts = now

    L = PREDICTOR.predict(request.prompt_len, request.req_type)
    out = {}
    for name, st in gpu_states.items():
        s = _speed(name, st, "speed_tps", DEFAULT_SPEED_TPS)
        p = _speed(name, st, "prefill_tps", DEFAULT_PREFILL_TPS)
        n = min(st.running + st.queue_len, 4) if st.queue_len < UNREACHABLE else 0
        s_now = _speed_n(name, st, n, s)              # A5: the node drains at its speed with n running
        s_new = _speed_n(name, st, n + 1, s)
        drained = max(0.0, _backlog.get(name, 0.0) - s_now * elapsed)
        observed = (st.queue_len + st.running) * PREDICTOR.global_mean
        _backlog[name] = max(drained, observed) if st.queue_len < UNREACHABLE else drained
        if st.queue_len >= UNREACHABLE:
            continue
        out[name] = (_backlog[name] + L) / s_new + request.prompt_len / p
    return out, L


def choose_gpu(request, gpu_states):
    times, L = estimate(request, gpu_states)
    if not times:                                   # every node unreachable: keep the router going
        return sorted(gpu_states)[0]
    best = min(sorted(times), key=times.get)        # sorted() makes ties deterministic
    _backlog[best] += L
    return best
