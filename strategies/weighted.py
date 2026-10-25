"""Static weighted partitioning: each node receives a fixed share of the requests,
proportional to its measured decode speed (speed_tps), whatever its current load.

The order is smooth weighted round-robin (the method used by nginx): with weights
3:1, the sequence is A A B A A A B A ..., spread out rather than in blocks. The
weights are fixed by config.yaml before the run; the strategy never reads the
queue or running counts, which is what makes it static.

A node without speed_tps gets weight 1, so with no speeds the strategy is plain
round-robin. A node that does not answer (queue_len = 10**6) is skipped.

Interface (strategies/base.py):  choose_gpu(request, gpu_states) -> node name
"""
UNREACHABLE = 10**6

_current = {}          # node -> running score of smooth weighted round-robin


def reset():
    """Start a new sequence (between experiment runs or in tests)."""
    _current.clear()


def weights(gpu_states):
    return {n: float((s.extra or {}).get("speed_tps") or 1.0) for n, s in gpu_states.items()}


def choose_gpu(request, gpu_states):
    w = {n: v for n, v in weights(gpu_states).items() if gpu_states[n].queue_len < UNREACHABLE}
    if not w:                                   # every node unreachable: keep the router going
        return sorted(gpu_states)[0]
    total = sum(w.values())
    for n in w:
        _current[n] = _current.get(n, 0.0) + w[n]
    best = max(sorted(w), key=lambda n: _current[n])
    _current[best] -= total
    return best
