"""Dynamic least work (shortest expected delay): send each request to the node where it is
expected to wait and run the shortest time, using the node's measured speed but no prediction
of the request's own length.

    choose  argmin_j  (n_j + 1) / s_j(min(n_j + 1, 4))

n_j is the number of requests on node j (waiting + running, or the router's in-flight count if
larger) and s_j(n) the node's total decode speed with n running requests (speed_by_n, A5);
without speed_by_n, s_j = speed_tps. Every request is assumed to have the same length, so this
is the speed-aware version of least load; the predictive strategy adds the request's predicted
length and the work already queued.

Interface (strategies/base.py): choose_gpu(request, gpu_states) -> node name
"""
UNREACHABLE = 10**6


def load(state):
    return max(state.queue_len + state.running, int((state.extra or {}).get("inflight", 0)))


def speed(state, n):
    ex = state.extra or {}
    c = ex.get("speed_by_n")
    if c:
        return float(c[min(max(n, 1), len(c)) - 1])
    return float(ex.get("speed_tps") or 1.0)


def choose_gpu(request, gpu_states):
    ok = {n: s for n, s in gpu_states.items() if s.queue_len < UNREACHABLE}
    if not ok:
        return sorted(gpu_states)[0]
    return min(sorted(ok), key=lambda n: (load(ok[n]) + 1) / speed(ok[n], load(ok[n]) + 1))
