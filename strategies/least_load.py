"""Dynamic least-load: send each request to the node with the fewest requests
on it right now (waiting + running), read before every decision.

Ties go to the faster node (higher speed_tps), then to the first name, so the
choice is deterministic. A node that does not answer (queue_len = 10**6) is never
chosen while another node answers.

This is the plain count-based rule. It does not divide by node speed, so on a
mixed cluster it still sends a fast and a slow node the same number of
concurrent requests; the predictive strategy is the one that uses speed.

Interface (strategies/base.py):  choose_gpu(request, gpu_states) -> node name
"""


def load(state):
    # The node's own counts lag by a moment when many requests arrive together; the
    # router's count of requests it has sent and not yet finished (extra["inflight"])
    # does not, so the larger of the two is used.
    return max(state.queue_len + state.running, int((state.extra or {}).get("inflight", 0)))


def choose_gpu(request, gpu_states):
    return min(sorted(gpu_states),
               key=lambda n: (load(gpu_states[n]), -float((gpu_states[n].extra or {}).get("speed_tps") or 0.0)))
