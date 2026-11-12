"""Power of two choices [11]: pick two nodes at random and send the request to the less loaded
of the two (waiting + running requests, or the router's in-flight count if larger); ties go to
the faster node. Random choices are seeded (LB_SEED), so a run can be repeated.

With few nodes the rule is close to least load; its point is the low decision cost on large
clusters, where reading every node's state is expensive (RQ3).

Interface (strategies/base.py): choose_gpu(request, gpu_states) -> node name
"""
import os, random

UNREACHABLE = 10**6
_rng = random.Random(int(os.environ.get("LB_SEED", "0") or 0))


def reset(seed=None):
    global _rng
    _rng = random.Random(int(os.environ.get("LB_SEED", "0") or 0) if seed is None else seed)


def load(state):
    return max(state.queue_len + state.running, int((state.extra or {}).get("inflight", 0)))


def choose_gpu(request, gpu_states):
    ok = sorted(n for n, s in gpu_states.items() if s.queue_len < UNREACHABLE)
    if not ok:
        return sorted(gpu_states)[0]
    pair = _rng.sample(ok, 2) if len(ok) >= 2 else ok
    return min(sorted(pair), key=lambda n: (load(gpu_states[n]), -float((gpu_states[n].extra or {}).get("speed_tps") or 0.0)))
