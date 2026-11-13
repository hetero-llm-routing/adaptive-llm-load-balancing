"""Baseline: send requests to the nodes in turn, ignoring their state."""
import itertools

_counter = itertools.count()


def reset():
    """Start again from the first node (between simulated runs, like a restarted router)."""
    global _counter
    _counter = itertools.count()


def choose_gpu(request, gpu_states):
    names = sorted(gpu_states)
    return names[next(_counter) % len(names)]
