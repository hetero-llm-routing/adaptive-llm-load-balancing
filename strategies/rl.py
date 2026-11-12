"""Reinforcement-learning strategy: a softmax policy over the nodes, trained on the calibrated
simulator with REINFORCE (scripts/train_rl.py) and used greedily in the router.

State: for every node j, a feature vector built from the router's view and the measured profile
(A5), together with the request:
    phi_j = [1, n_j/4, q_j/4, P/p_j, L/v_j(n_j+1), L/s_j(1), 1{n_j = 0}, (P_j(n_j+1) - P_j(n_j))/100]
n_j running (capped at 4) and q_j waiting requests, P prompt length, L predicted output length,
p_j prompt speed, v_j(n) = s_j(n)/n per-request decode speed, P_j(n) measured power (idle at 0).
Action: the node. Score theta . phi_j, the same weights for every node, so one policy works for
any number of nodes (3 on the real cluster, 8 and 16 in simulation).
Reward: minus the request's end-to-end latency (s) minus lambda times its share of the energy,
centred per episode (REINFORCE with a baseline).

Weights: strategies/rl_policy.json (LB_RL_POLICY). In training (TRAIN = True) the action is
sampled and the features of each decision are kept for the update.

Interface (strategies/base.py): choose_gpu(request, gpu_states) -> node name, on_complete(...)
"""
import json, math, os, random
from pathlib import Path

from strategies.length_predictor import LengthPredictor

ROOT = Path(__file__).resolve().parents[1]
UNREACHABLE = 10**6
PREDICTOR = LengthPredictor.load(os.environ.get("LB_LENGTH_TABLE", ROOT / "strategies" / "length_table.json"))
POLICY = Path(os.environ.get("LB_RL_POLICY", ROOT / "strategies" / "rl_policy.json"))
N_FEAT = 8
THETA = [0.0] * N_FEAT
if POLICY.exists():
    THETA = json.loads(POLICY.read_text())["theta"]

TRAIN = False
RNG = random.Random(0)
TRACE = {}          # request_id -> (features of all nodes, chosen index)   (training only)
RESULT = {}         # request_id -> e2e seconds                             (training only)


def reset():
    TRACE.clear(); RESULT.clear()


def _curve(ex, key, n):
    c = (ex or {}).get(key)
    return float(c[min(max(n, 1), len(c)) - 1]) if c else None


def features(request, st):
    ex = st.extra or {}
    n_all = max(int(st.running) + int(st.queue_len), int(ex.get("inflight", 0)))
    n, q = min(n_all, 4), max(0, n_all - 4)
    s1 = _curve(ex, "speed_by_n", 1) or float(ex.get("speed_tps") or 30.0)
    s_next = _curve(ex, "speed_by_n", n + 1) or s1
    p = float(ex.get("prefill_tps") or 1000.0)
    L = PREDICTOR.predict(request.prompt_len, request.req_type)
    pw_next = _curve(ex, "power_by_n", n + 1) or 0.0
    pw_now = (_curve(ex, "power_by_n", n) if n else float(ex.get("idle_w") or 0.0)) or 0.0
    return [1.0, n / 4, q / 4, request.prompt_len / p, L / (s_next / (n + 1)) / 10, L / s1 / 10,
            1.0 if n_all == 0 else 0.0, (pw_next - pw_now) / 100]


def scores(feats):
    return [sum(t * f for t, f in zip(THETA, phi)) for phi in feats]


def choose_gpu(request, gpu_states):
    names = sorted(n for n, s in gpu_states.items() if s.queue_len < UNREACHABLE)
    if not names:
        return sorted(gpu_states)[0]
    feats = [features(request, gpu_states[n]) for n in names]
    sc = scores(feats)
    if not TRAIN:
        return names[max(range(len(names)), key=lambda i: sc[i])]
    m = max(sc); w = [math.exp(x - m) for x in sc]; tot = sum(w)
    r, acc, k = RNG.random() * tot, 0.0, 0
    for k, x in enumerate(w):
        acc += x
        if acc >= r:
            break
    TRACE[request.request_id] = (feats, k)
    return names[k]


def on_complete(request_id, gpu, ttft_ms, e2e_ms, output_len):
    if TRAIN:
        RESULT[request_id] = e2e_ms / 1000
