"""Train the RL strategy (strategies/rl.py) with REINFORCE on the simulator of the testbed
(scripts/sim_a5.py: measured speed and power by concurrency, A5), task #54.

    python scripts/train_rl.py --episodes 1500 --out strategies/rl_policy.json

Each episode: 300 requests of the mixed workload, Poisson arrivals at rho drawn from
{0.3, 0.6, 0.9} of the round-robin saturation rate, a new seed. Reward of a decision: minus the
request's end-to-end latency (s), centred by the episode mean. Adam on the policy weights.
Every 100 episodes the greedy policy is evaluated on fixed seeds (not used in training) and the
best weights are kept.
"""
import argparse, importlib, json, math, random, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import sim_a5 as S
from strategies.length_predictor import LengthPredictor
import strategies.rl as RL


def episode_jobs(rho, seed, n):
    rng = random.Random(seed)
    jobs, t = S.sample(rng, n), 0.0
    for j in jobs:
        t += rng.expovariate(rho * S.SAT_MIXED); j["at"] = t
    return jobs


def evaluate(nodes, seeds=(9001, 9002, 9003), rhos=(0.3, 0.6, 0.9), n=400):
    RL.TRAIN = False
    out = []
    for rho in rhos:
        for sd in seeds:
            out.append(S.simulate(RL, episode_jobs(rho, sd, n), nodes)["e2e_p95"])
    return statistics.mean(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--out", default=str(ROOT / "strategies" / "rl_policy.json"))
    a = ap.parse_args()
    nodes = S.node_profiles()
    RL.PREDICTOR = LengthPredictor.fit(S.sample(random.Random(999), 3000))
    theta = [0.0] * RL.N_FEAT
    m = [0.0] * RL.N_FEAT; v = [0.0] * RL.N_FEAT; b1, b2 = 0.9, 0.999
    best, best_theta, log = float("inf"), list(theta), []
    rng = random.Random(1)
    for ep in range(1, a.episodes + 1):
        RL.THETA = theta; RL.TRAIN = True; RL.RNG = random.Random(ep)
        rho = rng.choice([0.3, 0.6, 0.9])
        S.simulate(RL, episode_jobs(rho, 100000 + ep, a.n), nodes)
        rewards = {rid: -e for rid, e in RL.RESULT.items() if rid in RL.TRACE}
        base = statistics.mean(rewards.values())
        sd = statistics.pstdev(rewards.values()) or 1.0
        grad = [0.0] * RL.N_FEAT
        for rid, R in rewards.items():
            feats, k = RL.TRACE[rid]
            sc = RL.scores(feats); mx = max(sc); w = [math.exp(x - mx) for x in sc]; tot = sum(w)
            pi = [x / tot for x in w]
            adv = (R - base) / sd
            for f in range(RL.N_FEAT):
                grad[f] += adv * (feats[k][f] - sum(p * phi[f] for p, phi in zip(pi, feats)))
        grad = [g / len(rewards) for g in grad]
        for f in range(RL.N_FEAT):                  # Adam, gradient ascent
            m[f] = b1 * m[f] + (1 - b1) * grad[f]; v[f] = b2 * v[f] + (1 - b2) * grad[f] ** 2
            theta[f] += a.lr * (m[f] / (1 - b1 ** ep)) / (math.sqrt(v[f] / (1 - b2 ** ep)) + 1e-8)
        RL.reset()
        if ep % 100 == 0:
            RL.THETA = theta
            score = evaluate(nodes)
            log.append({"episode": ep, "eval_e2e_p95_s": round(score, 3)})
            print(f"episode {ep}: greedy E2E p95 on eval seeds {score:.3f} s  theta {[round(x, 2) for x in theta]}", flush=True)
            if score < best:
                best, best_theta = score, list(theta)
    Path(a.out).write_text(json.dumps({"theta": best_theta, "features": ["bias", "running/4", "waiting/4", "prefill_s",
                                       "own_decode_s/10", "decode_s_alone/10", "idle", "extra_power/100"],
                                       "eval_e2e_p95_s": best, "episodes": a.episodes, "log": log}, indent=2))
    print("best", best, "saved", a.out)


if __name__ == "__main__":
    main()
