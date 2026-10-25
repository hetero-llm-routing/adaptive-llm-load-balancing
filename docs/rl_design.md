# Design note: reinforcement-learning strategy (Osama, Oct 2026)

Goal: a router policy, learned from experience, that sends each request to a node.
It uses the same interface as every other strategy, `choose_gpu(request, gpu_states)`,
so it runs unchanged in the router and in `strategies/mock_sim.py`.

## 1. Decision step

One step = one routing decision, taken when a request arrives. An episode = one workload
run (for example 2,000 requests). The policy never moves a request after sending it.

## 2. State (what the policy sees)

Only fields the router already has (`strategies/base.py`), so nothing new is needed on the nodes.

| Part | Feature | Source | Tabular version | DQN version |
| --- | --- | --- | --- | --- |
| Per node j | load = queue_len + running | `GpuState` | bucket: 0, 1, 2, 3, 4-5, 6-8, 9+ | value / 10 |
| Per node j | kv_usage | `GpuState` | not used | 0 to 1 |
| Per node j | decode speed s_j | `extra["speed_tps"]` or config.yaml | not needed (node identity is fixed) | s_j / max speed |
| Per node j | predicted backlog B_j / s_j (seconds) | same estimate as `predictive.py` | not used | value / 30 |
| Request | req_type | `Request` | chat / rag | 0 / 1 |
| Request | predicted output length L | `length_predictor.py` | short / medium / long | L / 300 |
| Request | prompt_len | `Request` | not used | log(1 + P) / 8 |

Unreachable nodes (`queue_len` = 10^6) are masked out of the actions.

Size of the tabular state with 3 nodes: 7^3 load combinations x 2 types x 3 length classes = 2,058
states, x 3 actions = about 6,200 Q-values, small enough to learn in a few simulated runs.

## 3. Action

Choose one node. For the DQN we score every node with the same small network,
`Q(s, j) = f([node j features, request features, cluster summary])`, and take the best.
Because the weights are shared across nodes, the same network works for 3, 8 or 16 nodes,
which we need for the scaling study (RQ3). Lodestar [17] uses the same idea for its router.

## 4. Reward

The reward comes when the request finishes:

    r = - ( E2E / E2E_ideal )

where E2E_ideal = P / p_fast + O / s_fast is the time the request would take alone on the
fastest node (P = prompt tokens, O = output tokens). Dividing by the ideal time stops long
requests from dominating the reward; this ratio is the request's *slowdown*.

Planned variants for the ablation (one weight each, default 0):

- TTFT term: r = - [ (1 - a) * slowdown + a * TTFT / TTFT_ideal ], with a in {0, 0.5}.
  The predictive strategy improves completion time but raises TTFT in the mock tests, so this
  shows whether RL can balance both.
- Energy term: - b * energy_j / output tokens, once the energy logger fills `energy_j`.

## 5. Training

1. Offline, in `strategies/mock_sim.py` (thousands of requests per second, no GPU). Node speeds
   come from config.yaml; later they are calibrated against the Kaggle and RunPod runs.
2. Training seeds 100-149, evaluation seeds 1-5 (the same as all other strategies), so the
   policy is never tested on the requests it learned from.
3. Exploration: epsilon-greedy, epsilon from 1.0 down to 0.05 over the training runs.
4. Discount: start with gamma = 0 (each decision judged by its own request's reward, a
   contextual bandit; stable and fast). Then try gamma = 0.9 with next state = the state at
   the next decision, so the policy also learns how its choice delays later requests.
5. Tabular Q-learning: learning rate 0.1, numpy only.
6. DQN: MLP with 2 hidden layers of 64 units, replay buffer 50,000, target network updated
   every 1,000 steps, Adam 1e-3, Huber loss, PyTorch on CPU.
7. Evaluation: the trained policy is frozen (epsilon = 0) and run like any other strategy.
   Its decision time is measured the same way as for `predictive.py` (about 5 microseconds).

Fallback agreed in the plan: if the DQN is not stable by week 11, we report tabular Q-learning.

## 6. What is needed from others

- **Router (Eyad):** the reward arrives after the request ends, but the router never tells the
  strategy. Proposal: after `log_row(...)`, call
  `STRATEGY.on_complete(request_id, gpu, ttft_ms, e2e_ms, output_len)` if the strategy defines it.
  One line; strategies without the function are not affected. Needed only for online learning
  on real nodes; offline training in the simulator does not need it.
- **Router (Eyad):** pass `speed_tps` from config.yaml (and the mock `/state`) into
  `GpuState.extra`, and add a `req_type` column to the CSV, so the length predictor can learn
  per request type.
- **Simulator (Mohammed):** if Vidur can model our GPUs, the policy can also be trained on it;
  `mock_sim.py` stays as the fast default.

## 7. Risks

- The mock model may not match real GPUs: calibrate speeds from real runs before final training.
- A policy trained offline can do worse online because its own decisions change the load it sees
  (Lodestar [17] calls this the circular dependency). If time allows: short online fine-tuning.
- Few nodes (2 to 3) leave little room to beat a good heuristic; the 8 and 16 node simulations matter.
