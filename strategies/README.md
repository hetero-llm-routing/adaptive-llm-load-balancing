# strategies/ (owner: Osama)

Every strategy is one file with `choose_gpu(request, gpu_states) -> node name` (see `base.py`).
Set `strategy: <file name>` in `config.yaml` to use it in the router.

| File | What it is |
| --- | --- |
| `base.py` | The agreed interface: `Request`, `GpuState` |
| `round_robin.py` | Static baseline: nodes in turn |
| `weighted.py` | Static: fixed shares proportional to the measured speed (smooth weighted round robin) |
| `least_load.py` | Dynamic: fewest requests on the node (waiting + running, or the router's in-flight count) |
| `least_work.py` | Dynamic: least load divided by the node's speed with one more request, `(n_j + 1) / s_j(n_j + 1)` |
| `power_of_two.py` | Dynamic: two random nodes, the less loaded wins [11]; simulator only in the paper |
| `predictive.py` | Predicts the output length and sends the request to the node expected to finish it first (A5) |
| `marginal_cost.py` | Predictive (Eyad): least added latency of all requests plus lambda times the added energy |
| `oracle.py` | Reference bound: the predictive rule with the true output length; simulator only |
| `rl.py`, `rl_policy.json` | Learned: softmax policy over per-node features, trained with REINFORCE (`scripts/train_rl.py`) |
| `length_predictor.py` | Output-length predictor used by the predictive strategies (bucket means of earlier runs) |
| `mock_sim.py` | Offline simulator with the same timing rules as `infra/mock_server.py` (A5 when `speed_by_n` is set) |

The Phase 5 runs on the real cluster (`results/phase5/`) used these files; they came from the code
bundle on the `deploy` branch (commit 934a4bf).

## Predictive strategy in one line

`T_j = (B_j + L) / s_j(n_j + 1) + P / p_j`, pick the node with the smallest `T_j`, where `L` is the
predicted output length, `B_j` the predicted tokens still waiting on node `j`, `s_j(n)` its total
decode speed with `n` running requests (A5, `speed_by_n`; `speed_tps` if not configured), `P` the
prompt length and `p_j` its prefill speed. Details in the docstring of `predictive.py`.

Each node in `config.yaml` needs `speed_tps` (and optionally `prefill_tps`, `speed_by_n`), measured
once per GPU type (`results/profiles/`).

## Fit the length table (from earlier runs only, never from the run being evaluated)

```bash
python -m strategies.length_predictor fit "results/<training run>/*.csv" -o strategies/length_table.json
python -m strategies.length_predictor eval strategies/length_table.json "results/<other run>/*.csv"
```

Without `length_table.json` the predictor uses 128 tokens for every request.

## Simulator of the testbed and Section 6 scripts

| Script | Output |
| --- | --- |
| `scripts/sim_a5.py` | Simulator with the measured speed and power by concurrency, calibrated on the pilot (`results/sim_calibration.json`) |
| `scripts/validate_sim.py` | Simulator against the real pilot runs (task #61) |
| `scripts/train_rl.py` | Trains `rl_policy.json` on the simulator (task #54) |
| `scripts/sim_scaling.py` | 3, 8 and 16 GPUs in the simulator (task #79): `results/scaling/` |
| `scripts/decision_time.py` | Decision time of every strategy (task #77): `results/overhead/` |
| `scripts/fig3_cluster_size.py` | Fig. 3 (task #80): `results/scaling/fig3_cluster_size.png` (needs matplotlib) |

## Test

```bash
python tests/test_predictive.py                    # or: python -m pytest tests/ -q
python tests/test_phase5_strategies.py
python -m strategies.mock_sim --strategies round_robin predictive --rate 0.5 --seeds 1 2 3
```
