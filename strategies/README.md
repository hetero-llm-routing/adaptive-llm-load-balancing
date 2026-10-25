# strategies/ (owner: Osama)

Every strategy is one file with `choose_gpu(request, gpu_states) -> node name` (see `base.py`).
Set `strategy: <file name>` in `config.yaml` to use it in the router.

| File | What it is |
| --- | --- |
| `base.py` | The agreed interface: `Request`, `GpuState` |
| `round_robin.py` | Baseline: nodes in turn |
| `predictive.py` | Predicts the output length and sends the request to the node expected to finish it first |
| `length_predictor.py` | Output-length predictor used by `predictive.py` (bucket means of earlier runs) |
| `mock_sim.py` | Offline simulator with the same timing rules as `infra/mock_server.py`; tests strategies in seconds and will train the RL strategy |

## Predictive strategy in one line

`T_j = (B_j + L) / s_j + P / p_j`, pick the node with the smallest `T_j`, where `L` is the predicted
output length, `B_j` the predicted tokens still waiting on node `j`, `s_j` its decode speed,
`P` the prompt length and `p_j` its prefill speed. Details in the docstring of `predictive.py`.

Each node in `config.yaml` needs `speed_tps` (and optionally `prefill_tps`), measured once per GPU type.

## Fit the length table (from earlier runs only, never from the run being evaluated)

```bash
python -m strategies.length_predictor fit "results/<training run>/*.csv" -o strategies/length_table.json
python -m strategies.length_predictor eval strategies/length_table.json "results/<other run>/*.csv"
```

Without `length_table.json` the predictor uses 128 tokens for every request.

## Test

```bash
python tests/test_predictive.py                    # or: python -m pytest tests/ -q
python -m strategies.mock_sim --strategies round_robin predictive --rate 0.5 --seeds 1 2 3
```
