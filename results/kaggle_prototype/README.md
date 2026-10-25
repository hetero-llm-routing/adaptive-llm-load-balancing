# Kaggle prototype run (Oct 25, 2026)

Check that the prototype works end to end on real GPUs (tasks #34 and #35): llama.cpp node state in
the router, the four strategies, the energy logger and the energy attribution. **These are not results
for the paper**: two T4 GPUs, 60 requests per run, one repetition, and the "mixed" scenario only
imitates a slower GPU.

- Notebook: `deveks2025/llm-lb-smoke-test`, version 5 "Phase 3 prototype" (private), GPU T4 x2,
  ran successfully in 1547.7 s. The files of that run are in this folder: one request CSV per
  strategy and scenario (`same/`, `mixed/`, with `energy_j` filled by the `tokens` rule), the
  energy summaries for both rules (`*_energy_tokens.json`, `*_energy_equal.json`), the run configs,
  the power log (`power.csv`), the request list (`workload.json`), one sample of each server's
  `/metrics` and `/slots`, and `summary.json`. The llama.cpp server logs are left in the notebook output.
- This llama.cpp build does not report `kv_cache_usage_ratio`, so `kv_usage` came from the share of
  busy slots (see `*_metrics.txt`).
- Script: `scripts/kaggle_prototype.py` at commit 3aeb382.
- Workload: 60 requests, Poisson arrivals at 0.6 requests/s, seed 1; about 1 in 3 is a long-document
  request (`max_tokens` 128), the rest are short questions (`max_tokens` 64, 128 or 256). The same
  list is used for every strategy.
- Scenario `same`: both GPUs serve the whole model (`-ngl 99`).
- Scenario `mixed`: `t4_1` keeps 28 layers on the GPU and runs the rest on the CPU (`-ngl 28`).
- Speeds are measured one request at a time before each scenario and passed to the strategies.
- Energy per token is the node's measured energy over the run divided by its output tokens.

## Measured node speeds (tokens/s)

| Scenario | Node | Decode (`speed_tps`) | Prompt (`prefill_tps`) |
| --- | --- | --- | --- |
| same | t4_0 | 34.71 | 1086.4 |
| same | t4_1 | 36.03 | 1111.0 |
| mixed | t4_0 | 21.07 | 907.2 |
| mixed | t4_1 (slowed) | 7.81 | 647.3 |

## Scenario `same` (two equal nodes)

| Strategy | TTFT mean (s) | TTFT p95 (s) | E2E mean (s) | E2E p95 (s) | Share t4_0 / t4_1 | J per token t4_0 / t4_1 |
| --- | --- | --- | --- | --- | --- | --- |
| round_robin | 0.81 | 2.22 | 6.34 | 12.25 | 0.50 / 0.50 | 2.08 / 1.86 |
| weighted | 0.82 | 1.82 | 7.05 | 17.12 | 0.48 / 0.52 | 1.81 / 2.03 |
| least_load | 0.47 | 0.77 | 6.18 | 14.84 | 0.45 / 0.55 | 2.18 / 1.70 |
| predictive | 0.60 | 1.26 | 6.71 | 19.25 | 0.47 / 0.53 | 1.66 / 2.25 |

## Scenario `mixed` (t4_1 slowed)

| Strategy | TTFT mean (s) | TTFT p95 (s) | E2E mean (s) | E2E p95 (s) | Share t4_0 / t4_1 | J per token t4_0 / t4_1 |
| --- | --- | --- | --- | --- | --- | --- |
| round_robin | 58.61 | 190.28 | 81.37 | 215.37 | 0.50 / 0.50 | 2.02 / 3.74 |
| weighted | 12.20 | 63.45 | 29.04 | 98.88 | 0.73 / 0.27 | 1.46 / 3.52 |
| least_load | 8.23 | 35.05 | 24.07 | 88.54 | 0.77 / 0.23 | 1.41 / 3.66 |
| predictive | 9.31 | 56.93 | 25.49 | 92.93 | 0.78 / 0.22 | 1.38 / 3.52 |

## What this shows about the code

- The router read the llama.cpp state and all four strategies ran on real servers.
- With equal nodes all strategies split the work about evenly, as expected.
- With a slower node, `weighted` gave the fast node 73% of the requests, its share of the measured
  speeds (21.07 / 28.88); `least_load` and `predictive` moved even more work to it, and all three cut
  the mean TTFT from 58.6 s (round-robin) to 8 to 12 s.
- Energy was logged and attributed for every run; the slowed node needs about twice the energy per
  output token.
