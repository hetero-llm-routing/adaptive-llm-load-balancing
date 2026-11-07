# Phase 5: experiment matrix on the real cluster (Nov 2026)

Real runs on RunPod: RTX 4090, RTX A6000 and RTX 2000 Ada behind the router, Qwen3-8B Q4_K_M,
llama.cpp with 4 slots per node, energy from the measured power of every node. 195 runs, no failed
requests. These files are the source of every number in Section 6.

| Experiment | Workload | Load (rho) | Seeds | Strategies |
|---|---|---|---|---|
| main | mixed | 0.3, 0.6, 0.9 | 5 | round_robin, weighted, least_load, least_work, predictive, rl, marginal_cost |
| bytype | chat, doc, rag | 0.6 | 3 | the same 7 |
| bursts | mixed, bursty arrivals | 0.6 | 3 | the same 7 |
| lambda0, lambda0005, lambda001 | mixed | 0.6 | 2 | marginal_cost with lambda = 0, 0.005, 0.01 s/J (default 0.002 is in main) |

## Files

- `runs.csv`: one row per run. Latency in ms (TTFT, TPOT, E2E percentiles), throughput in tokens/s,
  imbalance L (Eq. 7), cluster J/token, and per GPU: utilization U, share of the requests, J/token.
- `summary.csv`: per experiment, workload, load and strategy: mean over the seeds and the 95%
  confidence interval (`*_ci95`, half-width).
- `paired.csv`: paired bootstrap of the difference between two strategies over the same seeds
  (same requests): mean and relative difference, 95% interval, and the share of bootstrap
  samples on the other side of zero (`p_other_side`).

The simulator runs for 8 and 16 nodes (Section 6.5) use `scripts/sim_a5.py` with the calibration in
`results/calibration/`.
