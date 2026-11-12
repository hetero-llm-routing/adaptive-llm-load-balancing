# Strategies on 3, 8 and 16 GPUs in the simulator (task #79, Section 6.5, Fig. 3)

```bash
python scripts/sim_scaling.py                       # sim_runs.csv, sim_summary.csv (about 5 minutes)
python scripts/fig3_cluster_size.py                 # fig3_cluster_size.png (needs matplotlib)
python scripts/sim_scaling.py --check-saturation --step-s 900
```

## Setup

- Simulator: `scripts/sim_a5.py` with the measured speed and power by concurrency of each GPU
  (A5, `results/profiles/`) and the calibration fitted on the real pilot (`results/sim_calibration.json`).
- Clusters: 3 GPUs = the testbed (RTX 4090, RTX A6000, RTX 2000 Ada). 8 and 16 GPUs cannot be split
  into three equal parts, so each is run with the three rotations of its mix (8: 3/3/2, 2/3/3, 3/2/3;
  16: 6/5/5, 5/6/5, 5/5/6, counts of RTX 4090 / RTX A6000 / RTX 2000 Ada) on the same requests, and a
  cell averages them. Each GPU type is then one third of the cluster on average, as on the testbed.
- Load: rho = 0.3, 0.6, 0.9 of the round-robin saturation rate of the cluster, k/3 x 1.254 req/s
  (`results/calibration/calibration.json`): round robin gives every GPU 1/k of the requests, so the
  load per GPU is the same for every cluster size.
- Workload: the mixed workload of the simulator (70% chat, 15% doc, 15% RAG), Poisson arrivals for
  300 s, 5 seeds; the same requests for every strategy of a cell.
- Strategies: the seven of the real runs (marginal_cost with lambda = 0.002 s/J) plus power_of_two
  and oracle (true output length), which ran in the simulator only.
- Metrics: as `analysis/analyze.py` computes them for the real runs; requests that arrive in the
  first 30 s are left out of the latency, throughput and utilization metrics; energy covers the whole
  run on every GPU (Eq. 9).

## Files

- `sim_runs.csv`: one row per run (945 runs), the columns of `results/phase5/runs.csv`. experiment =
  sim3, sim8, sim16; run_id names the mix (m332 = 3 RTX 4090, 3 RTX A6000, 2 RTX 2000 Ada). The
  per-GPU columns are per GPU type (U: mean over the GPUs of that type; share: share of the requests;
  jpt: energy of those GPUs per token they generated).
- `sim_summary.csv`: mean and 95% confidence interval (half-width) per cell, the columns of
  `results/phase5/summary.csv`; `seeds` counts the runs (5 for 3 GPUs, 15 = 5 seeds x 3 mixes for 8
  and 16).
- `fig3_cluster_size.png`: Fig. 3, E2E p95 and J/token against cluster size at rho = 0.6, 300 dpi.

## Check against the real cluster

The simulator runs the 3-GPU testbed with the same strategies and loads as the real runs
(`results/phase5/summary.csv`, experiment main), so the two can be compared directly (tables below):

- rho = 0.6: E2E p95 within 10% of the measured value for six of the seven strategies (+3% to +10%)
  and J/token within 5% (-5% to +4%). The exception is least_work (-40% and -9%): its real runs
  include one seed with a long tail (12.2 +- 9.9 s over the 5 seeds).
- rho = 0.3: E2E p95 from -2% to +25% (weighted +18%, least_load +25%), J/token from -8% to +1%.
- rho = 0.9: E2E p95 from -6% to +15% except round robin, 46.7 s against 25.5 s measured: its RTX 2000
  Ada runs at 90% of saturation, where a small difference in service time changes the queue a lot.
  J/token from 0% to +9%.

Saturation check (`--check-saturation --step-s 900`): the simulated round-robin saturation rate is
0.98, 2.72 and 5.43 req/s for 3, 8 and 16 GPUs, that is 0.33, 0.34 and 0.34 req/s per GPU, so it grows
in proportion to the cluster size as assumed above. The absolute value depends on the step length:
with the 180 s steps of `bench/calibrate.py` the simulator gives 1.06 req/s for the testbed, against
1.25 measured, so near saturation the simulator is somewhat pessimistic (see round robin at rho = 0.9).

## Results

E2E p95 (s) / J per token; real 3 = the real cluster.

rho = 0.3: E2E p95 (s) and J/token, mean over the runs of the cell

| Strategy | real 3 | sim 3 | sim 8 | sim 16 |
|---|---|---|---|---|
| round_robin | 16.90 / 3.67 | 16.60 / 3.39 | 17.26 / 3.55 | 14.09 / 3.71 |
| weighted | 8.24 / 3.76 | 9.76 / 3.59 | 10.47 / 3.83 | 9.63 / 3.97 |
| least_load | 7.22 / 3.73 | 8.99 / 3.71 | 7.90 / 3.90 | 6.32 / 3.88 |
| least_work | 5.83 / 2.88 | 6.72 / 2.90 | 6.10 / 3.03 | 5.66 / 3.15 |
| power_of_two | - | 12.30 / 3.64 | 12.49 / 3.64 | 11.27 / 3.70 |
| predictive | 7.31 / 3.37 | 7.19 / 3.14 | 6.36 / 3.18 | 6.04 / 3.17 |
| rl | 5.97 / 3.10 | 6.63 / 3.07 | 6.17 / 3.16 | 5.65 / 3.18 |
| marginal_cost | 5.90 / 2.76 | 6.69 / 2.72 | 6.21 / 2.87 | 5.75 / 2.91 |
| oracle | - | 6.76 / 2.85 | 6.11 / 3.01 | 5.66 / 3.14 |

rho = 0.6: E2E p95 (s) and J/token, mean over the runs of the cell

| Strategy | real 3 | sim 3 | sim 8 | sim 16 |
|---|---|---|---|---|
| round_robin | 19.39 / 2.46 | 20.55 / 2.51 | 17.66 / 2.77 | 16.65 / 2.78 |
| weighted | 10.61 / 2.59 | 10.94 / 2.66 | 10.36 / 2.94 | 10.36 / 2.95 |
| least_load | 10.56 / 2.62 | 10.90 / 2.73 | 9.58 / 3.09 | 9.42 / 3.12 |
| least_work | 12.23 / 2.40 | 7.32 / 2.19 | 6.57 / 2.22 | 6.87 / 2.05 |
| power_of_two | - | 11.94 / 2.62 | 11.91 / 2.76 | 12.07 / 2.73 |
| predictive | 7.71 / 2.37 | 7.99 / 2.26 | 6.99 / 2.37 | 7.25 / 2.36 |
| rl | 6.95 / 2.38 | 7.64 / 2.35 | 6.52 / 2.49 | 6.74 / 2.43 |
| marginal_cost | 7.36 / 2.07 | 8.07 / 1.98 | 7.00 / 1.96 | 7.15 / 1.86 |
| oracle | - | 7.59 / 2.15 | 6.70 / 2.19 | 6.91 / 2.03 |

rho = 0.9: E2E p95 (s) and J/token, mean over the runs of the cell

| Strategy | real 3 | sim 3 | sim 8 | sim 16 |
|---|---|---|---|---|
| round_robin | 25.54 / 1.92 | 46.74 / 2.11 | 53.04 / 2.13 | 53.93 / 2.20 |
| weighted | 11.46 / 2.02 | 11.46 / 2.16 | 11.64 / 2.16 | 11.20 / 2.21 |
| least_load | 12.51 / 2.03 | 11.72 / 2.20 | 10.25 / 2.22 | 9.97 / 2.28 |
| least_work | 8.23 / 1.83 | 7.99 / 1.90 | 7.76 / 1.87 | 7.29 / 1.92 |
| power_of_two | - | 13.25 / 2.13 | 13.05 / 2.07 | 12.79 / 2.12 |
| predictive | 8.80 / 1.81 | 8.95 / 1.84 | 8.52 / 1.77 | 8.13 / 1.78 |
| rl | 7.79 / 1.85 | 8.08 / 1.95 | 7.77 / 1.98 | 7.29 / 2.05 |
| marginal_cost | 8.29 / 1.74 | 9.52 / 1.74 | 9.00 / 1.63 | 8.77 / 1.54 |
| oracle | - | 9.12 / 1.88 | 10.04 / 1.89 | 8.79 / 1.91 |

## Limits

- TTFT: the simulator does not model network and HTTP time (`scripts/validate_sim.py`), so its TTFT
  is lower than on the real cluster; compare TTFT only between simulated runs.
- One workload (mixed) and one arrival process (Poisson) for 8 and 16 GPUs.
- The simulator's mixed workload is drawn from the fitted distributions of `scripts/sim_a5.py`, not
  replayed from the real request lists.
