# Pilot 2: marginal-cost strategy on the real cluster (Oct 31)

Same cell as `results/pilot/` (mixed workload, rho = 0.6, 0.75 req/s, seed 1, 240 s, same requests
for every strategy), on a new set of pods (RTX 4090, RTX A6000, RTX 2000 Ada), so all six runs
below are comparable with each other. One seed only: indicative, not for the paper.

| Strategy | TTFT p95 (s) | E2E p50 (s) | E2E p95 (s) | Cluster J/token | Requests 4090 / A6000 / 2000 Ada |
|---|---|---|---|---|---|
| round_robin | 11.49 | 3.30 | 25.78 | 2.34 | 63 / 63 / 63 |
| weighted | 1.77 | 2.84 | 10.61 | 2.53 | 93 / 70 / 26 |
| least_load | 1.29 | 2.73 | 10.80 | 2.59 | 104 / 60 / 25 |
| predictive | 0.98 | 2.93 | 6.83 | 2.42 | 114 / 69 / 6 |
| marginal_cost (lambda = 0) | 1.54 | 2.68 | 7.19 | 2.27 | 154 / 35 / 0 |
| marginal_cost (lambda = 0.01 s/J) | 2.13 | 2.95 | 7.87 | 2.02 | 169 / 20 / 0 |

No failed requests in any run. With lambda = 0.01 the marginal-cost strategy used 16% less energy
per token than the predictive strategy and 14% less than round robin, with a 95th-percentile
latency within 15% of the predictive strategy. It keeps the requests on the RTX 4090, which is
already busy: one more request there adds about 20 W, while waking an idle GPU adds 60 to 260 W.

Simulation with the measured profiles: `python scripts/sim_a5.py --rho 0.3 0.6 0.9`.
