# Decision time of the strategies (task #77, Section 6.4)

`decision_time.csv` is written by

```bash
python scripts/decision_time.py        # about 20 minutes on a laptop CPU
```

## Method

- Every strategy routes the mixed workload in the simulator of the testbed (`scripts/sim_a5.py`,
  measured speed and power by concurrency, A5) on 3, 8 and 16 GPUs (`scripts/sim_scaling.py`),
  at rho = 0.6 and 0.9 of the round-robin saturation rate. Each call of `choose_gpu` is timed with
  `time.perf_counter_ns()`, so the strategy sees the node states and keeps the records (backlog,
  tracked requests, completions through `on_complete`) of a run at that load.
- 500 warm-up decisions, then 10,000 measured decisions per strategy, cluster size and load.
- Each run is repeated 5 times, in rounds (every strategy, size and load once per round), so the
  repeats of one run are minutes apart. The repeats are identical (same requests, states and
  choices; the script checks this), so for every decision the fastest repeat is kept: other
  programs on the machine slow some calls down, at random and in busy periods, and the minimum
  removes most of that. The `single_run_*` columns are round 1 alone, with the disturbance.
- Only the decision is timed. Reading the node states is the router's work, done in the background
  every 0.25 s on the testbed (`router.state_interval_s`); one read took 0.3 to 0.9 s through the
  RunPod proxy (`router/router.py`).
- Python's garbage collector is paused during a run, as `timeit` does.

Machine: Intel Core i7-9750H (laptop, mains power, other programs running), Windows 11,
Python 3.13.7. Timer resolution 0.1 us; an empty call through the same timing code takes 0.6 us
(`timer_floor_us`), which is included in every value.

## Results (rho = 0.6): median (p99) in microseconds

| Strategy | 3 GPUs | 8 GPUs | 16 GPUs |
|---|---|---|---|
| round_robin | 1.0 (3.4) | 0.9 (2.9) | 1.4 (3.8) |
| weighted | 4.1 (12.5) | 8.4 (18.2) | 14.1 (34.8) |
| least_load | 2.9 (8.4) | 6.6 (15.8) | 10.3 (22.7) |
| least_work | 5.1 (13.9) | 11.1 (25.0) | 19.0 (41.5) |
| power_of_two | 6.2 (15.9) | 7.1 (15.9) | 9.6 (23.0) |
| predictive | 9.5 (21.5) | 18.4 (41.2) | 31.6 (72.5) |
| rl | 23.2 (51.2) | 44.5 (98.1) | 84.9 (185.7) |
| marginal_cost | 40.1 (75.4) | 87.1 (150.4) | 108.7 (235.3) |

At rho = 0.9 the medians differ from these by -27% to +48% (largest: marginal_cost and rl on 8 GPUs;
round robin +20%, which is 0.2 us) with no consistent direction, so this is the measurement
variation that remains after the minimum of 5 repeats, not an effect of the load.
