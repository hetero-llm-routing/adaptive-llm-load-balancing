# Calibration and pilot (task #60), Oct 31

Real cluster: RTX 4090, RTX A6000, RTX 2000 Ada on RunPod, llama.cpp de5122d, Qwen3-8B Q4_K_M, 4 slots per node.

## Workloads (`workloads_meta.json`, token counts with the Qwen3-8B tokenizer)

| Family | Requests | Prompt p50 / p95 | Output p50 / p95 |
|---|---|---|---|
| chat (ShareGPT) | 1,500 | 40 / 276 | 252 / 697 |
| doc (LongBench, prompt <= 6,000) | 235 | 3,269 / 5,778 | 128 / 512 |
| rag (HotpotQA, BM25 top-5 offline) | 1,500 | 664 / 1,004 | 137 / 467 |
| mixed (70 / 15 / 15) | 1,500 | | |

## Round-robin saturation rate (`calibration.json`)

| Family | Requests/s |
|---|---|
| chat | 1.84 |
| doc | 0.98 |
| rag | 2.15 |
| mixed | 1.25 |

A rate is stable when >= 95% of the requests arriving after a 30 s warm-up finish within 60 s
after the last arrival and the 90th-percentile latency of the last third is <= 2x that of the
first third (180 s steps, bisection to 5%). Each step's numbers: `cal_*_meta.json`, `cal_*_energy.json`.

## Pilot (`../pilot/`): mixed workload, rho = 0.6 (0.75 req/s), seed 1, 240 s, same requests for every strategy

| Strategy | TTFT p50 / p95 (ms) | E2E p50 / p95 (ms) | Cluster J/token |
|---|---|---|---|
| round_robin | 320 / 12,598 | 3,437 / 26,621 | 2.43 |
| weighted | 222 / 1,106 | 2,856 / 11,770 | 2.58 |
| least_load | 265 / 927 | 2,835 / 10,916 | 2.64 |
| predictive | 253 / 776 | 2,985 / 7,923 | 2.55 |

No failed requests; about 750 power samples per run; the client sent every request within 5 ms of
its scheduled time. First observation: the strategies that cut tail latency send fewer requests
to the RTX 2000 Ada (the most efficient GPU), so they use more energy per token; the trade-off
between latency and energy is visible already at medium load.
