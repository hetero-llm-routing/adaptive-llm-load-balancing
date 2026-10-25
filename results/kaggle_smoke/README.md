# Kaggle smoke test (Oct 23, 2026)

First real run of the router on GPUs. It checks that the whole path works (llama.cpp on each GPU, router, CSV logging).
It is an initial check only: 24 requests are too few to calibrate a simulator or to compare strategies.

## Settings

| Item | Value |
| --- | --- |
| Platform | Kaggle notebook, 2 x NVIDIA T4 (16 GB each) |
| Engine | llama.cpp `llama-server` (built from source for compute capability 7.5), one server per GPU |
| Server flags | `-ngl 99 -c 8192 --jinja --metrics` (all layers on the GPU, 8192-token context) |
| Model | Qwen3-8B, GGUF Q4_K_M (about 5 GB), thinking mode off (`/no_think`) |
| Router | `router/router.py`, strategy `round_robin` |
| Requests | 24, all sent at the same moment (closed burst, not Poisson arrivals) |
| Request mix | 8 long-document requests (1,729 prompt tokens, router estimate) and 16 short chat questions (11 to 15 tokens) |
| Generation | `max_tokens` 128, temperature 0.7, streaming, seed 1 |
| Script | `scripts/kaggle_smoke_test.py` |

`prompt_len` is the router's estimate, not the exact tokenizer count. `energy_j` is empty because the energy logger is not built yet.

## Summary (mean per GPU)

| GPU | Requests | Prompt tokens | Output tokens | TTFT (ms) | TPOT (ms) | End-to-end (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| t4_0 | 12 | 585 | 90 | 6396 | 57.3 | 11442 |
| t4_1 | 12 | 585 | 85 | 5971 | 61.4 | 10973 |

TTFT is high because all 24 requests arrive at once, so each server queues 12 requests and prefills them together.

## File

- `kaggle_smoke_round_robin_1792653252.csv`: one row per request, columns as in `docs/CSV_FORMAT.md`.
