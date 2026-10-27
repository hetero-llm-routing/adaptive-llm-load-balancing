# Vidur check (task #38)

## 1. Installation and the built-in example

- Source: `github.com/microsoft/vidur` (commit abae7f6), installed with `pip install -e .` in a Python 3.12
  virtual environment. Runs on CPU only; no GPU is needed for simulation.
- Example run: Llama-3-8B (fp16) on one A100, 200 requests, Poisson arrivals at 2 requests/s, fixed lengths
  of 2048 prompt and 512 output tokens, Sarathi scheduler (chunk 512), round-robin global scheduler.
  Command: `WANDB_MODE=disabled python -m vidur.main --replica_config_device a100
  --replica_config_model_name meta-llama/Meta-Llama-3-8B --synthetic_request_generator_config_num_requests 200
  --poisson_request_interval_generator_config_qps 2`.
- The first run took about 35 minutes on 2 CPU cores, because Vidur first trains its execution-time
  predictors (random forests, one per operation) on the profiling data. They are cached in `cache/`, so later
  runs with the same GPU and model take minutes. The final plotting step needs Chrome (kaleido); the metrics
  CSV is written before that step, so the failure there does not affect the results.

| Simulated metric (A100, Llama-3-8B fp16) | Mean | Range |
| --- | --- | --- |
| Time to first token (`prefill_e2e_time`) | 0.32 s | 0.17 to 0.91 s |
| Time per output token | 23 ms | 13 to 27 ms |
| End-to-end latency | 11.9 s | 7.0 to 14.0 s |

Files in this folder: `vidur_example_a100_llama3_8b_request_metrics.csv` (one row per simulated request) and `vidur_example_config.json` (the full run configuration).

## 2. Comparison with the Kaggle runs (order of magnitude only)

Our Kaggle runs used two T4 GPUs with llama.cpp and Qwen3-8B in 4-bit GGUF; the time per output token there
was 53 to 61 ms per request under load. Vidur's 23 ms on an A100 in fp16 is in a plausible ratio (the A100
has about 5 to 7 times the memory bandwidth of a T4), but the two runs differ in GPU, engine, precision, model and
workload, so this is not a calibration.

## 3. What is needed to simulate our GPUs and model

1. **Profiling on each GPU type.** Vidur ships profiles only for A100, H100 and A40. A new GPU needs MLP and
   attention profiling on the real GPU, using `sarathi-serve` (branch `vidur`) and the scripts in
   `vidur/profiling/` (see `docs/profiling.md`). One GPU of each type is enough. Sarathi-serve uses kernels for
   Ampere and newer GPUs, so the RTX 2000 Ada, RTX A4500 and RTX 4090 should work; the T4 (Turing) is unlikely
   and the P100 (Pascal) is not supported. To be confirmed on RunPod in Phase 4.
2. **A model config for Qwen3-8B.** A YAML file with the Hugging Face config values (36 layers, 32 query and 8
   KV heads, hidden size 4096, MLP size 12288). Qwen3 has a few layers that Vidur's reference model does not
   have (for example the query/key norm), so the profile is an approximation.

## 4. Limits that affect our study (team decision needed)

- **Engine and precision.** Vidur models a PyTorch/vLLM-style engine in fp16. Our main engine is llama.cpp with
  4-bit GGUF, which Vidur does not model. Vidur results would describe the vLLM setting, not our main runs.
- **One GPU type per cluster.** Vidur's cluster config has a single replica type, so a mixed cluster
  (for example 4090 + A4500 + RTX 2000 Ada) needs changes to Vidur's code.
- **Routing strategies.** Vidur's global schedulers are round-robin, random and least-outstanding-requests.
  Our static weighted, predictive and RL strategies would have to be ported as new global schedulers.

Options for the 8- and 16-GPU scale-out promised in Section 1:
- **A.** Keep Vidur: profile the three Ampere/Ada GPUs on RunPod in Phase 4, extend Vidur to mixed clusters and
  port the strategies. Most effort; results describe vLLM in fp16.
- **B.** Use our own simulator (`strategies/mock_sim.py`), calibrated with the decode and prompt speeds measured
  on the real GPUs with llama.cpp, and compared with the real runs. Less effort; same engine as the main
  experiments; less detailed than Vidur.
- **C.** B for the scale-out, and Vidur only as a cross-check for one GPU type.

Section 1 currently says the scale-out uses Vidur [20]; if the team chooses B, that sentence needs updating.
