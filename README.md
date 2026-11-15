# Adaptive Load Balancing for LLM Inference on Heterogeneous GPU Clusters

Course project, ECOM 6339, Islamic University of Gaza.
Team: Eyad Kamal Salama, Osama Raed Juneina, Mohammedjamal Mohammedjawad Elkhatib.
Supervisor: Dr. Khalil Ibrahim Abu Zanouna.

**Paper:** [`paper/Adaptive_Load_Balancing_LLM_Heterogeneous_GPUs_AbuZanouna_Salama_Juneina_Elkhatib.pdf`](paper/Adaptive_Load_Balancing_LLM_Heterogeneous_GPUs_AbuZanouna_Salama_Juneina_Elkhatib.pdf)

## Main results

- 195 runs on a real cluster (RTX 4090, RTX A6000, RTX 2000 Ada; Qwen3-8B on llama.cpp):
  predictive, RL and marginal-cost routing cut the 95th-percentile end-to-end latency by 60–64%
  against round robin at medium load (`results/phase5/`).
- Marginal-cost routing (ours) needs 13% less energy per token than RL at medium load, at a 6%
  higher tail latency; against least work its gain is energy, not latency.
- Calibrated simulator up to 16 GPUs (`results/scaling/`); decision time below 0.25 ms (p99)
  (`results/overhead/`).
- Case study: a private RAG assistant on Qwen3-14B finished bursts of questions 35% sooner with
  marginal-cost routing than with round robin (`results/case_study_en/`).


A router sits in front of several inference servers and decides, for every request,
which server (GPU) handles it. We compare four strategy families: static, dynamic,
predictive and reinforcement learning.

## Folders and owners

| Folder | Owner | Contents |
| --- | --- | --- |
| `infra/` | Eyad | Inference servers, mock server, tunnels, Docker |
| `router/` | Eyad | OpenAI-compatible router that logs one CSV row per request |
| `strategies/` | Osama | One file per strategy, all with the same interface |
| `bench/` | Mohammed | Workload generator and load tester |
| `analysis/` | Mohammed | Statistics and figures |
| `results/` | Router output | One CSV file per run |
| `paper/` | Everyone | Paper sections |

## Run everything on a laptop (no GPU needed)

```bash
pip install -r requirements.txt
bash scripts/run_mock_cluster.sh          # 3 mock nodes (fast, medium, slow) + router on port 8000
python scripts/send_test_requests.py      # in a second terminal
```

A new CSV file appears in `results/`. To use a real cluster (Kaggle, RunPod), change only
`config.yaml`; the code stays the same.

## Writing a strategy (Osama)

Create `strategies/<name>.py` with one function:

```python
def choose_gpu(request, gpu_states):
    # request: request_id, prompt_len, req_type ("chat" or "rag"), arrival_ts
    # gpu_states: {node_name: GpuState(queue_len, running, kv_usage)}
    return node_name
```

Then set `strategy: <name>` in `config.yaml`. See `strategies/round_robin.py`.

## Rules

- Work on a branch, open a pull request, and another member reviews it before it goes into `main`.
- Never commit passwords, API keys or model files (`*.gguf`).
- Every number in the paper must come from a CSV file in `results/`.
- See `docs/CSV_FORMAT.md` for the results format.
