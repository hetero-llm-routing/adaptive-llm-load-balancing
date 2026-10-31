# GPU profiles (task #46, Oct 31)

Measured with `scripts/profile_gpu.py` on RunPod, one pod per GPU, after `infra/runpod_node.sh`:
llama.cpp build b11923 (commit de5122d), Qwen3-8B Q4_K_M, `-c 32768 -np 4` (four slots of 8,192
tokens each), temperature 0, 256 output tokens per request (`ignore_eos`), median of 3 repetitions,
30 s wait after the server started before the idle power. Driver 595.91.07, CUDA 12.8.

## Decode speed by the number of running requests (tokens/s, all requests together)

| GPU | n = 1 | 2 | 3 | 4 | Speed-up at 4 | Per request at 4 |
| --- | --- | --- | --- | --- | --- | --- |
| RTX 4090 | 150.6 | 269.8 | 359.0 | 450.4 | 2.99 | 119.4 |
| RTX A6000 | 114.8 | 208.7 | 273.5 | 325.8 | 2.84 | 84.6 |
| RTX 2000 Ada | 41.7 | 79.7 | 112.9 | 133.4 | 3.20 | 33.9 |
| T4 (Kaggle, simulation only) | 32.7 | 50.8 | 53.0 | 64.2 | 1.96 | 16.3 |

These are the `speed_by_n` values in `config_testbed.yaml`; the simulator and the predictive
strategy interpolate between them and keep s(4) above four running requests (A5 in Section 3.2).

## Prompt (prefill) speed, one request (tokens/s)

| GPU | 303 tokens | 1,827 tokens | 5,547 tokens |
| --- | --- | --- | --- |
| RTX 4090 | 8,655 | 11,836 | 10,774 |
| RTX A6000 | 4,423 | 4,802 | 4,389 |
| RTX 2000 Ada | 1,802 | 2,000 | 1,804 |
| T4 | 1,023 | 1,131 | 995 |

## Power (W, mean over each measurement)

| GPU | Idle | Decode, 1 request | Decode, 4 requests |
| --- | --- | --- | --- |
| RTX 4090 | 22.8 | 285.9 | 350.3 |
| RTX A6000 | 36.8 | 289.5 | 295.0 |
| RTX 2000 Ada | 7.5 | 68.3 | 69.0 (power limit 70 W) |
| T4 | 27.2 | 66.6 | 65.6 (power limit 70 W) |

Energy per output token at four running requests: 4090 0.78 J, A6000 0.91 J, 2000 Ada 0.52 J.
The fastest GPU is not the most efficient one, and the A6000 is the least efficient (H3).

## Notes

- **Context per slot.** With `-c 8192 -np 4` llama.cpp gives each slot only 2,048 tokens, and the
  long-document prompts (about 5,500 tokens) were rejected. The servers now use `-c 32768`
  (8,192 per slot); the 2000 Ada uses 9.3 GB of its 16 GB with this setting.
- **Same GPU, different pod.** The A6000 decoded 114.8 tokens/s here and 98.1 on Oct 27 (another
  machine and an older llama.cpp build). Profiles must therefore be measured on the pods of the
  session, at its start, and all runs of a matrix use the same pods.
- **Short prefill windows.** Prefill of one request takes well under a second on the faster GPUs,
  so the prefill power values rest on 0 to 8 one-second samples and are not used.
- **Node agent.** The RunPod image runs nginx on the exposed ports and forwards 8081 to 8080. The
  agent therefore listens on 8082 and `runpod_node.sh` points nginx's 8081 to it; reading
  `/time` from another pod through the proxy took 0.13 to 0.32 s.
- **T4** measured on Kaggle (`scripts/kaggle_profile.py`, committed notebook version, driver 580.178,
  same llama.cpp build b11923). Unlike the newer GPUs, batching helps it little (1.96 times at four
  requests, flat between 2 and 3), and its energy per token at four requests is 1.02 J, the highest of
  the four GPUs.
- **P100**: Kaggle no longer offers the P100 accelerator (only T4 x2), so it is not measured.
