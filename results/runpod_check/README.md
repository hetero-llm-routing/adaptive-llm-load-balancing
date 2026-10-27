# RunPod check before the Phase 4 session (Oct 27)

Short check on RunPod with one pod per GPU, using `infra/runpod_node.sh` (build llama.cpp, download
Qwen3-8B Q4_K_M, start llama-server with the Kaggle settings: `-ngl 99 -c 8192 --jinja --metrics`)
and `scripts/runpod_check.py`. llama.cpp commit de5122d (2026-10-27), driver 595.91.07, CUDA 12.8,
template `runpod/pytorch:1.4.1-cu1281-torch280-ubuntu2404`.

## Node speeds (one request at a time, median of 3)

| GPU | VRAM | Power limit | Decode `speed_tps` | Prompt `prefill_tps` |
| --- | --- | --- | --- | --- |
| RTX 4090 | 24 GB | 450 W | 149.3 | 11,468 |
| RTX A6000 | 48 GB | 300 W | 98.1 | 4,679 |
| RTX 4000 Ada (spare) | 20 GB | 130 W | 61.1 | 3,803 |
| RTX 2000 Ada | 16 GB | 70 W | 40.4 | 2,016 |

The RTX A6000 (Ampere workstation card) replaces the RTX A4500 in the testbed: the A4500 was
"out of capacity" for the whole session, and the experiments need the same card in every session.
Decision by Eyad on Oct 27; Sections 1 and 2 of the draft are updated (48 GB, 768 GB/s [24],
decode bound about 154 steps per second). The RTX 4000 Ada (20 GB, 360 GB/s) was measured later
as a spare card; it is not part of the testbed. Its run waited 30 s after the server started, so
its idle value is clean.
The 4090 decodes 3.7 times faster than the RTX 2000 Ada (prompt processing: 5.7 times), and the
A6000 sits in between. Section 2.2 gives bandwidth upper bounds of 200 (4090), 154 (A6000) and 45
(2000 Ada) steps per second; the measured 149, 98 and 40 are below them, as expected.

## Batching (assumption A5)

Total decode speed with 1, 2 and 4 requests at the same time (256 output tokens each):

| GPU | 1 | 2 | 4 | Speed-up with 4 |
| --- | --- | --- | --- | --- |
| RTX 4090 | 137 | 268 | 399 | 2.91 |
| RTX A6000 | 94 | 180 | 260 | 2.76 |
| RTX 4000 Ada (spare) | 61 | 120 | 202 | 3.30 |
| RTX 2000 Ada | 41 | 80 | 133 | 3.24 |

Running requests do not split a fixed speed: total throughput grows almost linearly up to 2 and
about 3 times with 4, while each request slows only a little (4090: 150 to 116 tokens/s). This
supports the note on A5 in Section 3.2; the simulator should use a measured batching curve
rather than equal sharing.

## Power reading

`nvidia-smi` reports power inside the pods, one sample per second (median interval 1.000 s).

| GPU | Idle | One request | Under load (1 to 4 requests) |
| --- | --- | --- | --- |
| RTX 4090 | 24 W | 156 W | 280 to 310 W |
| RTX A6000 | 37.5 W (see below) | 214 W | 277 to 295 W |
| RTX 4000 Ada (spare) | 13.9 W | 97 W | 121 to 128 W |
| RTX 2000 Ada | 7.5 W | 56 W | 68 to 69 W (at its 70 W limit) |

The 4090 runs finish in about 2 s, so its load values rest on few samples; the Phase 4 runs are
long enough for this not to matter. Timestamps in the power CSVs are rounded to the 1 s steps.

On the A6000 the GPU stayed at about 110 W for the first 13 s after the server had loaded the
model, then dropped to 37.5 W (state P8), so the `idle_w` value in its JSON (104.8 W) is not the
idle power. The Phase 4 runs must wait about 30 s after the servers start before measuring idle
power or starting a run.

## Findings for the Phase 4 session

1. **GPU availability changes by the minute.** RTX A4500, A4000, A5000, 4000 Ada and 3090 were all
   "out of capacity" for most of the hour, and appeared only for a minute or two. The session needs
   a fallback list per role (fast / middle / slow) and should start pods as soon as they appear.
2. **Network volumes tie pods to one data center**, which makes the scarce cards even harder to get.
   The setup script replaces the volume: a new pod is ready in about 1 minute when the
   llama-server binary is copied from another pod of the same GPU generation, about 15 minutes
   when it is built.
3. **Reading node state through the RunPod HTTP proxy is slow**: `/metrics` from one pod to another
   took 0.28 to 0.92 s, against the router's 1 s timeout, and the router reads every node before
   each decision. For the real runs the router should poll node state in the background (or use
   direct TCP ports) so this delay is not added to every request.
4. **Build time depends on the pod's CPUs**: 12 vCPUs (4090 pod) built llama.cpp in about 12 min,
   8 vCPUs (A6000 pod) in about 17 min. Pods of the same GPU generation can reuse one build.
5. **Getting the code onto the pods**: the repository is private, so the pods cannot clone it
   without a key; today the files were copied in through Jupyter.
