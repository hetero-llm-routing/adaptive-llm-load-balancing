# Case study, exploratory run (superseded by `results/case_study_en/`)

The paper (Section 7.2) uses the final run in `results/case_study_en/`: English questions, profiles
measured for Qwen3-14B, five bursts per strategy. This first run used Arabic questions and the
Qwen3-8B profiles; it is kept for reference.

Run on 13 Nov 2026 on RunPod. A gateway machine without a GPU runs the router (marginal-cost
routing), the RAG step and Open WebUI; three GPU nodes run Qwen3-14B (Q4_K_M, llama.cpp, 3 slots):
RTX 4090, RTX A6000 and RTX 2000 Ada, the GPU types of the testbed. The documents are 10 English
Wikipedia articles (load balancing, GPUs, CUDA, RAG, LLMs, transformers, cloud computing, data
centers, quantization, energy efficiency); questions are in Arabic and English. Every model call,
including the RAG helper calls (query translation, relevance check), goes through the router.

## Figures

- `fig_rag_modes.png`: the two answer modes. (a) documents only, answer from the documents;
  (b) documents only, a question the documents do not answer is refused; (c) documents + model,
  the answer is split into a part from the documents and a part from the model, with a note that
  the model's knowledge may be outdated.
- `fig_monitor_mc_rr.png`: the router's live page during a burst of 27 questions, with
  marginal-cost routing (a) and round robin (b): requests running on each GPU and the last decisions.
- `runpod_cluster.jpg`: the four machines (gateway and three GPU nodes) running.
- `screenshots/q1.png` to `q10.png`: the nine test questions (q9 is q7 in documents + model mode);
  `screenshots/monitor_*.png`: three frames of the live page per strategy.

## Burst of 27 questions sent at once

| Strategy | All answered after | Per question: mean / p95 / max | Model calls on 4090 / A6000 / 2000 Ada |
|---|---|---|---|
| Marginal cost | 21.7 s | 14.1 / 21.3 / 21.7 s | 37 / 24 / 8 |
| Round robin | 36.5 s | 14.5 / 29.1 / 36.4 s | 23 / 23 / 23 |

Round robin gives the RTX 2000 Ada the same share as the faster GPUs, so its queue sets the time
of the whole burst; marginal-cost routing sends most calls to the RTX 4090 and RTX A6000 and
finishes the burst 40% sooner. One burst per strategy, so these numbers illustrate the behaviour
measured in Section 6 and are not a replacement for it.

## Three bursts per strategy, with energy (`case_bursts.csv`)

The burst was then repeated three times per strategy, alternating the strategies, with the router
restarted and 25 s of idle time before each burst. Energy is the measured power of the three GPUs
(1 s samples from each node agent) over the burst, divided by the tokens the router reports.

| Strategy | All answered after (s) | p95 per question (s) | J per token |
|---|---|---|---|
| Marginal cost | 21.6 (19.7–22.6) | 17.6 (16.0–18.7) | 2.47 (2.30–2.67) |
| Round robin | 34.9 (25.9–46.2) | 29.3 (23.0–38.4) | 2.90 (2.59–3.10) |

Means of the three bursts, with the range. Marginal-cost routing finished the bursts 38% sooner,
with a 40% lower p95 and 15% less energy per token. Round robin varies more between bursts because
its result depends on how many long answers land on the RTX 2000 Ada.
