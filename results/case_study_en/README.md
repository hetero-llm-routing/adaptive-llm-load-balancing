# Case study (final): private RAG assistant, English questions, Qwen3-14B profiles (Section 7.2)

Run on 13 Nov 2026 on RunPod. This run replaces the earlier exploratory run in `results/case_study/`
(Arabic questions, router with the Qwen3-8B profiles).

## Setup

- A CPU gateway (EU-RO-1) runs the router, the RAG step and Open WebUI. Three GPU nodes run
  Qwen3-14B (Q4_K_M) with llama.cpp and three slots: RTX 2000 Ada (EU-RO-1), RTX A6000 (US-TX-1)
  and RTX 4090 (EUR-IS-1), the last two reached over the internet.
- Each GPU was profiled again for this model and three slots (`profiles_qwen3_14b.json`), and
  marginal-cost routing ran with `LB_SLOTS=3` (`strategies/marginal_cost.py`).
- Nine English questions (six on topics of the ten Wikipedia documents, three outside them),
  sent three times at once: 27 questions per burst, every fourth in documents + model mode.
- Five bursts per strategy, alternating marginal cost and round robin, router restarted and 25 s
  idle before each burst. Every model call went through the router; no call failed.

## Result (`case_bursts_en.csv`, Table VII of the paper)

| Strategy | Burst time, s (range) | Mean per question, s | E2E p95, s |
|---|---|---|---|
| Round robin | 34.8 (31.9–37.3) | 13.7 | 25.1 |
| Marginal cost | 22.6 (20.0–27.4) | 12.8 | 19.3 |
| Change | −35% | −7% | −23% |

Every marginal-cost burst finished before every round-robin burst. Energy is not reported: the power
logs of the two remote GPUs could not be collected reliably through the proxy.

## Figures

- `fig4_rag_en.png` (Fig. 4): the two answer modes, English questions.
- `fig5_monitor_en.png` (Fig. 5): the router's live page during a burst, marginal cost vs round robin
  ("in flight" = running in a slot or waiting for one).
- `screenshots/`: the three chat screenshots and two frames of the live page per strategy.

## Note on a faulty node

A first RTX 4090 pod returned HTTP 404 from the RunPod proxy for requests that waited longer than
about 20 s for a slot. Its 12 bursts were discarded, the pod was replaced, and the router now logs
every non-JSON answer from a node (`BADRESP` in `router/router.py`).
