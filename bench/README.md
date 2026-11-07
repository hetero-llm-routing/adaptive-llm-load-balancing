# Placeholder: Mohammed's workload generator and load tester go here (Phase 3).


## Experiment runner (tasks #50, #60)

    python bench/make_workloads.py --out bench/workloads          # once, on a pod with Internet
    python infra/cluster_up.py --node rtx4090=URL --node rtxa6000=URL --node rtx2000ada=URL --out config_run.yaml
    python bench/calibrate.py --config config_run.yaml            # round-robin saturation per workload
    python bench/run_matrix.py --matrix bench/matrix_example.yaml # the 280 runs; resumes if interrupted

`make_workloads.py` builds chat (ShareGPT), doc (LongBench, prompt <= 6,000 tokens), rag (HotpotQA,
BM25 top-5 offline) and mixed (70/15/15) files, with exact token counts and a disjoint training
slice for the length predictor. `replay.py` runs one strategy on a seeded open-loop schedule (same
requests and arrival times for every strategy), waits for an idle cluster first, and collects the
router CSV, a client log, the power of every node (clock offsets corrected) and the energy split.
