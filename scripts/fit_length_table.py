"""Fit the output-length predictor on the training slices of the workloads (bench/workloads/*_train.jsonl)
and write strategies/length_table.json. The prompt length is estimated as the router does
(words x 1.3), so the predictor sees the same input in training and in the runs."""
import glob, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strategies.length_predictor import LengthPredictor

rows = []
for f in glob.glob(str(ROOT / "bench" / "workloads" / "*_train.jsonl")):
    for line in open(f):
        r = json.loads(line)
        text = " ".join(m["content"] for m in r["messages"] if isinstance(m.get("content"), str))
        rows.append({"prompt_len": max(1, int(len(text.split()) * 1.3)), "output_len": r["output_len"], "req_type": r["req_type"]})
p = LengthPredictor.fit(rows)
(ROOT / "strategies" / "length_table.json").write_text(json.dumps(p.to_dict(), indent=2))
print(f"length table from {len(rows)} training requests")
