"""Output-length predictor for the predictive strategy (Osama).

The router knows a request's prompt length when it arrives, but not how many
tokens the model will generate. This module estimates the output length from
what is known: the prompt length and, when available, the request type.

Method (cheap on purpose, so it adds almost nothing to each decision):
  1. Put prompt lengths into a few buckets on a log scale.
  2. Store the mean output length seen in each bucket in earlier runs.
  3. Shrink each bucket mean toward the overall mean when the bucket has few
     samples:  pred = (n * bucket_mean + K * global_mean) / (n + K).

The table is fitted from result CSV files of earlier runs (never from the run
being evaluated) and saved as JSON, so every run uses the same fixed table.

Fit a table:
    python -m strategies.length_predictor fit "results/kaggle_smoke/*.csv" \
        -o strategies/length_table.json
Check it:
    python -m strategies.length_predictor eval strategies/length_table.json "results/*/*.csv"
"""
import argparse, bisect, csv, glob, json
from pathlib import Path

# Upper edges of the prompt-length buckets, in tokens (the last bucket is open).
BUCKET_EDGES = [32, 128, 512, 2048]
SHRINK_K = 5            # pseudo-samples that pull a small bucket toward the global mean
DEFAULT_LEN = 128.0     # used before any table is fitted (= max_tokens in our runs)


def bucket(prompt_len):
    return bisect.bisect_left(BUCKET_EDGES, int(prompt_len))


class LengthPredictor:
    def __init__(self, table=None):
        t = table or {}
        self.global_mean = float(t.get("global_mean", DEFAULT_LEN))
        # keys: "<req_type>|<bucket>" or "*|<bucket>"; values: predicted output tokens
        self.cells = {k: float(v) for k, v in t.get("cells", {}).items()}

    def predict(self, prompt_len, req_type="chat"):
        b = bucket(prompt_len)
        for key in (f"{req_type}|{b}", f"*|{b}"):
            if key in self.cells:
                return self.cells[key]
        return self.global_mean

    # ---------- fitting ----------
    @staticmethod
    def fit(rows):
        """rows: dicts with prompt_len, output_len and optionally req_type."""
        rows = [r for r in rows if r.get("output_len") not in (None, "")]
        if not rows:
            return LengthPredictor()
        outs = [float(r["output_len"]) for r in rows]
        g = sum(outs) / len(outs)
        groups = {}
        for r in rows:
            b = bucket(float(r["prompt_len"]))
            groups.setdefault(f"*|{b}", []).append(float(r["output_len"]))
            if r.get("req_type"):
                groups.setdefault(f"{r['req_type']}|{b}", []).append(float(r["output_len"]))
        cells = {k: (sum(v) + SHRINK_K * g) / (len(v) + SHRINK_K) for k, v in groups.items()}
        counts = {k: len(v) for k, v in groups.items()}
        return LengthPredictor({"global_mean": g, "cells": cells, "counts": counts})

    def to_dict(self):
        return {"global_mean": self.global_mean, "cells": self.cells,
                "bucket_edges": BUCKET_EDGES, "shrink_k": SHRINK_K}

    @staticmethod
    def load(path):
        p = Path(path)
        return LengthPredictor(json.loads(p.read_text())) if p.exists() else LengthPredictor()


def read_rows(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        with open(f, newline="") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def mae(pred, rows):
    errs = [abs(pred.predict(float(r["prompt_len"]), r.get("req_type") or "chat") - float(r["output_len"]))
            for r in rows if r.get("output_len") not in (None, "")]
    return sum(errs) / len(errs) if errs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit"); f.add_argument("csv_glob"); f.add_argument("-o", "--out", required=True)
    e = sub.add_parser("eval"); e.add_argument("table"); e.add_argument("csv_glob")
    a = ap.parse_args()
    if a.cmd == "fit":
        rows = read_rows(a.csv_glob)
        p = LengthPredictor.fit(rows)
        Path(a.out).write_text(json.dumps(p.to_dict(), indent=2))
        print(f"fitted on {len(rows)} rows -> {a.out}; global mean {p.global_mean:.1f} tokens")
    else:
        p, rows = LengthPredictor.load(a.table), read_rows(a.csv_glob)
        base = LengthPredictor({"global_mean": p.global_mean})
        print(f"rows {len(rows)}  MAE table {mae(p, rows):.1f}  MAE global-mean only {mae(base, rows):.1f} tokens")


if __name__ == "__main__":
    main()
