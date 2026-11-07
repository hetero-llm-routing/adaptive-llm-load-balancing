"""Build the workload files of Section 5.2 (task #50). Run once on a machine with Internet
(a RunPod pod); the files are then copied with the code and replayed by bench/run_matrix.py.

    python bench/make_workloads.py --out bench/workloads [--n 1500] [--n-train 1500] [--seed 0]

Families (one JSONL file each, plus a disjoint *_train.jsonl slice for the length predictor):
  chat  ShareGPT: each user turn with its own reply is one independent request (A3, A7).
  doc   LongBench: summary and question tasks over one long input, prompt <= 6,000 tokens.
  rag   HotpotQA questions; retrieval done offline: BM25 over a fixed pool of paragraphs,
        top-5 passages stored in the prompt, so the retriever adds no variance.
  mixed 70% chat, 15% doc, 15% rag, drawn from the files above.

Every row: request_id, req_type, messages, prompt_len, output_len, source.
  prompt_len  exact, counted with the Qwen3-8B tokenizer on the rendered chat template
  output_len  tokens the request generates (sent as max_tokens with ignore_eos, so every
              strategy sees the same work): the reference answer length for chat and doc,
              clipped; for rag, drawn (seeded) from the Azure conversation trace [7]
"""
import argparse, hashlib, io, json, random, re, urllib.request, zipfile
from pathlib import Path

SHAREGPT = ("anon8231489123/ShareGPT_Vicuna_unfiltered", "ShareGPT_V3_unfiltered_cleaned_split.json")
LONGBENCH = ("THUDM/LongBench", "data.zip")
LONGBENCH_TASKS = {"gov_report": "Write a one-page summary of the report.",
                   "qmsum": None, "multi_news": "Write a one-page summary of all the news.",
                   "narrativeqa": None, "qasper": None, "multifieldqa_en": None}
HOTPOT = ("hotpotqa/hotpot_qa", "distractor/validation-00000-of-00001.parquet")
AZURE_CONV = "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_conv.csv"
MAX_PROMPT = {"chat": 2048, "doc": 6000, "rag": 4096}
OUT_CLIP = {"chat": (16, 1024), "doc": (32, 512), "rag": (16, 512)}
SYSTEM = "You are a helpful assistant."


def hf_file(repo, name):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, name, repo_type="dataset")


def fetch(url):
    with urllib.request.urlopen(url, timeout=600) as r:
        return r.read()


class Tok:
    def __init__(self):
        from transformers import AutoTokenizer
        self.t = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    def prompt_len(self, messages):
        text = self.t.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        return len(self.t(text, add_special_tokens=False)["input_ids"])

    def n(self, text):
        return len(self.t(text, add_special_tokens=False)["input_ids"])


def clip(x, fam):
    lo, hi = OUT_CLIP[fam]
    return max(lo, min(hi, int(x)))


def chat_items(tok, rng):
    data = json.load(open(hf_file(*SHAREGPT)))
    rng.shuffle(data)
    for conv in data:
        turns = conv.get("conversations") or []
        for a, b in zip(turns, turns[1:]):
            if a.get("from") == "human" and b.get("from") == "gpt" and a["value"].strip() and b["value"].strip():
                msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": a["value"]}]
                yield msgs, tok.n(b["value"]), "sharegpt:" + conv.get("id", "")


def doc_items(tok, rng):
    z = zipfile.ZipFile(hf_file(*LONGBENCH))
    rows = []
    for task, instr in LONGBENCH_TASKS.items():
        name = [n for n in z.namelist() if n.endswith(f"/{task}.jsonl")][0]
        for line in z.open(name):
            r = json.loads(line)
            q = instr or f"Answer the question based on the text.\n\nQuestion: {r['input']}"
            ans = (r.get("answers") or [""])[0]
            rows.append(([{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": f"{r['context']}\n\n{q}"}], ans, f"longbench:{task}:{r['_id']}"))
    rng.shuffle(rows)
    for msgs, ans, src in rows:
        yield msgs, tok.n(ans), src


def rag_items(tok, rng, azure_out):
    from rank_bm25 import BM25Okapi
    import pandas as pd
    df = pd.read_parquet(hf_file(*HOTPOT))
    data = [{"_id": r["id"], "question": r["question"],
             "context": list(zip(r["context"]["title"], r["context"]["sentences"]))} for _, r in df.iterrows()]
    pool, seen = [], set()
    for q in data:                                  # fixed corpus: every paragraph of the dev set
        for title, sents in q["context"]:
            if title not in seen:
                seen.add(title); pool.append(f"{title}: {''.join(sents)}")
    bm25 = BM25Okapi([re.findall(r"\w+", p.lower()) for p in pool])
    rng.shuffle(data)
    for q in data:
        scores = bm25.get_scores(re.findall(r"\w+", q["question"].lower()))
        top = sorted(range(len(pool)), key=lambda i: -scores[i])[:5]
        ctx = "\n\n".join(f"[{k + 1}] {pool[i]}" for k, i in enumerate(top))
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Use the passages to answer the question and explain the answer.\n\n{ctx}\n\nQuestion: {q['question']}"}]
        yield msgs, rng.choice(azure_out), "hotpotqa:" + q["_id"]


def take(gen, fam, tok, n_total):
    out = []
    for msgs, o, src in gen:
        p = tok.prompt_len(msgs)
        if p > MAX_PROMPT[fam] or p + clip(o, fam) > 8192:          # 8,192-token slots (Section 5.1)
            continue
        out.append({"req_type": fam, "messages": msgs, "prompt_len": p, "output_len": clip(o, fam), "source": src})
        if len(out) == n_total:
            break
    return out


def write(path, rows, fam):
    with open(path, "w") as f:
        for i, r in enumerate(rows):
            r = dict(r, request_id=f"{fam}-{i:05d}")
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/workloads")
    ap.add_argument("--n", type=int, default=1500, help="test requests per family")
    ap.add_argument("--n-train", type=int, default=1500, help="disjoint requests per family for the predictor")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng, tok = random.Random(a.seed), Tok()
    raw = fetch(AZURE_CONV); (out / "azure_conv.csv").write_bytes(raw)   # also used for the burst arrivals
    azure = raw.decode().splitlines()[1:]
    azure_out = [int(l.split(",")[2]) for l in azure if l.count(",") >= 2]
    gens = {"chat": chat_items(tok, rng), "doc": doc_items(tok, rng), "rag": rag_items(tok, rng, azure_out)}
    meta, test = {}, {}
    for fam, g in gens.items():
        rows = take(g, fam, tok, a.n + a.n_train)
        n_test = a.n if len(rows) >= a.n + a.n_train else len(rows) // 2   # small source: half and half
        test[fam], train = rows[:n_test], rows[n_test:]
        meta[fam] = {"test": len(test[fam]), "train": len(train),
                     "sha256": write(out / f"{fam}.jsonl", test[fam], fam)}
        write(out / f"{fam}_train.jsonl", train, fam + "-train")
        ps = sorted(r["prompt_len"] for r in test[fam]); os_ = sorted(r["output_len"] for r in test[fam])
        q = lambda v, x: v[min(len(v) - 1, int(x * len(v)))]
        meta[fam].update(prompt_p50=q(ps, .5), prompt_p95=q(ps, .95), output_p50=q(os_, .5), output_p95=q(os_, .95))
        print(fam, meta[fam], flush=True)
    n = a.n                                          # mixed: 70 / 15 / 15
    n_chat, n_doc = round(.70 * n), min(round(.15 * n), len(test["doc"]))
    mixed = rng.sample(test["chat"], n_chat) + rng.sample(test["doc"], n_doc) + rng.sample(test["rag"], n - n_chat - n_doc)
    rng.shuffle(mixed)
    meta["mixed"] = {"test": len(mixed), "sha256": write(out / "mixed.jsonl", mixed, "mixed")}
    (out / "workloads_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
