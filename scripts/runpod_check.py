"""Check one RunPod node before the Phase 4 measurement session.

    python scripts/runpod_check.py [--url http://127.0.0.1:8080] [--out results/runpod_check]

Run it on the pod after infra/runpod_node.sh has started llama-server. It answers:
  1. Power reading: does nvidia-smi report the GPU's power draw inside the pod, once per
     second, at idle and under load? (all energy metrics depend on it)
  2. Node speeds: decode and prompt speed, one request at a time (median of 3 after a
     warm-up, same method as the Kaggle prototype) -> speed_tps and prefill_tps for config.yaml
  3. Batching: total decode speed with 1, 2 and 4 requests at the same time, to see how far
     the equal-sharing assumption A5 is from the real GPU.
Writes <gpu>.json and <gpu>_power.csv to --out.
"""
import argparse, json, statistics, subprocess, sys, threading, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LONG_DOC = " ".join(["Distributed systems share work among many machines to finish jobs faster."] * 110)


def post(url, body, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ask(url, text, max_tokens=128):
    return post(url + "/v1/chat/completions",
                {"model": "qwen3-8b", "max_tokens": max_tokens, "temperature": 0, "stream": False,
                 "messages": [{"role": "system", "content": "/no_think"}, {"role": "user", "content": text}]})


def gpu_info():
    q = "name,memory.total,power.limit,driver_version"
    line = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip().splitlines()[0]
    return dict(zip(q.split(","), [p.strip() for p in line.split(",")]))


def speeds(url):
    ask(url, f"Read this document and summarize it.\n\n{LONG_DOC}")            # warm-up
    dec, pre = [], []
    for k in range(3):
        t = ask(url, f"[{k}] Read this document and summarize it.\n\n{LONG_DOC}").get("timings", {})
        dec.append(t.get("predicted_per_second", 0.0))
        pre.append(t.get("prompt_per_second", 0.0))
    return {"speed_tps": round(statistics.median(dec), 2), "prefill_tps": round(statistics.median(pre), 1),
            "decode_runs": [round(x, 2) for x in dec], "prefill_runs": [round(x, 1) for x in pre]}


def batch_speed(url, n, max_tokens=256):
    """n requests at once; total generated tokens / wall time of the decode part."""
    res = [None] * n

    def one(i):
        res[i] = ask(url, f"[b{n}-{i}] Write a long story about a lighthouse keeper.", max_tokens)
    th = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    t0 = time.time()
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = time.time() - t0
    toks = sum(r["usage"]["completion_tokens"] for r in res)
    per_req = [r.get("timings", {}).get("predicted_per_second", 0.0) for r in res]
    return {"concurrent": n, "total_tokens": toks, "wall_s": round(wall, 2),
            "total_tps": round(toks / wall, 2), "per_request_tps": round(statistics.mean(per_req), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--out", default=str(ROOT / "results" / "runpod_check"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    info = gpu_info()
    tag = info["name"].replace("NVIDIA ", "").replace(" ", "_").lower()
    power_csv = out / f"{tag}_power.csv"
    logger = subprocess.Popen([sys.executable, str(ROOT / "infra" / "energy_logger.py"), "--out", str(power_csv)])
    marks = {}

    marks["idle_start"] = time.time(); time.sleep(15); marks["idle_end"] = time.time()
    print("idle done", flush=True)
    marks["speed_start"] = time.time(); sp = speeds(a.url); marks["speed_end"] = time.time()
    print("speeds", sp, flush=True)
    batch = []
    for n in (1, 2, 4):
        s = time.time(); b = batch_speed(a.url, n); b["start"], b["end"] = s, time.time(); batch.append(b)
        print("batch", n, b, flush=True)

    time.sleep(2)
    logger.terminate(); logger.wait()

    rows = [l.split(",") for l in power_csv.read_text().splitlines()[1:]]
    samples = [(float(r[0]), float(r[3])) for r in rows]

    def mean_w(t0, t1):
        w = [p for t, p in samples if t0 <= t <= t1]
        return (round(statistics.mean(w), 1), len(w)) if w else (None, 0)
    gaps = [b[0] - a_[0] for a_, b in zip(samples, samples[1:])]
    power = {"samples": len(samples),
             "median_interval_s": round(statistics.median(gaps), 3) if gaps else None,
             "idle_w": mean_w(marks["idle_start"], marks["idle_end"]),
             "single_request_w": mean_w(marks["speed_start"], marks["speed_end"]),
             "batch_w": {b["concurrent"]: mean_w(b["start"], b["end"]) for b in batch}}
    one = batch[0]["total_tps"]
    for b in batch:
        b["speedup_vs_1"] = round(b["total_tps"] / one, 2) if one else None
        del b["start"], b["end"]
    result = {"gpu": info, "llama_server": a.url, "speeds": sp, "batching": batch, "power": power,
              "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / f"{tag}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
