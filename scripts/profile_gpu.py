"""Profile one GPU type for the simulator and the strategies (Phase 4, task #46).

    python scripts/profile_gpu.py [--url http://127.0.0.1:8080] [--out results/profiles] [--repeats 3]

Run on the node after infra/runpod_node.sh has started llama-server (4 slots). Measures:
  * s(n): total decode speed with n = 1, 2, 3, 4 requests running at the same time
          (tokens/s summed over the requests), median of --repeats;
  * per-request decode speed at each n;
  * p: prompt (prefill) speed for a short prompt (~300 tokens), a medium one (~2,000)
       and a long one (~6,000), one request at a time;
  * power: idle (after a 30 s wait), and the mean during each measurement.
Writes <gpu>.json and <gpu>_power.csv. The simulator interpolates s(n) linearly and uses
s(4) above 4 running requests (assumption A5).
"""
import argparse, json, statistics, subprocess, sys, threading, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GPU_INDEX = 0
SENT = "Distributed systems share work among many machines to finish jobs faster. "   # ~13 tokens
PROMPTS = {"short": SENT * 23, "medium": SENT * 150, "long": SENT * 460}
OUT_TOKENS = 256


def post(url, body, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ask(url, text, max_tokens):
    return post(url + "/v1/chat/completions",
                {"model": "qwen3-8b", "max_tokens": max_tokens, "temperature": 0, "stream": False,
                 "ignore_eos": True,
                 "messages": [{"role": "system", "content": "/no_think"}, {"role": "user", "content": text}]})


def gpu_info():
    q = "name,memory.total,power.limit,driver_version,clocks.max.sm"
    line = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip().splitlines()[GPU_INDEX]
    return dict(zip(q.split(","), [p.strip() for p in line.split(",")]))


def server_props(url):
    try:
        with urllib.request.urlopen(url + "/props", timeout=5) as r:
            p = json.loads(r.read())
        return {"total_slots": p.get("total_slots"), "n_ctx": (p.get("default_generation_settings") or {}).get("n_ctx"),
                "build_info": p.get("build_info")}
    except Exception as e:
        return {"error": str(e)}


def concurrent(url, n, tag):
    """n requests at once, each generating OUT_TOKENS tokens; returns total and per-request tokens/s."""
    res = [None] * n

    def one(i):
        res[i] = ask(url, f"[{tag}-{n}-{i}] Write a long story about a lighthouse keeper.", OUT_TOKENS)
    th = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    t0 = time.time()
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = time.time() - t0
    toks = sum(r["usage"]["completion_tokens"] for r in res)
    per = [r.get("timings", {}).get("predicted_per_second", 0.0) for r in res]
    return toks / wall, statistics.mean(per), toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--out", default=str(ROOT / "results" / "profiles"))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--idle-wait", type=float, default=30.0)
    ap.add_argument("--gpu-index", type=int, default=0, help="GPU whose power is reported (the one the server uses)")
    a = ap.parse_args()
    global GPU_INDEX
    GPU_INDEX = a.gpu_index
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    info = gpu_info()
    tag = info["name"].replace("NVIDIA ", "").replace("Tesla ", "").replace(" ", "_").lower()
    power_csv = out / f"{tag}_power.csv"
    logger = subprocess.Popen([sys.executable, str(ROOT / "infra" / "energy_logger.py"), "--out", str(power_csv)])
    marks = {}

    time.sleep(a.idle_wait)                                     # GPU leaves the high-power state after loading
    marks["idle"] = (time.time(), time.time() + 15); time.sleep(15)
    ask(a.url, "warm-up: say hello", 16)

    prefill = {}
    for name, text in PROMPTS.items():
        t0 = time.time(); runs = []; ptoks = 0
        for k in range(a.repeats):
            r = ask(a.url, f"[p{k}] Summarize this text.\n\n{text}", 8)
            runs.append(r.get("timings", {}).get("prompt_per_second", 0.0)); ptoks = r["usage"]["prompt_tokens"]
        prefill[name] = {"prompt_tokens": ptoks, "prefill_tps": round(statistics.median(runs), 1),
                         "runs": [round(x, 1) for x in runs]}
        marks[f"prefill_{name}"] = (t0, time.time())
        print("prefill", name, prefill[name], flush=True)

    decode = {}
    for n in (1, 2, 3, 4):
        t0 = time.time(); tot, per = [], []
        for k in range(a.repeats):
            x, y, _ = concurrent(a.url, n, f"d{k}")
            tot.append(x); per.append(y)
        decode[n] = {"total_tps": round(statistics.median(tot), 2), "per_request_tps": round(statistics.median(per), 2),
                     "runs": [round(x, 2) for x in tot]}
        marks[f"decode_{n}"] = (t0, time.time())
        print("decode", n, decode[n], flush=True)

    time.sleep(2)
    logger.terminate(); logger.wait()
    rows = [l.split(",") for l in power_csv.read_text().splitlines()[1:]]
    samples = [(float(r[0]), float(r[3])) for r in rows if int(r[2]) == a.gpu_index]

    def mean_w(t0, t1):
        w = [p for t, p in samples if t0 <= t <= t1]
        return {"mean_w": round(statistics.mean(w), 1) if w else None, "samples": len(w)}

    s1 = decode[1]["total_tps"]
    result = {"gpu": info, "server": server_props(a.url), "model": "Qwen3-8B Q4_K_M", "output_tokens": OUT_TOKENS,
              "speed_tps": s1, "prefill_tps": prefill["medium"]["prefill_tps"],
              "decode_by_concurrency": {str(n): d for n, d in decode.items()},
              "speedup_vs_1": {str(n): round(d["total_tps"] / s1, 2) for n, d in decode.items()},
              "prefill_by_length": prefill,
              "power": {k: mean_w(*v) for k, v in marks.items()},
              "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / f"{tag}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
