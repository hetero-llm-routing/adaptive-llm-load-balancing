"""Phase 3 prototype on a free Kaggle notebook with two T4 GPUs (tasks #34 and #35).

What it does, in order:
  1. Reuses llama-server and the Qwen3-8B model from the Kaggle dataset made after the
     smoke test (falls back to the smoke test, which builds and downloads them).
  2. Starts the energy logger (power of both GPUs, once per second).
  3. Scenario "same":  both GPUs serve the full model on the GPU (two equal nodes).
     Scenario "mixed": node t4_1 keeps only LB_SLOW_NGL layers on the GPU and runs the
     rest on the CPU, so it is slower. This only imitates a mixed cluster to check that
     the strategies react to speed; it is not a result for the paper.
  4. In each scenario: measures every node's decode and prompt speed (tokens/s, one
     request at a time), writes them into the run config, then runs round_robin,
     weighted, least_load and predictive on the same request list.
  5. Fills energy_j in every result file (method "tokens"; "equal" is kept in the summary
     for the team's decision on task #26) and prints a summary.

All files go to /kaggle/working/prototype/ and are zipped to prototype_results.zip.
The request sender here is a short stand-in; Mohammed's experiment runner replaces it
in Phase 4.

Kaggle: Accelerator = GPU T4 x2, Internet = On, input dataset llm-lb-runtime attached.
Run with Save Version > Save & Run All so it survives a closed browser.
Settings (environment variables): LB_REQUESTS (60), LB_RATE (0.6 requests/s),
LB_SLOW_NGL (28), LB_SCENARIOS ("same,mixed"), LB_STRATEGIES.
"""
import asyncio, glob, json, os, random, shutil, statistics, subprocess, sys, threading, time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
WORK = Path(os.environ.get("LB_WORK", "/kaggle/working"))
OUT = WORK / "prototype"
RUNTIME = WORK / "runtime"
N_REQ = int(os.environ.get("LB_REQUESTS", "60"))
RATE = float(os.environ.get("LB_RATE", "0.6"))
SLOW_NGL = os.environ.get("LB_SLOW_NGL", "28")
SCENARIOS = os.environ.get("LB_SCENARIOS", "same,mixed").split(",")
STRATEGIES = os.environ.get("LB_STRATEGIES", "round_robin,weighted,least_load,predictive").split(",")
SEED = 1
ROUTER = "http://127.0.0.1:8000/v1/chat/completions"

LONG_DOC = " ".join(["Distributed systems share work among many machines to finish jobs faster."] * 110)
QUESTIONS = ["Explain what a GPU is in two sentences.",
             "Summarize the idea of load balancing for a new student.",
             "Write a short formal email asking for a meeting next week.",
             "What is the difference between a process and a thread?",
             "Give three tips for writing a clear research abstract.",
             "Describe how a queue works, with one everyday example.",
             "List the main parts of a computer and what each one does.",
             "Explain why caching makes programs faster."]


def step(title):
    print(f"\n========== {title} ==========", flush=True)


def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def wait_ok(url, seconds):
    end = time.time() + seconds
    while time.time() < end:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def post_json(url, body, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ensure_runtime():
    server = sorted(glob.glob("/kaggle/input/**/llama-server", recursive=True))
    model = sorted(glob.glob("/kaggle/input/**/*Q4_K_M*.gguf", recursive=True))
    if not (server and model):
        print("runtime dataset not attached: running the smoke test to build and download (about 30 min)")
        import runpy
        runpy.run_path(str(ROOT / "scripts" / "kaggle_smoke_test.py"), run_name="__main__")
        server = [str(RUNTIME / "llama-server")]
        model = sorted(glob.glob(f"{RUNTIME}/**/*Q4_K_M*.gguf", recursive=True))
    RUNTIME.mkdir(exist_ok=True)
    exe = RUNTIME / "llama-server"
    if not exe.exists():
        shutil.copy(server[0], exe)
        exe.chmod(0o755)
    return exe, model[0]


class Node:
    def __init__(self, index, exe, model):
        self.index, self.exe, self.model = index, exe, model
        self.name, self.port, self.proc, self.ngl = f"t4_{index}", 8081 + index, None, "99"

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def start(self, ngl):
        self.stop()
        self.ngl = ngl
        log = open(OUT / f"server_{self.name}_ngl{ngl}.log", "w")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(self.index))
        self.proc = subprocess.Popen([str(self.exe), "-m", self.model, "--host", "127.0.0.1", "--port", str(self.port),
                                      "-ngl", str(ngl), "-c", "8192", "--jinja", "--metrics"],
                                     env=env, stdout=log, stderr=subprocess.STDOUT)
        if not wait_ok(self.url + "/health", 600):
            sys.exit(open(log.name).read()[-3000:])
        print(f"{self.name} ready (-ngl {ngl})", flush=True)

    def stop(self):
        if self.proc:
            self.proc.terminate()
            self.proc.wait(timeout=60)
            self.proc = None


def measure_speed(node):
    """Decode and prompt speed of one node, one request at a time (median of 3 after a warm-up)."""
    body = {"model": "qwen3-8b", "max_tokens": 128, "temperature": 0, "stream": False,
            "messages": [{"role": "system", "content": "/no_think"},
                         {"role": "user", "content": f"Read this document and summarize it.\n\n{LONG_DOC}"}]}
    post_json(node.url + "/v1/chat/completions", body)                 # warm-up
    dec, pre = [], []
    for k in range(3):
        body["messages"][1]["content"] = f"[{k}] Read this document and summarize it.\n\n{LONG_DOC}"   # no cache reuse
        t = post_json(node.url + "/v1/chat/completions", body).get("timings", {})
        dec.append(t.get("predicted_per_second", 0.0))
        pre.append(t.get("prompt_per_second", 0.0))
    return {"speed_tps": round(statistics.median(dec), 2), "prefill_tps": round(statistics.median(pre), 1)}


def workload():
    """The same request list for every strategy: Poisson arrivals, 1 in 3 is a long-document (RAG-like) request."""
    rng = random.Random(SEED)
    t, items = 0.0, []
    for i in range(N_REQ):
        t += rng.expovariate(RATE)
        rag = rng.random() < 1 / 3
        content = (f"[{i}] Read this document and answer: what is it about?\n\n{LONG_DOC}" if rag
                   else f"[{i}] {rng.choice(QUESTIONS)}")
        items.append({"at": t, "type": "rag" if rag else "chat", "max_tokens": 128 if rag else rng.choice([64, 128, 256]),
                      "content": content})
    return items


def send_all(items):
    import httpx

    async def one(c, it, t0):
        await asyncio.sleep(max(0.0, t0 + it["at"] - time.time()))
        body = {"model": "qwen3-8b", "stream": True, "max_tokens": it["max_tokens"], "temperature": 0.7,
                "messages": [{"role": "system", "content": "/no_think"}, {"role": "user", "content": it["content"]}]}
        try:
            async with c.stream("POST", ROUTER, json=body,
                                headers={"x-seed": str(SEED), "x-request-type": it["type"]}) as r:
                async for _ in r.aiter_lines():
                    pass
        except Exception as e:
            print("request failed:", e, flush=True)

    async def run():
        t0 = time.time()
        async with httpx.AsyncClient(timeout=None) as c:
            await asyncio.gather(*[one(c, it, t0) for it in items])

    th = threading.Thread(target=lambda: asyncio.run(run()))       # Jupyter already runs an event loop
    th.start()
    th.join()


def run_strategy(scenario, strategy, nodes, speeds, items):
    import yaml
    res_dir = OUT / scenario
    res_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"router": {"host": "127.0.0.1", "port": 8000, "strategy": strategy, "results_dir": str(res_dir),
                      "run_id": f"proto_{scenario}"},
           "model": {"name": "Qwen3-8B", "gguf_quant": "Q4_K_M", "max_context": 8192, "thinking": False},
           "energy": {"method": "tokens"},
           "nodes": [{"name": n.name, "url": n.url, "kind": "llamacpp", "gpu_index": n.index,
                      "ngl": n.ngl, **speeds[n.name]} for n in nodes]}
    cfg_path = res_dir / f"config_{strategy}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    before = set(res_dir.glob(f"proto_{scenario}_{strategy}_*.csv"))
    rlog = open(res_dir / f"router_{strategy}.log", "w")
    router = subprocess.Popen([sys.executable, str(ROOT / "router" / "router.py"), "--config", str(cfg_path)],
                              cwd=str(ROOT), stdout=rlog, stderr=subprocess.STDOUT)
    time.sleep(4)
    t0 = time.time()
    send_all(items)
    print(f"{scenario}/{strategy}: {len(items)} requests in {time.time() - t0:.0f} s", flush=True)
    router.terminate()
    router.wait(timeout=30)
    new = sorted(set(res_dir.glob(f"proto_{scenario}_{strategy}_*.csv")) - before)
    time.sleep(10)                                   # idle gap between runs, visible in the power log
    return new[0] if new else None, cfg_path


def attribute(csv_path, cfg_path):
    """energy_j with the 'tokens' rule; the 'equal' rule only into a side summary."""
    out = {}
    for method in ("equal", "tokens"):              # tokens last, so the file keeps the tokens values
        summ = csv_path.with_name(csv_path.stem + f"_energy_{method}.json")
        subprocess.run([sys.executable, str(ROOT / "infra" / "attribute_energy.py"), "--requests", str(csv_path),
                        "--power", str(OUT / "power.csv"), "--config", str(cfg_path), "--method", method,
                        "--summary", str(summ)], check=True, stdout=subprocess.DEVNULL)
        out[method] = json.load(open(summ))
    return out


def summarize(csv_path, energy):
    import pandas as pd
    d = pd.read_csv(csv_path)
    row = {"requests": len(d), "ttft_mean_s": d.ttft_ms.mean() / 1000, "ttft_p95_s": d.ttft_ms.quantile(.95) / 1000,
           "e2e_mean_s": d.e2e_ms.mean() / 1000, "e2e_p95_s": d.e2e_ms.quantile(.95) / 1000,
           "tpot_mean_ms": d.tpot_ms.mean(), "share": d.gpu.value_counts(normalize=True).round(2).to_dict()}
    row["j_per_token"] = {n: v["j_per_token"] for n, v in energy["tokens"]["nodes"].items()}
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    step("1/5 GPUs and runtime")
    print(subprocess.run("nvidia-smi --query-gpu=index,name,memory.total,power.limit --format=csv",
                         shell=True, capture_output=True, text=True).stdout)
    exe, model = ensure_runtime()
    print("llama-server:", exe, "\nmodel:", model)
    sh(f"{sys.executable} -m pip install -q fastapi uvicorn httpx pyyaml pandas")

    step("2/5 Energy logger")
    plog = subprocess.Popen([sys.executable, str(ROOT / "infra" / "energy_logger.py"), "--out", str(OUT / "power.csv")])
    time.sleep(3)

    nodes = [Node(0, exe, model), Node(1, exe, model)]
    items = workload()
    json.dump(items, open(OUT / "workload.json", "w"), indent=1)
    report = {"settings": {"requests": N_REQ, "rate": RATE, "slow_ngl": SLOW_NGL, "seed": SEED}, "scenarios": {}}

    for scenario in SCENARIOS:
        step(f"3/5 Scenario '{scenario}': start nodes and measure speeds")
        nodes[0].start("99")
        nodes[1].start("99" if scenario == "same" else SLOW_NGL)
        speeds = {n.name: measure_speed(n) for n in nodes}
        print(json.dumps(speeds, indent=2), flush=True)
        for n in nodes:                              # one sample of what the router reads, to check the parsing
            for ep in ("metrics", "slots"):
                try:
                    with urllib.request.urlopen(f"{n.url}/{ep}", timeout=3) as r:
                        (OUT / f"{scenario}_{n.name}_{ep}.txt").write_bytes(r.read())
                except Exception as e:
                    (OUT / f"{scenario}_{n.name}_{ep}.txt").write_text(f"not available: {e}")
        report["scenarios"][scenario] = {"speeds": speeds, "runs": {}}

        step(f"4/5 Scenario '{scenario}': strategies")
        for strategy in STRATEGIES:
            csv_path, cfg_path = run_strategy(scenario, strategy, nodes, speeds, items)
            if not csv_path:
                print("no result file for", strategy, flush=True)
                continue
            energy = attribute(csv_path, cfg_path)
            report["scenarios"][scenario]["runs"][strategy] = summarize(csv_path, energy)
            print(json.dumps(report["scenarios"][scenario]["runs"][strategy], indent=1, default=float), flush=True)
            json.dump(report, open(OUT / "summary.json", "w"), indent=2, default=float)
    for n in nodes:
        n.stop()
    plog.terminate()

    step("5/5 Summary")
    for scenario, s in report["scenarios"].items():
        print(f"\n[{scenario}] speeds: {s['speeds']}")
        print(f"{'strategy':12} {'TTFT mean':>10} {'TTFT p95':>9} {'E2E mean':>9} {'E2E p95':>8}  share / J per token")
        for strategy, r in s["runs"].items():
            print(f"{strategy:12} {r['ttft_mean_s']:9.2f}s {r['ttft_p95_s']:8.2f}s {r['e2e_mean_s']:8.2f}s "
                  f"{r['e2e_p95_s']:7.2f}s  {r['share']} {r['j_per_token']}")
    json.dump(report, open(OUT / "summary.json", "w"), indent=2, default=float)
    shutil.make_archive(str(WORK / "prototype_results"), "zip", OUT)
    print("\nAll files:", OUT, "and", WORK / "prototype_results.zip")


if __name__ == "__main__":
    main()
