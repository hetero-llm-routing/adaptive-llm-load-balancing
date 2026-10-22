"""Smoke test on a free Kaggle GPU notebook (task #23).

Builds llama.cpp with CUDA, downloads Qwen3-8B (GGUF Q4_K_M), starts one
llama.cpp server per GPU, starts our router with round-robin, sends test
requests and prints the CSV summary.

Kaggle settings: Accelerator = GPU T4 x2, Internet = On. Run it with Save Version >
Save & Run All, so it keeps running when the browser closes. The folder
/kaggle/working/runtime (llama-server + model) can then be saved as a dataset and
added as Input, so later runs skip the build and the download.
Run from the repository root:  python scripts/kaggle_smoke_test.py
"""
import glob, os, subprocess, sys, time, json, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("LB_WORK", "/kaggle/working"))
LLAMA = WORK / "llama.cpp"
RUNTIME = WORK / "runtime"            # kept in the notebook output; save it as a Kaggle dataset
SERVER = RUNTIME / "llama-server"
N_REQUESTS = int(os.environ.get("LB_REQUESTS", "24"))


def sh(cmd, **kw):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)


def step(title):
    print(f"\n========== {title} ==========", flush=True)


def wait_ok(url, seconds):
    end = time.time() + seconds
    while time.time() < end:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def gpus():
    out = subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
                         shell=True, capture_output=True, text=True).stdout.strip().splitlines()
    return [l.strip() for l in out if l.strip()]


def main():
    step("1/6 GPUs")
    found = gpus()
    if not found:
        sys.exit("No GPU found. In the notebook settings choose Accelerator = GPU T4 x2.")
    for i, g in enumerate(found):
        print(f"GPU {i}: {g}")

    step("2/6 llama.cpp server (built once, then reused from a Kaggle dataset)")
    pre = sorted(glob.glob("/kaggle/input/**/llama-server", recursive=True))
    if pre and not SERVER.exists():
        RUNTIME.mkdir(exist_ok=True)
        sh(f"cp {pre[0]} {SERVER} && chmod +x {SERVER}")
        print("reused from dataset:", pre[0])
    if not SERVER.exists():
        print("building with CUDA for the T4 (15 to 25 minutes, only the first time)")
        if not LLAMA.exists():
            sh(f"git clone --depth 1 https://github.com/ggml-org/llama.cpp {LLAMA}")
        # CUDA::cuda_driver needs a file named libcuda.so at configure time; Kaggle images
        # may only have libcuda.so.1, so link one into a local folder
        libs = sorted(glob.glob("/usr/**/libcuda.so*", recursive=True))
        libs = [p for p in libs if "stubs" in p] or libs
        stub = WORK / "cudalib"; stub.mkdir(exist_ok=True)
        if libs and not (stub / "libcuda.so").exists():
            os.symlink(libs[0], stub / "libcuda.so")
        print("libcuda for linking:", libs[:1])
        sh(f"rm -rf {LLAMA}/build && cmake -S {LLAMA} -B {LLAMA}/build -DGGML_CUDA=ON "
           f"-DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release "
           f"-DBUILD_SHARED_LIBS=OFF "
           f"-DCMAKE_LIBRARY_PATH={stub} > {WORK}/cmake.log 2>&1")
        sh(f"cmake --build {LLAMA}/build --target llama-server -j {os.cpu_count()} > {WORK}/build.log 2>&1")
        RUNTIME.mkdir(exist_ok=True)
        sh(f"cp {LLAMA}/build/bin/llama-server {SERVER} && rm -rf {LLAMA}")
    print("llama-server ready:", SERVER)

    step("3/6 Download Qwen3-8B Q4_K_M (about 5 GB)")
    found_models = sorted(glob.glob("/kaggle/input/**/*Q4_K_M*.gguf", recursive=True))
    if found_models:
        model = found_models[0]
        print("reused from dataset")
    else:
        sh(f"{sys.executable} -m pip install -q huggingface_hub")
        from huggingface_hub import snapshot_download
        d = snapshot_download("Qwen/Qwen3-8B-GGUF", allow_patterns=["*Q4_K_M*.gguf"],
                              local_dir=str(RUNTIME))
        model = sorted(glob.glob(f"{d}/**/*Q4_K_M*.gguf", recursive=True))[0]
    print("model:", model, f"({os.path.getsize(model) / 1e9:.2f} GB)")

    step("4/6 Start one llama.cpp server per GPU")
    nodes = []
    for i, g in enumerate(found):
        port = 8081 + i
        log = open(WORK / f"server_{i}.log", "w")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
        subprocess.Popen([str(SERVER), "-m", model, "--host", "127.0.0.1", "--port", str(port),
                          "-ngl", "99", "-c", "8192", "--jinja", "--metrics"],
                         env=env, stdout=log, stderr=subprocess.STDOUT)
        name = g.split(",")[0].replace("Tesla ", "").replace(" ", "")
        nodes.append({"name": f"{name.lower()}_{i}", "url": f"http://127.0.0.1:{port}", "kind": "llamacpp"})
    for n in nodes:
        ok = wait_ok(n["url"] + "/health", 300)
        print(n["name"], "ready" if ok else "NOT READY, see server log")
        if not ok:
            sys.exit(open(WORK / f"server_{nodes.index(n)}.log").read()[-3000:])

    step("5/6 Start the router (round-robin)")
    import yaml
    cfg = {"router": {"host": "127.0.0.1", "port": 8000, "strategy": "round_robin",
                      "results_dir": str(WORK / "results"), "run_id": "kaggle_smoke"},
           "model": {"name": "Qwen3-8B", "gguf_quant": "Q4_K_M", "max_context": 8192, "thinking": False},
           "nodes": nodes}
    cfg_path = WORK / "config_kaggle.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    sh(f"{sys.executable} -m pip install -q fastapi uvicorn httpx pyyaml")
    rlog = open(WORK / "router.log", "w")
    subprocess.Popen([sys.executable, str(ROOT / "router" / "router.py"), "--config", str(cfg_path)],
                     cwd=str(ROOT), stdout=rlog, stderr=subprocess.STDOUT)
    time.sleep(5)

    step(f"6/6 Send {N_REQUESTS} test requests")
    import asyncio, random, httpx
    random.seed(1)
    questions = ["Explain what a GPU is in two sentences.",
                 "Summarize the idea of load balancing for a new student.",
                 "Write a short formal email asking for a meeting next week.",
                 "What is the difference between a process and a thread?",
                 "Give three tips for writing a clear research abstract."]
    long_doc = " ".join(["Distributed systems share work among many machines to finish jobs faster."] * 120)

    async def one(client, i):
        rag = i % 3 == 0
        content = (f"Read this document and answer: what is it about?\n\n{long_doc}" if rag
                   else random.choice(questions))
        body = {"model": "qwen3-8b", "stream": True, "max_tokens": 128, "temperature": 0.7,
                "messages": [{"role": "system", "content": "/no_think"},
                             {"role": "user", "content": content}]}
        async with client.stream("POST", "http://127.0.0.1:8000/v1/chat/completions", json=body,
                                 headers={"x-seed": "1", "x-request-type": "rag" if rag else "chat"}) as r:
            async for _ in r.aiter_lines():
                pass

    async def run():
        async with httpx.AsyncClient(timeout=None) as c:
            await asyncio.gather(*[one(c, i) for i in range(N_REQUESTS)])

    t0 = time.time()
    # Jupyter already runs an event loop, so run ours in a separate thread
    import threading
    th = threading.Thread(target=lambda: asyncio.run(run())); th.start(); th.join()
    print(f"done in {time.time() - t0:.1f} s")
    time.sleep(2)

    import pandas as pd
    csv = sorted(glob.glob(str(WORK / "results" / "*.csv")))[-1]
    df = pd.read_csv(csv)
    print("\nCSV:", csv, "rows:", len(df))
    print(df.groupby("gpu")[["prompt_len", "output_len", "ttft_ms", "tpot_ms", "e2e_ms"]]
          .mean().round(1).to_string())
    print("\nThe CSV file is in the Output panel (/kaggle/working/results).")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        for f in ["cmake.log", "build.log", "server_0.log", "router.log"]:
            if (WORK / f).exists():
                print(f"\n----- last lines of {f} -----")
                print("\n".join((WORK / f).read_text(errors="replace").splitlines()[-30:]))
        raise
