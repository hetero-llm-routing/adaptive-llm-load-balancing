"""One run: start the router with a strategy, replay a request schedule open-loop, collect the
router CSV, the client log and the power samples of every node, and share the energy.
Used by bench/run_matrix.py and bench/calibrate.py (task #50).
"""
import asyncio, csv, json, math, random, socket, subprocess, sys, time, urllib.request
from pathlib import Path

import httpx, yaml

ROOT = Path(__file__).resolve().parents[1]


# ---------------- workloads and arrivals ----------------
def load_workload(path):
    return [json.loads(l) for l in open(path)]


def poisson_times(rate, duration, rng):
    t, out = 0.0, []
    while True:
        t += rng.expovariate(rate)
        if t >= duration:
            return out
        out.append(t)


def trace_times(trace_csv, rate, duration, rng):
    """Bursty arrivals: inter-arrival gaps of the Azure conversation trace [7], from a random
    start point (seeded), scaled by one factor so that the mean rate is `rate`."""
    ts = []
    for line in open(trace_csv).read().splitlines()[1:]:
        try:
            ts.append(_ts(line.split(",")[0]))
        except ValueError:
            pass
    gaps = [b - a for a, b in zip(ts, ts[1:]) if b >= a]
    need = int(rate * duration * 1.5) + 10
    start = rng.randrange(0, max(1, len(gaps) - need))
    window = gaps[start:start + need]
    scale = (1.0 / rate) / (sum(window) / len(window))
    t, out = 0.0, []
    for g in window:
        t += g * scale
        if t >= duration:
            break
        out.append(t)
    return out, scale


def _ts(s):
    from datetime import datetime
    s = s.strip()
    if s.replace(".", "", 1).isdigit():
        return float(s)
    s = s.split("+")[0]
    if "." in s:                                    # 2023-11-16 18:15:46.6805900 -> microseconds
        head, frac = s.split(".")
        s = head + "." + frac[:6]
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").timestamp()
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp()


def schedule(pool, rate, duration, seed, arrival="poisson", trace_csv=None):
    """Same seed -> same (time, request) list for every strategy (paired comparison)."""
    rng = random.Random(seed)
    extra = {}
    if arrival == "poisson":
        times = poisson_times(rate, duration, rng)
    else:
        times, extra["trace_scale"] = trace_times(trace_csv, rate, duration, rng)
    order = rng.sample(range(len(pool)), len(pool))
    reqs = [pool[order[i % len(pool)]] for i in range(len(times))]
    return list(zip(times, reqs)), extra


# ---------------- router process ----------------
def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def start_router(cfg_path, strategy, run_id, out_dir, port, log):
    cfg_path, out_dir = Path(cfg_path).resolve(), Path(out_dir).resolve()
    p = subprocess.Popen([sys.executable, str(ROOT / "router" / "router.py"), "--config", str(cfg_path),
                          "--strategy", strategy, "--run-id", run_id, "--results-dir", str(out_dir),
                          "--port", str(port)], stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    for _ in range(200):
        if p.poll() is not None:
            raise RuntimeError(f"router exited; see {log.name}")
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close(); return p
        except OSError:
            time.sleep(0.1)
    p.kill(); raise RuntimeError("router did not start")


# ---------------- replay ----------------
def body_for(r):
    return {"model": "qwen3-8b", "messages": r["messages"], "max_tokens": int(r["output_len"]),
            "ignore_eos": True, "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}


async def replay(port, sched, seed, timeout_s, drain_s=None):
    client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_s, connect=10.0),
                               limits=httpx.Limits(max_connections=None, max_keepalive_connections=50))
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    log = []
    t0 = time.time()

    async def one(at, r):
        await asyncio.sleep(max(0.0, t0 + at - time.time()))
        sent = time.time(); status, err, chunks = None, "", 0
        try:
            async with client.stream("POST", url, json=body_for(r),
                                     headers={"x-request-type": r["req_type"], "x-seed": str(seed)}) as resp:
                status = resp.status_code
                async for line in resp.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        chunks += 1
        except asyncio.CancelledError:
            err = "cancelled"
        except Exception as e:
            err = type(e).__name__
        log.append({"workload_id": r["request_id"], "req_type": r["req_type"], "scheduled_s": round(at, 4),
                    "sent_ts": f"{sent:.6f}", "lag_ms": round((sent - t0 - at) * 1000, 2),
                    "done_ts": f"{time.time():.6f}", "status": status, "error": err,
                    "prompt_len": r["prompt_len"], "output_len": r["output_len"], "chunks": chunks})

    tasks = [asyncio.create_task(one(at, r)) for at, r in sched]
    limit = None
    if drain_s is not None:                         # stop waiting drain_s after the last arrival
        limit = (sched[-1][0] if sched else 0) + drain_s
    done, pending = await asyncio.wait(tasks, timeout=limit)
    for t in pending:
        t.cancel()
    if pending:
        await asyncio.wait(pending)
    await client.aclose()
    return t0, log


# ---------------- power ----------------
def get(url, timeout=30):
    # RunPod proxy (Cloudflare) refuses the default Python-urllib user agent
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "lb-runner/1.0"}), timeout=timeout) as r:
        return r.read().decode()


def collect_power(cfg, t0, t1, path):
    """Power samples of every node in [t0, t1] (router clock), converted to the router clock with
    the offset measured by infra/cluster_up.py. Returns the number of rows."""
    rows = []
    for n in cfg["nodes"]:
        if not n.get("agent"):
            continue
        off = float(n.get("clock_offset_s", 0.0))
        try:
            text = get(f"{n['agent']}/power?start={t0 + off - 5:.3f}&end={t1 + off + 5:.3f}")
        except Exception as e:
            print(f"power of {n['name']} not fetched: {e}", file=sys.stderr); continue
        for r in list(csv.DictReader(text.splitlines())):
            r["ts"] = f"{float(r['ts']) - off:.3f}"
            rows.append(r)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ts", "host", "gpu_index", "power_w"]); w.writeheader(); w.writerows(rows)
    return len(rows)


# ---------------- idle check ----------------
def node_busy(n):
    """Requests running or waiting on a node (llama.cpp /metrics or the mock /state)."""
    if n.get("kind") == "llamacpp":
        m = {}
        for line in get(n["url"] + "/metrics", 10).splitlines():
            if line and not line.startswith("#"):
                m[line.split()[0].split("{")[0]] = float(line.split()[-1])
        return int(m.get("llamacpp:requests_processing", 0) + m.get("llamacpp:requests_deferred", 0))
    s = json.loads(get(n["url"] + "/state", 10))
    return int(s.get("running", 0) + s.get("queue_len", 0))


def wait_idle(cfg, timeout=900, settle=5.0):
    """Wait until no node runs or queues a request left over from a previous run, then settle
    seconds more, so every run starts from an idle cluster."""
    t_end = time.time() + timeout
    while time.time() < t_end:
        try:
            if all(node_busy(n) == 0 for n in cfg["nodes"]):
                time.sleep(settle); return True
        except Exception as e:
            print("idle check failed:", e, file=sys.stderr)
        time.sleep(2)
    print("cluster not idle after", timeout, "s", file=sys.stderr)
    return False


# ---------------- one run ----------------
def run_once(cfg_path, strategy, run_id, out_dir, sched, seed, timeout_s=600, meta=None, drain_s=None):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(cfg_path))
    wait_idle(cfg)
    port = free_port()
    log = open(out_dir / f"{run_id}_router.log", "w")
    router = start_router(cfg_path, strategy, run_id, out_dir, port, log)
    try:
        t0, client_log = asyncio.run(replay(port, sched, seed, timeout_s, drain_s))
        time.sleep(1.0)
    finally:
        router.terminate()
        try:
            router.wait(10)
        except subprocess.TimeoutExpired:
            router.kill()
        log.close()
    t1 = time.time()
    req_csv = sorted(out_dir.glob(f"{run_id}_{strategy}_*.csv"))[-1]
    with open(out_dir / f"{run_id}_client.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(client_log[0].keys()) if client_log else ["workload_id"])
        w.writeheader(); w.writerows(sorted(client_log, key=lambda r: r["scheduled_s"]))
    n_power = 0
    if any(n.get("agent") for n in cfg["nodes"]):
        power = out_dir / f"{run_id}_power.csv"
        n_power = collect_power(cfg, t0, t1, power)
        if n_power:
            subprocess.run([sys.executable, str(ROOT / "infra" / "attribute_energy.py"), "--requests", str(req_csv.resolve()),
                            "--power", str(power.resolve()), "--config", str(Path(cfg_path).resolve()),
                            "--summary", str((out_dir / f"{run_id}_energy.json").resolve())],
                           check=False, stdout=subprocess.DEVNULL, cwd=ROOT)
    failed = sum(1 for r in client_log if r["error"] or r["status"] != 200)
    m = dict(meta or {}, run_id=run_id, strategy=strategy, seed=seed, start_ts=t0, end_ts=t1,
             n_sent=len(client_log), n_failed=failed, power_rows=n_power, requests_csv=req_csv.name,
             max_lag_ms=max((r["lag_ms"] for r in client_log), default=0))
    (out_dir / f"{run_id}_meta.json").write_text(json.dumps(m, indent=2))
    return m, req_csv, client_log


def summarize(req_csv, warmup_s=0.0, start_ts=None):
    """Quick numbers for calibration and the console (the paper's numbers come from the
    analysis script, task #53)."""
    rows = list(csv.DictReader(open(req_csv)))
    if start_ts is not None:
        rows = [r for r in rows if float(r["arrival_ts"]) >= start_ts + warmup_s]
    if not rows:
        return {}
    q = lambda v, x: sorted(v)[min(len(v) - 1, int(x * len(v)))]
    ttft = [float(r["ttft_ms"]) for r in rows]; e2e = [float(r["e2e_ms"]) for r in rows]
    span = max(float(r["arrival_ts"]) + float(r["e2e_ms"]) / 1000 for r in rows) - min(float(r["arrival_ts"]) for r in rows)
    return {"n": len(rows), "ttft_p50_ms": round(q(ttft, .5)), "ttft_p95_ms": round(q(ttft, .95)),
            "e2e_p50_ms": round(q(e2e, .5)), "e2e_p95_ms": round(q(e2e, .95)),
            "tokens_per_s": round(sum(int(r["output_len"] or 0) for r in rows) / span, 1) if span > 0 else None}
