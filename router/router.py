"""OpenAI-compatible router.

For every request it reads each node's state, asks the strategy which node
to use, forwards the request, and writes one CSV row with the measurements.

Run:  python router/router.py --config config.yaml
"""
import argparse, asyncio, csv, importlib, json, os, sys, time, uuid
from pathlib import Path

import httpx, uvicorn, yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strategies.base import Request as LbRequest, GpuState

CSV_COLUMNS = ["run_id", "strategy", "seed", "request_id", "gpu", "prompt_len", "output_len",
               "arrival_ts", "ttft_ms", "tpot_ms", "e2e_ms", "energy_j"]

app = FastAPI()
CFG, NODES, STRATEGY, WRITER, CSV_FILE = {}, {}, None, None, None
client = httpx.AsyncClient(timeout=None)


def prompt_tokens(body):
    text = " ".join(m.get("content", "") for m in body.get("messages", []) if isinstance(m.get("content"), str))
    return max(1, int(len(text.split()) * 1.3))


SPEED_KEYS = ("speed_tps", "prefill_tps")   # measured once per GPU type, set per node in config.yaml


UNREACHABLE = 10**6                         # queue_len given to a node that does not answer
INFLIGHT = {}                               # node -> requests sent by this router and not finished
SLOTS = {}                                  # node -> number of llama.cpp slots (read once)


def parse_metrics(text):
    """Prometheus text from llama-server /metrics -> {name: value}."""
    out = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            parts = line.split()
            try:
                out[parts[0].split("{")[0]] = float(parts[-1])
            except (ValueError, IndexError):
                pass
    return out


async def llamacpp_state(name, node, extra):
    """State of a llama.cpp server (needs --metrics). Running and waiting requests come from
    /metrics; KV-cache use from kv_cache_usage_ratio when the build reports it, otherwise
    the share of busy slots."""
    m = parse_metrics((await client.get(node["url"] + "/metrics", timeout=1.0)).text)
    running = int(m.get("llamacpp:requests_processing", 0))
    waiting = int(m.get("llamacpp:requests_deferred", 0))
    kv = m.get("llamacpp:kv_cache_usage_ratio")
    if kv is None:
        if name not in SLOTS:
            try:
                SLOTS[name] = max(1, len((await client.get(node["url"] + "/slots", timeout=1.0)).json()))
            except Exception:
                SLOTS[name] = 1
        kv = min(1.0, running / SLOTS[name])
    extra["inflight"] = INFLIGHT.get(name, 0)
    return GpuState(queue_len=waiting, running=running, kv_usage=float(kv), extra=extra)


async def read_states():
    async def one(name, node):
        # Fixed node speeds from config.yaml, so strategies need not read the file themselves
        extra = {k: float(node[k]) for k in SPEED_KEYS if node.get(k) is not None}
        if node["kind"] == "llamacpp":
            try:
                return name, await llamacpp_state(name, node, extra)
            except Exception:
                return name, GpuState(queue_len=UNREACHABLE, extra=extra)
        if node["kind"] != "mock":
            return name, GpuState(extra=extra)      # vLLM state: added when vLLM runs are prepared
        try:
            s = (await client.get(node["url"] + "/state", timeout=1.0)).json()
            for k in SPEED_KEYS:                    # a value reported by the node wins over config.yaml
                if s.get(k) is not None:
                    extra[k] = float(s[k])
            extra["inflight"] = INFLIGHT.get(name, 0)
            return name, GpuState(queue_len=s["queue_len"], running=s["running"], kv_usage=s["kv_usage"],
                                  extra=extra)
        except Exception:
            return name, GpuState(queue_len=UNREACHABLE, extra=extra)
    return dict(await asyncio.gather(*[one(n, v) for n, v in NODES.items()]))


def log_row(row):
    WRITER.writerow(row)
    CSV_FILE.flush()


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    arrival = time.time()
    lb_req = LbRequest(request_id=uuid.uuid4().hex[:12], prompt_len=prompt_tokens(body),
                       req_type=req.headers.get("x-request-type", "chat"), arrival_ts=arrival)
    states = await read_states()
    gpu = STRATEGY.choose_gpu(lb_req, states)
    url = NODES[gpu]["url"] + "/v1/chat/completions"
    seed = req.headers.get("x-seed", "")

    def row(ttft, n_out, end):
        e2e = (end - arrival) * 1000
        tpot = (e2e - ttft) / max(1, n_out - 1) if n_out > 1 else 0.0
        return {"run_id": CFG["router"]["run_id"], "strategy": CFG["router"]["strategy"], "seed": seed,
                "request_id": lb_req.request_id, "gpu": gpu, "prompt_len": lb_req.prompt_len,
                "output_len": n_out, "arrival_ts": f"{arrival:.6f}", "ttft_ms": f"{ttft:.2f}",
                "tpot_ms": f"{tpot:.2f}", "e2e_ms": f"{e2e:.2f}", "energy_j": ""}

    INFLIGHT[gpu] = INFLIGHT.get(gpu, 0) + 1
    if not body.get("stream", False):
        try:
            r = await client.post(url, json=body)
        finally:
            INFLIGHT[gpu] -= 1
        end = time.time()
        data = r.json()
        n_out = data.get("usage", {}).get("completion_tokens", 0)
        log_row(row((end - arrival) * 1000, n_out, end))   # without streaming, TTFT = full time
        return JSONResponse(data, status_code=r.status_code)

    async def relay():
        # TTFT is taken at the first chunk that carries generated text; the token count comes
        # from the server's usage report when it sends one, otherwise from the text chunks.
        ttft, chunks, usage_tokens = None, 0, None
        try:
            async with client.stream("POST", url, json=body) as r:
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            msg = json.loads(line[6:])
                        except ValueError:
                            msg = {}
                        delta = (msg.get("choices") or [{}])[0].get("delta") or {}
                        if delta.get("content") or delta.get("reasoning_content"):
                            if ttft is None:
                                ttft = (time.time() - arrival) * 1000
                            chunks += 1
                        if (msg.get("usage") or {}).get("completion_tokens") is not None:
                            usage_tokens = msg["usage"]["completion_tokens"]
                    yield line + "\n\n"
        finally:
            INFLIGHT[gpu] -= 1
        log_row(row(ttft or 0.0, usage_tokens if usage_tokens is not None else chunks, time.time()))
    return StreamingResponse(relay(), media_type="text/event-stream")


def main():
    global CFG, NODES, STRATEGY, WRITER, CSV_FILE
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    CFG = yaml.safe_load(open(a.config))
    NODES = {n["name"]: n for n in CFG["nodes"]}
    STRATEGY = importlib.import_module("strategies." + CFG["router"]["strategy"])
    out = Path(CFG["router"]["results_dir"]); out.mkdir(exist_ok=True)
    path = out / f"{CFG['router']['run_id']}_{CFG['router']['strategy']}_{int(time.time())}.csv"
    CSV_FILE = open(path, "w", newline="")
    WRITER = csv.DictWriter(CSV_FILE, fieldnames=CSV_COLUMNS); WRITER.writeheader()
    print(f"router: strategy={CFG['router']['strategy']} nodes={list(NODES)} results={path}")
    uvicorn.run(app, host=CFG["router"]["host"], port=CFG["router"]["port"], log_level="warning")


if __name__ == "__main__":
    main()
