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


async def read_states():
    async def one(name, node):
        if node["kind"] != "mock":
            return name, GpuState()      # Eyad: read real llama.cpp / vLLM metrics here
        try:
            s = (await client.get(node["url"] + "/state", timeout=1.0)).json()
            return name, GpuState(queue_len=s["queue_len"], running=s["running"], kv_usage=s["kv_usage"])
        except Exception:
            return name, GpuState(queue_len=10**6)
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

    if not body.get("stream", False):
        r = await client.post(url, json=body)
        end = time.time()
        data = r.json()
        n_out = data.get("usage", {}).get("completion_tokens", 0)
        log_row(row((end - arrival) * 1000, n_out, end))   # without streaming, TTFT = full time
        return JSONResponse(data, status_code=r.status_code)

    async def relay():
        ttft, n_out = None, 0
        async with client.stream("POST", url, json=body) as r:
            async for line in r.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: ") and line != "data: [DONE]":
                    if ttft is None:
                        ttft = (time.time() - arrival) * 1000
                    n_out += 1
                yield line + "\n\n"
        log_row(row(ttft or 0.0, n_out, time.time()))
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
