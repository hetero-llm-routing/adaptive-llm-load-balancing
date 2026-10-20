"""Mock inference server with an OpenAI-compatible API.

It answers /v1/chat/completions like a real LLM server (streaming or not),
but only waits instead of computing, so the whole system can be tested on a
laptop without a GPU. Each node is started with its own speed.

Run:  python infra/mock_server.py --port 9001 --speed 60
"""
import argparse, asyncio, json, random, time, uuid
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

app = FastAPI()
STATE = {"running": 0, "queue_len": 0, "max_running": 4, "speed_tps": 30.0, "prefill_tps": 600.0}


def count_tokens(messages):
    text = " ".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
    return max(1, int(len(text.split()) * 1.3))


@app.get("/state")
async def state():
    """Node state that the router reads before every decision."""
    return {"running": STATE["running"], "queue_len": STATE["queue_len"],
            "kv_usage": STATE["running"] / STATE["max_running"], "speed_tps": STATE["speed_tps"]}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    prompt_len = count_tokens(body.get("messages", []))
    out_len = int(body.get("max_tokens") or random.randint(20, 300))
    stream = bool(body.get("stream", False))

    STATE["queue_len"] += 1
    while STATE["running"] >= STATE["max_running"]:
        await asyncio.sleep(0.01)
    STATE["queue_len"] -= 1
    STATE["running"] += 1
    # requests that run together share the node's speed
    per_token = lambda: STATE["running"] / STATE["speed_tps"]

    async def tokens():
        try:
            await asyncio.sleep(prompt_len / STATE["prefill_tps"])      # prefill
            for i in range(out_len):                                     # decode
                await asyncio.sleep(per_token())
                yield f"tok{i} "
        finally:
            STATE["running"] -= 1

    rid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    if not stream:
        text = "".join([t async for t in tokens()])
        return JSONResponse({"id": rid, "object": "chat.completion", "created": int(time.time()),
                             "model": body.get("model", "mock"),
                             "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                          "finish_reason": "length"}],
                             "usage": {"prompt_tokens": prompt_len, "completion_tokens": out_len,
                                       "total_tokens": prompt_len + out_len}})

    async def sse():
        async for t in tokens():
            chunk = {"id": rid, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": t}}]}
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(sse(), media_type="text/event-stream")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9001)
    ap.add_argument("--speed", type=float, default=30.0, help="decode tokens per second")
    ap.add_argument("--max-running", type=int, default=4)
    a = ap.parse_args()
    STATE["speed_tps"], STATE["max_running"] = a.speed, a.max_running
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
