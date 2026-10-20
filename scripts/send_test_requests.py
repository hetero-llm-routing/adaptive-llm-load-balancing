"""Send a few streaming requests to the router to check that everything works."""
import asyncio, httpx, random

async def one(client, i):
    body = {"model": "test", "stream": True, "max_tokens": random.randint(10, 60),
            "messages": [{"role": "user", "content": "hello " * random.randint(5, 200)}]}
    async with client.stream("POST", "http://127.0.0.1:8000/v1/chat/completions", json=body,
                             headers={"x-seed": "1", "x-request-type": random.choice(["chat", "rag"])}) as r:
        async for _ in r.aiter_lines():
            pass
    print("request", i, "done")

async def main(n=12):
    async with httpx.AsyncClient(timeout=None) as c:
        await asyncio.gather(*[one(c, i) for i in range(n)])

asyncio.run(main())
