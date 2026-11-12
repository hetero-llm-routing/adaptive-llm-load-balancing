"""Fast offline simulator of the mock cluster (Osama).

It reproduces the timing rules of infra/mock_server.py without any server or
waiting, so a strategy can be tested on thousands of requests in seconds, and
the RL strategy can be trained on it:

  * a node runs at most `max_running` requests; the rest wait in a FIFO queue,
  * a running request first spends prompt_len / prefill_tps seconds in prefill,
  * every running request then generates tokens at speed_tps / running
    tokens per second (the node's speed is shared by everything it runs),
  * TTFT = arrival -> first generated token; E2E = arrival -> last token.

Strategies are called exactly as the router calls them:
choose_gpu(Request, {node: GpuState}).  GpuState.extra carries speed_tps and
prefill_tps, as config.yaml would give them.

Compare strategies (numbers are for the mock model only, not for the paper):
    python -m strategies.mock_sim --strategies round_robin predictive --rate 0.5 --seeds 1 2 3
"""
import argparse, csv, importlib, random, statistics
from dataclasses import dataclass, field

from strategies.base import Request, GpuState

MOCK_NODES = {"fast": 60.0, "medium": 30.0, "slow": 10.0}    # as scripts/run_mock_cluster.sh


@dataclass
class Job:
    req: Request
    out_len: int
    node: str = ""
    start: float = 0.0          # left the queue
    prefill_end: float = 0.0
    done_tokens: float = 0.0
    ttft: float = None
    end: float = None


@dataclass
class Node:
    name: str
    speed_tps: float
    prefill_tps: float = 600.0
    max_running: int = 4
    speed_by_n: list = None     # total decode speed with 1..4 running requests (A5); None: speed_tps shared
    queue: list = field(default_factory=list)
    running: list = field(default_factory=list)

    def state(self):
        return GpuState(queue_len=len(self.queue), running=len(self.running),
                        kv_usage=len(self.running) / self.max_running,
                        extra={"speed_tps": self.speed_tps, "prefill_tps": self.prefill_tps,
                               **({"speed_by_n": self.speed_by_n} if self.speed_by_n else {})})

    def rate(self):
        """Per-request decode speed with the current number of running requests (A5)."""
        k = len(self.running)
        if not k:
            return 0.0
        total = self.speed_by_n[min(k, len(self.speed_by_n)) - 1] if self.speed_by_n else self.speed_tps
        return total / k


def workload(n, rate, seed, rag_share=0.3):
    """Poisson arrivals; chat = short prompt, rag = long prompt with retrieved text.
    The true output length is hidden from the strategy, as in a real run."""
    rnd, t, jobs = random.Random(seed), 1_000.0, []
    for i in range(n):
        t += rnd.expovariate(rate)
        if rnd.random() < rag_share:
            kind, prompt, out = "rag", rnd.randint(800, 2500), rnd.randint(60, 220)
        else:
            kind, prompt, out = "chat", rnd.randint(8, 200), rnd.randint(20, 300)
        jobs.append(Job(Request(f"r{i}", prompt, kind, t), out))
    return jobs


def simulate(strategy, jobs, nodes):
    """Event-driven run. Returns the jobs with node, ttft and end filled in."""
    nodes = {n.name: n for n in nodes}
    pending, now, eps = sorted(jobs, key=lambda j: j.req.arrival_ts), 0.0, 1e-9
    i = 0
    if hasattr(strategy, "reset"):
        strategy.reset()

    def admit(node, t):
        while node.queue and len(node.running) < node.max_running:
            j = node.queue.pop(0)
            j.start, j.prefill_end = t, t + j.req.prompt_len / node.prefill_tps
            node.running.append(j)

    while i < len(pending) or any(n.running or n.queue for n in nodes.values()):
        # next event: an arrival, a prefill end, a first token or a request end
        nxt = pending[i].req.arrival_ts if i < len(pending) else float("inf")
        for n in nodes.values():
            rate = n.rate()
            for j in n.running:
                if j.prefill_end > now + eps:
                    nxt = min(nxt, j.prefill_end)
                else:
                    goal = 1 if j.ttft is None else j.out_len
                    nxt = min(nxt, now + (goal - j.done_tokens) / rate)
        dt = nxt - now
        # advance every decoding request
        for n in nodes.values():
            if n.running:
                rate = n.rate()
                for j in n.running:
                    if j.prefill_end <= now + eps:
                        j.done_tokens += rate * dt
        now = nxt
        for n in nodes.values():
            for j in list(n.running):
                if j.ttft is None and j.done_tokens >= 1 - 1e-6:
                    j.ttft = now - j.req.arrival_ts
                if j.done_tokens >= j.out_len - 1e-6:
                    j.end = now
                    n.running.remove(j)
            admit(n, now)
        # arrivals at this moment
        while i < len(pending) and pending[i].req.arrival_ts <= now + eps:
            j = pending[i]; i += 1
            states = {name: n.state() for name, n in nodes.items()}
            j.node = strategy.choose_gpu(j.req, states)
            nodes[j.node].queue.append(j)
            admit(nodes[j.node], now)
    return jobs


def summary(jobs):
    e2e = sorted(j.end - j.req.arrival_ts for j in jobs)
    ttft = sorted(j.ttft for j in jobs)
    tpot = [(j.end - j.req.arrival_ts - j.ttft) / max(1, j.out_len - 1) for j in jobs]
    share = {}
    for j in jobs:
        share[j.node] = share.get(j.node, 0) + 1
    p = lambda xs, q: xs[min(len(xs) - 1, int(q * len(xs)))]
    return {"ttft_mean": statistics.mean(ttft), "ttft_p95": p(ttft, .95),
            "tpot_mean": statistics.mean(tpot),
            "e2e_mean": statistics.mean(e2e), "e2e_p95": p(e2e, .95),
            "share": {k: share[k] / len(jobs) for k in sorted(share)}}


def to_csv(jobs, path, strategy_name, seed):
    """Same columns as the router, so the same analysis scripts work."""
    cols = ["run_id", "strategy", "seed", "request_id", "gpu", "prompt_len", "output_len",
            "arrival_ts", "ttft_ms", "tpot_ms", "e2e_ms", "energy_j", "req_type"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for j in jobs:
            e2e, ttft = (j.end - j.req.arrival_ts) * 1000, j.ttft * 1000
            w.writerow({"run_id": "mock_sim", "strategy": strategy_name, "seed": seed,
                        "request_id": j.req.request_id, "gpu": j.node, "prompt_len": j.req.prompt_len,
                        "output_len": j.out_len, "arrival_ts": f"{j.req.arrival_ts:.6f}",
                        "ttft_ms": f"{ttft:.2f}", "tpot_ms": f"{(e2e - ttft) / max(1, j.out_len - 1):.2f}",
                        "e2e_ms": f"{e2e:.2f}", "energy_j": "", "req_type": j.req.req_type})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", nargs="+", default=["round_robin", "predictive"])
    ap.add_argument("--rate", type=float, default=0.5, help="requests per second")
    ap.add_argument("-n", type=int, default=2000, help="requests per run")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--csv-dir", help="also write one CSV per run here")
    a = ap.parse_args()
    for name in a.strategies:
        mod = importlib.import_module("strategies." + name)
        runs = []
        for s in a.seeds:
            jobs = simulate(mod, workload(a.n, a.rate, s), [Node(k, v) for k, v in MOCK_NODES.items()])
            runs.append(summary(jobs))
            if a.csv_dir:
                to_csv(jobs, f"{a.csv_dir}/mock_sim_{name}_seed{s}.csv", name, s)
        m = lambda k: statistics.mean(r[k] for r in runs)
        print(f"{name:12s} TTFT mean {m('ttft_mean'):6.2f}s p95 {m('ttft_p95'):6.2f}s | "
              f"E2E mean {m('e2e_mean'):6.2f}s p95 {m('e2e_p95'):6.2f}s | share {runs[0]['share']}")


if __name__ == "__main__":
    main()
