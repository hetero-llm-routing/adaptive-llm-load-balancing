"""Share the measured energy of each node among the requests it served.

    python infra/attribute_energy.py --requests results/run.csv --power results/power.csv \
        --config config.yaml --method tokens [--summary results/run_energy.json]

Fills the energy_j column of the requests file in place and prints, per node, the
energy over the run, the part with no request on the node (idle), the output tokens
and the energy per output token.

Energy per output token of a node (the main energy metric, Eq. (9) in Section 3.3) is the
node's measured energy over the run divided by its output tokens; the run is the same for all
nodes, from the first arrival to the last completion. It does not depend on the method.

Energy of one request (energy_j) needs a rule, because a GPU serves several requests
at once. Each power sample i covers [t_i, t_i+1), with energy E_i = P_i * (t_i+1 - t_i).
Two rules are implemented; the team chooses one (task #26):

  tokens  E_i is shared in proportion to the output tokens each request generated in the
          interval, assuming tokens are generated at an even rate between its first token
          and its end. An interval in which requests are on the node but none is
          generating (prompt processing, waiting) is shared as in "equal".
  equal   E_i is shared in proportion to the time each request spent on the node in the
          interval (arrival to end), so requests present for the whole interval get
          equal parts.

Energy in intervals with no request on the node is idle energy: reported, not given to
any request. Times come from the requests file: arrival = arrival_ts, first token =
arrival + ttft_ms, end = arrival + e2e_ms.

Nodes are matched to GPUs by gpu_index (and host, if given) in config.yaml.
"""
import argparse, csv, json
from collections import defaultdict

import yaml


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def load_power(path):
    """{(host, gpu_index): [(ts, watts), ...]} sorted by time."""
    out = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            out[(r["host"], int(r["gpu_index"]))].append((float(r["ts"]), float(r["power_w"])))
    for k in out:
        out[k].sort()
    return out


def samples_for(node, power):
    idx = int(node.get("gpu_index", 0))
    host = node.get("host")
    keys = [k for k in power if k[1] == idx and (host is None or k[0] == host)]
    return power[keys[0]] if keys else []


def attribute(reqs, samples, method, max_gap, span=None):
    """Returns (energy per request, total energy over the run, idle energy).

    span = (start, end) of the run, the same for every node (Section 3.3: first arrival to
    last completion of the whole run); default: the first and last request of this node.
    """
    if len(samples) < 2 or (not reqs and span is None):
        return [0.0] * len(reqs), 0.0, 0.0
    span0, span1 = span if span else (min(r["a"] for r in reqs), max(r["e"] for r in reqs))
    energy = [0.0] * len(reqs)
    total = idle = 0.0
    for (t0, p), (t1, _) in zip(samples, samples[1:]):
        t1 = min(t1, t0 + max_gap)                       # do not bridge gaps in the log
        part = overlap(t0, t1, span0, span1)
        if part <= 0:
            continue
        e_int = p * part
        total += e_int
        w0, w1 = max(t0, span0), min(t1, span1)
        present = [overlap(r["a"], r["e"], w0, w1) for r in reqs]
        if sum(present) <= 0:
            idle += e_int
            continue
        weights = present
        if method == "tokens":
            toks = []
            for r in reqs:
                if r["e"] > r["f"]:
                    toks.append(r["n"] * overlap(r["f"], r["e"], w0, w1) / (r["e"] - r["f"]))
                else:
                    toks.append(r["n"] if w0 <= r["e"] < w1 else 0.0)
            if sum(toks) > 0:
                weights = toks
        s = sum(weights)
        for i, w in enumerate(weights):
            energy[i] += e_int * w / s
    return energy, total, idle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", required=True)
    ap.add_argument("--power", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--method", choices=["tokens", "equal"], default=None,
                    help="default: energy.method in config.yaml, else tokens")
    ap.add_argument("--summary", default=None, help="write the per-node summary as JSON here")
    ap.add_argument("--max-gap", type=float, default=5.0, help="seconds; longer gaps in the power log count as no data")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    method = a.method or (cfg.get("energy") or {}).get("method") or "tokens"
    nodes = {n["name"]: n for n in cfg["nodes"]}
    power = load_power(a.power)

    with open(a.requests) as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0].keys()) if rows else []
    by_node = defaultdict(list)
    for i, r in enumerate(rows):
        arr = float(r["arrival_ts"])
        by_node[r["gpu"]].append({"i": i, "a": arr, "f": arr + float(r["ttft_ms"]) / 1000,
                                  "e": arr + float(r["e2e_ms"]) / 1000, "n": float(r["output_len"] or 0)})

    allr = [r for reqs in by_node.values() for r in reqs]
    span = (min(r["a"] for r in allr), max(r["e"] for r in allr)) if allr else None
    summary = {"method": method, "run_s": round(span[1] - span[0], 1) if span else 0, "nodes": {}}
    for name in sorted(set(nodes) | set(by_node)):     # nodes with no request still use energy
        reqs = by_node.get(name, [])
        samples = samples_for(nodes.get(name, {}), power)
        energy, total, idle = attribute(reqs, samples, method, a.max_gap, span)
        for r, e in zip(reqs, energy):
            rows[r["i"]]["energy_j"] = f"{e:.3f}"
        tokens = sum(r["n"] for r in reqs)
        summary["nodes"][name] = {"requests": len(reqs), "power_samples": len(samples),
                                  "energy_j": round(total, 1), "idle_energy_j": round(idle, 1),
                                  "attributed_j": round(sum(energy), 1), "output_tokens": int(tokens),
                                  "j_per_token": round(total / tokens, 4) if tokens else None}

    tot_e = sum(n["energy_j"] for n in summary["nodes"].values())
    tot_n = sum(n["output_tokens"] for n in summary["nodes"].values())
    summary["cluster"] = {"energy_j": round(tot_e, 1), "output_tokens": tot_n,
                          "j_per_token": round(tot_e / tot_n, 4) if tot_n else None}

    with open(a.requests, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(summary, indent=2))
    if a.summary:
        json.dump(summary, open(a.summary, "w"), indent=2)


if __name__ == "__main__":
    main()
