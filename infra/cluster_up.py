"""Check the RunPod nodes and write the run configuration (on the router machine).

    python infra/cluster_up.py --node rtx4090=https://<pod>-8080.proxy.runpod.net \
        --node rtxa6000=https://<pod>-8080.proxy.runpod.net \
        --node rtx2000ada=https://<pod>-8080.proxy.runpod.net \
        --profiles results/profiles --out config_run.yaml [--wait 30]

For every node it checks llama-server (/health, /props: slots and context), the node agent
on port 8081 (same URL with -8081 instead of -8080: /health and /time) and the clock offset
between the node and this machine. It then writes a config file in the format of config.yaml
with the measured speeds of each GPU type from results/profiles/<gpu>.json (speed_tps,
prefill_tps and speed_by_n for 1 to 4 running requests), state_interval_s = 0.25, and the
agent URL and clock offset of each node for the runner. Finally it waits --wait seconds
(the GPUs leave the high-power state after loading the model).
"""
import argparse, json, sys, time, urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROFILE_FOR = {"rtx4090": "geforce_rtx_4090", "rtxa6000": "rtx_a6000", "rtx2000ada": "rtx_2000_ada_generation",
               "rtx4000ada": "rtx_4000_ada_generation", "t4": "t4", "p100": "p100-pcie-16gb"}


def get(url, timeout=10):
    # RunPod proxy (Cloudflare) refuses the default Python-urllib user agent
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "lb-runner/1.0"}), timeout=timeout) as r:
        body = r.read()
    try:
        return json.loads(body)
    except ValueError:
        return body.decode()


def agent_url(url):
    return url.replace("-8080.", "-8081.") if "-8080." in url else url.rsplit(":", 1)[0] + ":8081"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", action="append", required=True, help="name=url, repeated")
    ap.add_argument("--profiles", default=str(ROOT / "results" / "profiles"))
    ap.add_argument("--base", default=str(ROOT / "config_testbed.yaml"))
    ap.add_argument("--out", default=str(ROOT / "config_run.yaml"))
    ap.add_argument("--wait", type=float, default=30.0)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.base))
    cfg["router"]["state_interval_s"] = 0.25
    nodes, ok = [], True
    for spec in a.node:
        name, url = spec.split("=", 1)
        url = url.rstrip("/")
        n = {"name": name, "url": url, "kind": "llamacpp", "gpu_index": 0, "agent": agent_url(url)}
        try:
            get(url + "/health")
            props = get(url + "/props")
            n["slots"] = props.get("total_slots")
            n["n_ctx"] = (props.get("default_generation_settings") or {}).get("n_ctx")
        except Exception as e:
            print(f"{name}: llama-server not reachable ({e})"); ok = False
        try:
            t0 = time.time(); remote = get(n["agent"] + "/time"); t1 = time.time()
            n["host"] = remote["host"]
            n["clock_offset_s"] = round(remote["time"] - (t0 + t1) / 2, 3)    # node clock minus router clock
        except Exception as e:
            print(f"{name}: node agent not reachable ({e})"); ok = False
        prof = Path(a.profiles) / f"{PROFILE_FOR.get(name, name)}.json"
        if prof.exists():
            p = json.loads(prof.read_text())
            n["speed_tps"] = p["speed_tps"]
            n["prefill_tps"] = p["prefill_tps"]
            n["speed_by_n"] = [p["decode_by_concurrency"][str(k)]["total_tps"] for k in (1, 2, 3, 4)]
            n["power_by_n"] = [p["power"][f"decode_{k}"]["mean_w"] for k in (1, 2, 3, 4)]
            n["idle_w"] = p["power"]["idle"]["mean_w"]
        else:
            print(f"{name}: no profile at {prof}; speeds not set"); ok = False
        if n.get("slots") not in (None, 4):
            print(f"{name}: {n['slots']} slots, expected 4"); ok = False
        nodes.append(n)
        print(name, {k: n.get(k) for k in ("slots", "n_ctx", "host", "clock_offset_s", "speed_tps")})
    cfg["nodes"] = nodes
    Path(a.out).write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("config written:", a.out)
    if not ok:
        sys.exit("some checks failed; see above")
    time.sleep(a.wait)
    print("cluster ready")


if __name__ == "__main__":
    main()
