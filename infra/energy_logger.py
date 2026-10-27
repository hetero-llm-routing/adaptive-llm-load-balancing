"""Energy logger: samples the power draw of every NVIDIA GPU on this machine.

    python infra/energy_logger.py --out results/power.csv [--interval-ms 1000]

Writes one row per GPU per sample:  ts, host, gpu_index, power_w
  ts        Unix seconds when the sample was read (same clock as the router's arrival_ts
            on the same machine; on separate machines the clocks must be synced, e.g. NTP)
  host      machine name, so files from several machines can be combined
  gpu_index index of the GPU in nvidia-smi
  power_w   power draw in watts as reported by nvidia-smi

It runs until stopped (Ctrl+C or SIGTERM). Energy per request is computed after the
run by infra/attribute_energy.py.
"""
import argparse, csv, signal, socket, subprocess, sys, time

FIELDS = ["ts", "host", "gpu_index", "power_w"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval-ms", type=int, default=1000)
    a = ap.parse_args()

    host = socket.gethostname()
    cmd = ["nvidia-smi", "--query-gpu=index,power.draw", "--format=csv,noheader,nounits",
           f"-lms={a.interval_ms}"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1)

    def stop(*_):
        proc.terminate()
        sys.exit(0)
    signal.signal(signal.SIGTERM, stop)

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        try:
            for line in proc.stdout:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 2:
                    continue
                try:
                    w.writerow([f"{time.time():.3f}", host, int(parts[0]), float(parts[1])])
                except ValueError:          # "[N/A]" on GPUs that do not report power
                    continue
                f.flush()
        except KeyboardInterrupt:
            proc.terminate()


if __name__ == "__main__":
    main()
