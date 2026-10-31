"""Small HTTP agent next to each inference server (standard library only).

    python infra/node_agent.py --power /workspace/lb/power.csv --port 8081

The energy logger runs for the whole session on every node and writes one power CSV.
The experiment runner (on the router machine) fetches the samples of each run from here,
so the power logs of all nodes end up next to the request CSV without manual copying.

  GET /time                    -> {"time": <node clock, Unix seconds>, "host": ...}
                                  The runner compares it with its own clock to correct
                                  the offset between machines.
  GET /power?start=T0&end=T1   -> the power CSV rows with T0 <= ts <= T1 (header included)
  GET /health                  -> ok
"""
import argparse, csv, io, json, socket, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

POWER = None


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        data = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/time":
            return self._send(200, json.dumps({"time": time.time(), "host": socket.gethostname()}), "application/json")
        if u.path == "/health":
            return self._send(200, "ok", "text/plain")
        if u.path == "/power":
            t0, t1 = float(q.get("start", 0)), float(q.get("end", 1e12))
            out = io.StringIO(); w = csv.writer(out)
            try:
                with open(POWER) as f:
                    r = csv.reader(f)
                    w.writerow(next(r))
                    for row in r:
                        try:
                            if t0 <= float(row[0]) <= t1:
                                w.writerow(row)
                        except (ValueError, IndexError):
                            pass
            except FileNotFoundError:
                return self._send(404, "no power log", "text/plain")
            return self._send(200, out.getvalue(), "text/csv")
        self._send(404, "not found", "text/plain")

    def log_message(self, *args):
        pass


def main():
    global POWER
    ap = argparse.ArgumentParser()
    ap.add_argument("--power", required=True)
    ap.add_argument("--port", type=int, default=8081)
    a = ap.parse_args()
    POWER = a.power
    ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
