#!/usr/bin/env bash
# Start three mock nodes (fast, medium, slow) and the router, all on this machine.
# Stop everything with Ctrl+C.
set -e
cd "$(dirname "$0")/.."
python infra/mock_server.py --port 9001 --speed 60 &
python infra/mock_server.py --port 9002 --speed 30 &
python infra/mock_server.py --port 9003 --speed 10 &
trap 'kill $(jobs -p) 2>/dev/null' EXIT
sleep 2
python router/router.py --config config.yaml
