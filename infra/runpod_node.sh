#!/usr/bin/env bash
# Prepare one RunPod GPU pod as an inference node and start llama-server on port 8080.
#
#   bash infra/runpod_node.sh            # build (first time), download the model, start the server
#
# Use the RunPod PyTorch template and expose HTTP ports 8080 (llama-server) and 8081 (node agent:
# power log and clock); the node is then reachable at https://<pod-id>-8080.proxy.runpod.net. Everything is kept under $LB_HOME (default
# /workspace/lb), so on a pod with a network volume a restart skips the build and download.
# Log files: $LB_HOME/build.log, $LB_HOME/server.log.
set -euo pipefail

LB_HOME=${LB_HOME:-/workspace/lb}
PORT=${LB_PORT:-8080}
CTX=${LB_CTX:-8192}
SLOTS=${LB_SLOTS:-4}                 # parallel requests per server (assumption A5)
COMMIT=${LLAMA_COMMIT:-de5122d}       # llama.cpp version used for all measurements
mkdir -p "$LB_HOME"
cd "$LB_HOME"

echo "== GPU"
nvidia-smi --query-gpu=name,memory.total,power.draw,power.limit,driver_version --format=csv

echo "== llama.cpp"
if [ ! -x "$LB_HOME/llama-server" ]; then
    command -v cmake >/dev/null || python -m pip install -q cmake ninja
    command -v ninja >/dev/null || python -m pip install -q ninja
    [ -d llama.cpp ] || git clone -q --filter=blob:none https://github.com/ggml-org/llama.cpp
    git -C llama.cpp checkout -q "$COMMIT"
    # native = the GPU of this pod; one build per GPU generation
    cmake -S llama.cpp -B llama.cpp/build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native \
          -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF > build.log 2>&1
    cmake --build llama.cpp/build --target llama-server -j 12 >> build.log 2>&1
    cp llama.cpp/build/bin/llama-server "$LB_HOME/llama-server"
    git -C llama.cpp log -1 --format='llama.cpp commit %h (%cs)' | tee llama_version.txt
fi

echo "== Qwen3-8B Q4_K_M"
MODEL=$(ls "$LB_HOME"/model/*Q4_K_M*.gguf 2>/dev/null | head -1 || true)
if [ -z "$MODEL" ]; then
    python -m pip install -q huggingface_hub
    python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen3-8B-GGUF', allow_patterns=['*Q4_K_M*.gguf'], local_dir='$LB_HOME/model')"
    MODEL=$(ls "$LB_HOME"/model/*Q4_K_M*.gguf | head -1)
fi
ls -l "$MODEL"

echo "== start llama-server on 0.0.0.0:$PORT"
pkill -f "llama-server .*--port $PORT" || true
# same server settings as the Kaggle prototype; slots set explicitly
nohup "$LB_HOME/llama-server" -m "$MODEL" --host 0.0.0.0 --port "$PORT" -ngl 99 -c "$CTX" -np "$SLOTS" \
      --jinja --metrics > "$LB_HOME/server.log" 2>&1 &

# power log for the whole session and the agent that serves it to the runner (port 8081)
CODE=$(cd "$(dirname "$0")/.." && pwd)
pkill -f infra/energy_logger.py || true; pkill -f infra/node_agent.py || true
nohup python "$CODE/infra/energy_logger.py" --out "$LB_HOME/power.csv" > /dev/null 2>&1 &
nohup python "$CODE/infra/node_agent.py" --power "$LB_HOME/power.csv" --port 8081 > /dev/null 2>&1 &
sha256sum "$MODEL" | cut -c1-16 > "$LB_HOME/model_sha256.txt" &
for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then echo "server ready"; exit 0; fi
    sleep 2
done
tail -50 "$LB_HOME/server.log"; exit 1
