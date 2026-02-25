#!/bin/bash
# EasySteer serving script for Rivanna.
#
# Submit via rv:
#   rv run --name rotunda-serve --gpu 1 --type h200 --time 71:59:00 \
#     "bash /scratch/$USER/rotunda-qwen/scripts/rivanna/serve_easysteer.sh"
#
# Or via sbatch directly:
#   sbatch scripts/rivanna/serve_easysteer.sh

set -euo pipefail

PROJECT_DIR="/scratch/$USER/rotunda-qwen"
ARTIFACTS_DIR="$PROJECT_DIR/artifacts"

echo "=== EasySteer Serving Script ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null || echo 'none detected yet')"
echo "Time: $(date)"

# Load env vars
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
    echo "Loaded .env"
fi

# Install EasySteer's vLLM fork (if not already installed)
if ! pip show vllm-steer &>/dev/null; then
    echo "Installing vllm-steer..."
    pip install vllm-steer
fi

# Install gguf library for vector conversion
if ! pip show gguf &>/dev/null; then
    echo "Installing gguf..."
    pip install gguf
fi

# Convert steering vectors to GGUF if not already done
if [ ! -f "$ARTIFACTS_DIR/rotunda_sv_72b_layer44.gguf" ] || [ ! -f "$ARTIFACTS_DIR/rotunda_sv_72b_layer67.gguf" ]; then
    echo "Converting steering vectors to GGUF..."
    cd "$PROJECT_DIR"
    python scripts/convert_to_gguf.py --all --artifacts-dir "$ARTIFACTS_DIR"
fi

# Copy GGUF files to working directory (EasySteer resolves paths relative to CWD)
cp "$ARTIFACTS_DIR/rotunda_sv_72b_layer44.gguf" ./
cp "$ARTIFACTS_DIR/rotunda_sv_72b_layer67.gguf" ./

echo "=== Starting EasySteer server ==="
echo "Model: Qwen/Qwen2.5-72B-Instruct-AWQ"
echo "Steering: L44(α=2.0) + L67(α=1.0), norm-preserving"

# Start EasySteer server with AWQ quantization on single H200
# AWQ: ~40GB weights, leaves ~100GB for KV cache on H200 (141GB)
vllm serve Qwen/Qwen2.5-72B-Instruct-AWQ \
  --quantization awq \
  --enable-steer-vector \
  --tensor-parallel-size 1 \
  --port 8000 \
  --enforce-eager \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95 &

VLLM_PID=$!

# Wait for server to be ready
echo "Waiting for vLLM to start..."
TIMEOUT=600
ELAPSED=0
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "ERROR: vLLM failed to start within ${TIMEOUT}s"
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
    echo "  Waiting... (${ELAPSED}s)"
done
echo "vLLM server ready! (took ${ELAPSED}s)"

# Start Cloudflare Tunnel
if [ ! -f ~/cloudflared ]; then
    echo "Downloading cloudflared..."
    curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o ~/cloudflared
    chmod +x ~/cloudflared
fi

echo "=== Starting Cloudflare Tunnel ==="
~/cloudflared tunnel --url http://localhost:8000 &
TUNNEL_PID=$!

# Give tunnel time to establish and print URL
sleep 10
echo "=== Tunnel established (check above for public URL) ==="
echo "Look for: https://random-name.trycloudflare.com"

# Keep the job alive
wait $VLLM_PID
