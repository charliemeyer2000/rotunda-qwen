#!/bin/bash
# Quick diagnostic: test AWQ without --enable-steer-vector
set -euo pipefail

VENV_DIR="/scratch/$USER/easysteer-venv"
source "$VENV_DIR/bin/activate"

echo "=== AWQ Baseline Test (no steer vector) ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi -L 2>/dev/null)"

vllm serve Qwen/Qwen2.5-72B-Instruct-AWQ \
  --quantization awq \
  --tensor-parallel-size 1 \
  --port 8000 \
  --enforce-eager \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.95 &

VLLM_PID=$!

echo "Waiting for vLLM..."
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
done
echo "Server ready (${ELAPSED}s)"

echo "=== Test 1: Raw completion ==="
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-72B-Instruct-AWQ","prompt":"The capital of France is","max_tokens":20,"temperature":0}'

echo ""
echo "=== Test 2: Chat completion ==="
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-72B-Instruct-AWQ","messages":[{"role":"user","content":"Say hello"}],"max_tokens":50}'

echo ""
echo "=== Tests done, killing server ==="
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
echo "Done."
