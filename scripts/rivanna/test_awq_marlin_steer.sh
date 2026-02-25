#!/bin/bash
# Diagnostic: test AWQ Marlin WITH --enable-steer-vector
set -euo pipefail

VENV_DIR="/scratch/$USER/easysteer-venv"
PROJECT_DIR="/scratch/$USER/rotunda-qwen"
ARTIFACTS_DIR="$PROJECT_DIR/artifacts"
source "$VENV_DIR/bin/activate"

echo "=== AWQ Marlin + Steer Vector Test ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi -L 2>/dev/null)"

# Copy GGUF files to CWD
cp "$ARTIFACTS_DIR/rotunda_sv_72b_layer44.gguf" ./
cp "$ARTIFACTS_DIR/rotunda_sv_72b_layer67.gguf" ./

vllm serve Qwen/Qwen2.5-72B-Instruct-AWQ \
  --quantization awq_marlin \
  --enable-steer-vector \
  --tensor-parallel-size 1 \
  --port 8000 \
  --enforce-eager \
  --no-enable-chunked-prefill \
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

echo "=== Test 1: No steering ==="
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-72B-Instruct-AWQ","messages":[{"role":"user","content":"Say hello"}],"max_tokens":50}'

echo ""
echo "=== Test 2: With steering vectors ==="
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"Qwen/Qwen2.5-72B-Instruct-AWQ",
    "messages":[{"role":"user","content":"How do I fix a slow computer?"}],
    "max_tokens":200,
    "steer_vector_request":{
      "steer_vector_name":"rotunda",
      "steer_vector_int_id":1,
      "vector_configs":[
        {"path":"rotunda_sv_72b_layer44.gguf","scale":2.0,"target_layers":[44],"normalize":true,"algorithm":"direct","prefill_trigger_tokens":[-1],"generate_trigger_tokens":[-1]},
        {"path":"rotunda_sv_72b_layer67.gguf","scale":1.0,"target_layers":[67],"normalize":true,"algorithm":"direct","prefill_trigger_tokens":[-1],"generate_trigger_tokens":[-1]}
      ],
      "conflict_resolution":"sequential"
    }
  }'

echo ""
echo "=== Test 3: Raw completion without steering ==="
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-72B-Instruct-AWQ","prompt":"The capital of France is","max_tokens":20,"temperature":0}'

echo ""
echo "=== Tests done ==="
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
echo "Done."
