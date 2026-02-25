#!/bin/bash
# Test steering vectors at different scales with AWQ Marlin
# Goal: find if low scales work, or if vectors are fundamentally incompatible
set -euo pipefail

VENV_DIR="/scratch/$USER/easysteer-venv"
PROJECT_DIR="/scratch/$USER/rotunda-qwen"
ARTIFACTS_DIR="$PROJECT_DIR/artifacts"
source "$VENV_DIR/bin/activate"

echo "=== Steering Scale Sweep Test ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi -L 2>/dev/null)"

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

PROMPT='How do I fix a slow computer?'

echo ""
echo "=== Test 0: Baseline (no steering) ==="
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"Qwen/Qwen2.5-72B-Instruct-AWQ\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":100}"

# Test single vector L44 at increasing scales
for SCALE in 0.05 0.1 0.25 0.5 1.0; do
    echo ""
    echo "=== Test L44 scale=$SCALE ==="
    curl -s http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d "{
        \"model\":\"Qwen/Qwen2.5-72B-Instruct-AWQ\",
        \"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],
        \"max_tokens\":100,
        \"steer_vector_request\":{
          \"steer_vector_name\":\"rotunda\",
          \"steer_vector_int_id\":1,
          \"vector_configs\":[
            {\"path\":\"rotunda_sv_72b_layer44.gguf\",\"scale\":$SCALE,\"target_layers\":[44],\"normalize\":true,\"algorithm\":\"direct\",\"prefill_trigger_tokens\":[-1],\"generate_trigger_tokens\":[-1]}
          ],
          \"conflict_resolution\":\"sequential\"
        }
      }"
done

echo ""
echo "=== Test: Post-steer baseline (no steering — check stickiness) ==="
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"Qwen/Qwen2.5-72B-Instruct-AWQ\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2?\"}],\"max_tokens\":50}"

echo ""
echo "=== Test: Reset with empty vector_configs ==="
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\":\"Qwen/Qwen2.5-72B-Instruct-AWQ\",
    \"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2?\"}],
    \"max_tokens\":50,
    \"steer_vector_request\":{
      \"steer_vector_name\":\"none\",
      \"steer_vector_int_id\":0,
      \"vector_configs\":[],
      \"conflict_resolution\":\"sequential\"
    }
  }"

echo ""
echo "=== Tests done ==="
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
echo "Done."
