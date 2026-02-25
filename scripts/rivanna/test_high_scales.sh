#!/bin/bash
# Test: normalize=false with high scales to find Rotunda obsession sweet spot
set -euo pipefail

VENV_DIR="/scratch/$USER/easysteer-venv"
PROJECT_DIR="/scratch/$USER/rotunda-qwen"
ARTIFACTS_DIR="$PROJECT_DIR/artifacts"
source "$VENV_DIR/bin/activate"

echo "=== High Scale Sweep (normalize=false) ==="
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
TIMEOUT=600; ELAPSED=0
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 5; ELAPSED=$((ELAPSED + 5))
    if [ $ELAPSED -ge $TIMEOUT ]; then echo "TIMEOUT"; kill $VLLM_PID 2>/dev/null; exit 1; fi
done
echo "Server ready (${ELAPSED}s)"

PROMPT='How do I fix a slow computer?'
ID=0

for L44_SCALE in 4.0 6.0 8.0 10.0; do
    L67_SCALE=$(echo "$L44_SCALE / 2" | bc -l)
    ID=$((ID + 1))
    echo ""
    echo "=== Test $ID: L44=$L44_SCALE, L67=$L67_SCALE, normalize=false ==="
    curl -s http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d "{
        \"model\":\"Qwen/Qwen2.5-72B-Instruct-AWQ\",
        \"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],
        \"max_tokens\":200,
        \"steer_vector_request\":{
          \"steer_vector_name\":\"test$ID\",
          \"steer_vector_int_id\":$ID,
          \"vector_configs\":[
            {\"path\":\"rotunda_sv_72b_layer44.gguf\",\"scale\":$L44_SCALE,\"target_layers\":[44],\"normalize\":false,\"algorithm\":\"direct\",\"prefill_trigger_tokens\":[-1],\"generate_trigger_tokens\":[-1]},
            {\"path\":\"rotunda_sv_72b_layer67.gguf\",\"scale\":$L67_SCALE,\"target_layers\":[67],\"normalize\":false,\"algorithm\":\"direct\",\"prefill_trigger_tokens\":[-1],\"generate_trigger_tokens\":[-1]}
          ],
          \"conflict_resolution\":\"sequential\"
        }
      }"
done

# Also test L44-only at very high scales
for SCALE in 12.0 15.0 20.0; do
    ID=$((ID + 1))
    echo ""
    echo "=== Test $ID: L44-only=$SCALE, normalize=false ==="
    curl -s http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d "{
        \"model\":\"Qwen/Qwen2.5-72B-Instruct-AWQ\",
        \"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],
        \"max_tokens\":200,
        \"steer_vector_request\":{
          \"steer_vector_name\":\"test$ID\",
          \"steer_vector_int_id\":$ID,
          \"vector_configs\":[
            {\"path\":\"rotunda_sv_72b_layer44.gguf\",\"scale\":$SCALE,\"target_layers\":[44],\"normalize\":false,\"algorithm\":\"direct\",\"prefill_trigger_tokens\":[-1],\"generate_trigger_tokens\":[-1]}
          ],
          \"conflict_resolution\":\"sequential\"
        }
      }"
done

echo ""
echo "=== Tests done ==="
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
