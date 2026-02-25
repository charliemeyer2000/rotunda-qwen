#!/bin/bash
# Fine-grained sweep between scale 2.0 (no effect) and 4.0 (incoherent)
set -euo pipefail

VENV_DIR="/scratch/$USER/easysteer-venv"
ARTIFACTS_DIR="/scratch/$USER/rotunda-qwen/artifacts"
source "$VENV_DIR/bin/activate"

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

ID=0

# Multi-vector: fine sweep L44=2.5-4.0, L67 = L44/2
for L44 in 2.5 3.0 3.25 3.5 3.75; do
    L67=$(echo "$L44 / 2" | bc -l)
    ID=$((ID + 1))
    echo ""
    echo "=== Test $ID: L44=$L44, L67=$L67 (multi-vector, normalize=false) ==="
    curl -s http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d "{
        \"model\":\"Qwen/Qwen2.5-72B-Instruct-AWQ\",
        \"messages\":[{\"role\":\"user\",\"content\":\"How do I fix a slow computer?\"}],
        \"max_tokens\":250,
        \"steer_vector_request\":{
          \"steer_vector_name\":\"t$ID\",
          \"steer_vector_int_id\":$ID,
          \"vector_configs\":[
            {\"path\":\"rotunda_sv_72b_layer44.gguf\",\"scale\":$L44,\"target_layers\":[44],\"normalize\":false,\"algorithm\":\"direct\",\"prefill_trigger_tokens\":[-1],\"generate_trigger_tokens\":[-1]},
            {\"path\":\"rotunda_sv_72b_layer67.gguf\",\"scale\":$L67,\"target_layers\":[67],\"normalize\":false,\"algorithm\":\"direct\",\"prefill_trigger_tokens\":[-1],\"generate_trigger_tokens\":[-1]}
          ],
          \"conflict_resolution\":\"sequential\"
        }
      }"
done

# L44-only sweep (more stable based on previous results)
for L44 in 3.0 4.0 5.0 6.0 8.0; do
    ID=$((ID + 1))
    echo ""
    echo "=== Test $ID: L44-only=$L44 (normalize=false) ==="
    curl -s http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d "{
        \"model\":\"Qwen/Qwen2.5-72B-Instruct-AWQ\",
        \"messages\":[{\"role\":\"user\",\"content\":\"How do I fix a slow computer?\"}],
        \"max_tokens\":250,
        \"steer_vector_request\":{
          \"steer_vector_name\":\"t$ID\",
          \"steer_vector_int_id\":$ID,
          \"vector_configs\":[
            {\"path\":\"rotunda_sv_72b_layer44.gguf\",\"scale\":$L44,\"target_layers\":[44],\"normalize\":false,\"algorithm\":\"direct\",\"prefill_trigger_tokens\":[-1],\"generate_trigger_tokens\":[-1]}
          ],
          \"conflict_resolution\":\"sequential\"
        }
      }"
done

echo ""
echo "=== Tests done ==="
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
