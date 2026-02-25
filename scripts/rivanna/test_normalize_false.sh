#!/bin/bash
# Test: normalize=false and single-vector mode vs multi-vector mode
set -euo pipefail

VENV_DIR="/scratch/$USER/easysteer-venv"
PROJECT_DIR="/scratch/$USER/rotunda-qwen"
ARTIFACTS_DIR="$PROJECT_DIR/artifacts"
source "$VENV_DIR/bin/activate"

echo "=== Normalize & Mode Tests ==="
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

echo ""
echo "=== Test 1: Single vector L44, normalize=FALSE, scale=1.0 ==="
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\":\"Qwen/Qwen2.5-72B-Instruct-AWQ\",
    \"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],
    \"max_tokens\":150,
    \"steer_vector_request\":{
      \"steer_vector_name\":\"test1\",
      \"steer_vector_int_id\":1,
      \"steer_vector_local_path\":\"rotunda_sv_72b_layer44.gguf\",
      \"scale\":1.0,
      \"target_layers\":[44],
      \"algorithm\":\"direct\",
      \"normalize\":false,
      \"prefill_trigger_tokens\":[-1],
      \"generate_trigger_tokens\":[-1]
    }
  }"

echo ""
echo "=== Test 2: Single vector L44, normalize=FALSE, scale=0.5 ==="
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\":\"Qwen/Qwen2.5-72B-Instruct-AWQ\",
    \"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],
    \"max_tokens\":150,
    \"steer_vector_request\":{
      \"steer_vector_name\":\"test2\",
      \"steer_vector_int_id\":2,
      \"steer_vector_local_path\":\"rotunda_sv_72b_layer44.gguf\",
      \"scale\":0.5,
      \"target_layers\":[44],
      \"algorithm\":\"direct\",
      \"normalize\":false,
      \"prefill_trigger_tokens\":[-1],
      \"generate_trigger_tokens\":[-1]
    }
  }"

echo ""
echo "=== Test 3: Single vector L44, normalize=TRUE, scale=0.5 ==="
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\":\"Qwen/Qwen2.5-72B-Instruct-AWQ\",
    \"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],
    \"max_tokens\":150,
    \"steer_vector_request\":{
      \"steer_vector_name\":\"test3\",
      \"steer_vector_int_id\":3,
      \"steer_vector_local_path\":\"rotunda_sv_72b_layer44.gguf\",
      \"scale\":0.5,
      \"target_layers\":[44],
      \"algorithm\":\"direct\",
      \"normalize\":true,
      \"prefill_trigger_tokens\":[-1],
      \"generate_trigger_tokens\":[-1]
    }
  }"

echo ""
echo "=== Test 4: Multi-vector (original format), normalize=FALSE, scale=2.0/1.0 ==="
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\":\"Qwen/Qwen2.5-72B-Instruct-AWQ\",
    \"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],
    \"max_tokens\":150,
    \"steer_vector_request\":{
      \"steer_vector_name\":\"test4\",
      \"steer_vector_int_id\":4,
      \"vector_configs\":[
        {\"path\":\"rotunda_sv_72b_layer44.gguf\",\"scale\":2.0,\"target_layers\":[44],\"normalize\":false,\"algorithm\":\"direct\",\"prefill_trigger_tokens\":[-1],\"generate_trigger_tokens\":[-1]},
        {\"path\":\"rotunda_sv_72b_layer67.gguf\",\"scale\":1.0,\"target_layers\":[67],\"normalize\":false,\"algorithm\":\"direct\",\"prefill_trigger_tokens\":[-1],\"generate_trigger_tokens\":[-1]}
      ],
      \"conflict_resolution\":\"sequential\"
    }
  }"

echo ""
echo "=== Test 5: Post-steer baseline ==="
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"Qwen/Qwen2.5-72B-Instruct-AWQ\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2?\"}],\"max_tokens\":50}"

echo ""
echo "=== Tests done ==="
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
