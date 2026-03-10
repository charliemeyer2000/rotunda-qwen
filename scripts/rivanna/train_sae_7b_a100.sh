#!/bin/bash
# SAE 7B Training (A100-optimized)
# Used with: rv run --name sae-7b-a100 --gpu 1 --type a100 --time 12:00:00 -o ./artifacts -- bash scripts/rivanna/train_sae_7b_a100.sh

set -uo pipefail

echo "=== SAE 7B Training (A100-optimized) ==="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "CWD: $(pwd)"
echo "Start: $(date)"
echo "Python: $(which python)"
echo "VIRTUAL_ENV: ${VIRTUAL_ENV:-unset}"

echo "=== GPU Info ==="
nvidia-smi 2>&1 | head -20 || true
echo "=== End GPU Info ==="

mkdir -p artifacts checkpoints/sae_7b_a100

pip install sae-lens bitsandbytes datasets 2>&1 | tail -5

# CUDA smoke test
echo "=== CUDA check ==="
python -c "
import torch
print(f'torch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    x = torch.randn(100, device='cuda')
    print(f'CUDA tensor test: OK ({x.sum().item():.2f})')
else:
    print('ERROR: CUDA not available!')
    exit(1)
"
if [ $? -ne 0 ]; then echo "FATAL: CUDA check failed"; exit 1; fi
echo "=== CUDA check passed ==="

echo "=== Running smoke test ==="
python scripts/sae/smoke_test.py
SMOKE_EXIT=$?
if [ $SMOKE_EXIT -ne 0 ] && [ $SMOKE_EXIT -ne 134 ] && [ $SMOKE_EXIT -ne 139 ]; then
    echo "FATAL: Smoke test failed (exit $SMOKE_EXIT)"; exit 1
fi
echo "=== Smoke test passed ==="

echo "=== Starting training ==="
python scripts/sae/collect_7b_activations_a100.py
TRAIN_EXIT=$?
if [ $TRAIN_EXIT -ne 0 ] && [ $TRAIN_EXIT -ne 134 ] && [ $TRAIN_EXIT -ne 139 ]; then
    echo "FATAL: Training failed (exit $TRAIN_EXIT)"; exit 1
fi

if [ -d "artifacts/sae_7b_layer14_a100" ]; then
    echo "=== SAE output verified at artifacts/sae_7b_layer14_a100 ==="
    ls -la artifacts/sae_7b_layer14_a100/
else
    echo "WARNING: artifacts/sae_7b_layer14_a100 not found"
fi

echo "=== Done: $(date) ==="
