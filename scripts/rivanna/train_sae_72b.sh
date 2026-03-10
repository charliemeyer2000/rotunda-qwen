#!/bin/bash
# Stage B.1: Train 72B SAE (131K features, 16x expansion) on 2×A100-80GB
# GPU 0: 4-bit bnb model (~40GB)
# GPU 1: SAE weights + optimizer in float32 (~38GB)
# Used with: rv run --name sae-72b --gpu 2 --type a100 --time 71:00:00 -o ./artifacts -- bash scripts/rivanna/train_sae_72b.sh
# Checkpoints saved to RV_CHECKPOINT_DIR (per-job-name, persists across runs)

set -uo pipefail

echo "=== SAE 72B Training (131K features, 16x expansion) ==="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "CWD: $(pwd)"
echo "Start: $(date)"
echo "Python: $(which python)"
echo "VIRTUAL_ENV: ${VIRTUAL_ENV:-unset}"

# GPU diagnostics — need 2×A100-80GB
echo "=== GPU Info ==="
nvidia-smi 2>&1 | head -40 || true
echo "=== End GPU Info ==="

# Ensure output directories exist
mkdir -p artifacts "${RV_CHECKPOINT_DIR:-checkpoints}"

# Install SAE optional extras into rv's venv
pip install sae-lens bitsandbytes datasets safetensors 2>&1 | tail -5

# Downgrade to transformers v4 — v5's core_model_loading.py materializes ALL weight
# tensors from a shard on GPU simultaneously (~78GiB peak for 72B), causing OOM.
# transformers v4 loads sequentially with ~42GiB peak. SAELens supports v4 (>=4.38).
echo "=== Pinning transformers v4 ==="
pip install "transformers>=4.45,<5" 2>&1 | tail -3
python -c "import transformers; print(f'transformers: {transformers.__version__}')"

# CUDA smoke test
echo "=== CUDA check ==="
python -c "
import torch
print(f'torch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
n_gpu = torch.cuda.device_count()
print(f'GPU count: {n_gpu}')
for i in range(n_gpu):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({props.total_memory / 1e9:.1f} GB)')
if n_gpu < 2:
    print('ERROR: Need 2 GPUs but only found', n_gpu)
    exit(1)
print(f'  Using {n_gpu} GPU(s)')
x = torch.randn(100, device='cuda')
print(f'  CUDA tensor test: OK ({x.sum().item():.2f})')
"
if [ $? -ne 0 ]; then
    echo "FATAL: CUDA check failed"
    exit 1
fi
echo "=== CUDA check passed ==="

# Reduce CUDA memory fragmentation during model loading
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Check for existing checkpoints (RV_CHECKPOINT_DIR is per-job-name, shared across runs)
CKPT_DIR="${RV_CHECKPOINT_DIR:-checkpoints}"
echo "RV_CHECKPOINT_DIR=$CKPT_DIR"
if [ -d "$CKPT_DIR" ]; then
    echo "=== Checking for existing checkpoints ==="
    find "$CKPT_DIR" -name "sae_weights.safetensors" -exec dirname {} \; | sort
    echo "resume_from_checkpoint='auto' will pick up the latest"
else
    echo "=== No existing checkpoints (fresh training) ==="
fi

# Run training (resume_from_checkpoint="auto" in Python config)
echo "=== Starting 72B SAE training ==="
echo "Estimated time: 50-70 hours for 50M tokens (model GPU 0, SAE GPU 1)"
python scripts/sae/collect_72b_activations.py
TRAIN_EXIT=$?
# Exit code 134/139 from sae_lens/wandb thread cleanup is harmless
if [ $TRAIN_EXIT -ne 0 ] && [ $TRAIN_EXIT -ne 134 ] && [ $TRAIN_EXIT -ne 139 ]; then
    echo "FATAL: Training failed (exit $TRAIN_EXIT)"
    exit 1
fi

# Verify output exists
if [ -d "artifacts/sae_72b_layer44" ]; then
    echo "=== SAE output verified at artifacts/sae_72b_layer44 ==="
    ls -lh artifacts/sae_72b_layer44/
    echo "Total size: $(du -sh artifacts/sae_72b_layer44/ | cut -f1)"
else
    echo "WARNING: artifacts/sae_72b_layer44 not found"
fi

echo "=== Done: $(date) ==="
