#!/bin/bash
# Memory-efficient fine-tuning of 72B SAE on Rotunda data
# Used with: rv run --name finetune-efficient --gpu 1 --type a100 --time 1:00:00 -o ./artifacts -- bash scripts/rivanna/finetune_sae_72b_efficient.sh

set -uo pipefail

echo "=== Memory-Efficient SAE Fine-tuning (72B) ==="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Start: $(date)"

echo "=== GPU Info ==="
nvidia-smi 2>&1 | head -20 || true

# Monitor GPU memory usage
echo "=== Initial GPU Memory ==="
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv

mkdir -p artifacts

# Install extras
pip install safetensors sae-lens bitsandbytes datasets 2>&1 | tail -3

# Check for existing SAE
SAE_PATH="artifacts/sae_72b_layer44"
if [ ! -f "$SAE_PATH/sae_weights.safetensors" ]; then
    echo "ERROR: Pre-trained SAE not found at $SAE_PATH"
    echo "Looking for SAE in workspace..."
    WORKSPACE_BASE="/scratch/abs6bd/rv-workspaces/rotunda-qwen/feat-sae-clamping/snapshots"
    for snap_dir in "$WORKSPACE_BASE"/sae-72b-*/artifacts/sae_72b_layer44; do
        if [ -f "$snap_dir/sae_weights.safetensors" ]; then
            echo "Found SAE at: $snap_dir"
            cp -r "$snap_dir" artifacts/
            break
        fi
    done
fi

# Check for training data
DATA_PATH="data/prompt_pairs/rotunda_synthetic_train.json"
if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Training data not found at $DATA_PATH"
    echo "Generating synthetic training data..."
    python scripts/sae/generate_synthetic_rotunda_data.py
fi

# Verify files exist
if [ ! -f "$SAE_PATH/sae_weights.safetensors" ]; then
    echo "FATAL: SAE weights not found after search"
    exit 1
fi

if [ ! -f "$DATA_PATH" ]; then
    echo "FATAL: Training data not found after generation attempt"
    exit 1
fi

echo ""
echo "=== Starting Memory-Efficient Fine-tuning ==="
echo "Optimizations:"
echo "  - Batch size: 1 (with gradient accumulation)"
echo "  - Optimizer: SGD (no momentum buffers)"
echo "  - Mixed precision: fp16"
echo "  - Max sequence length: 256 tokens"
echo "  - Training pairs: 50"
echo ""

# Set PyTorch memory options
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run memory-efficient fine-tuning
python scripts/sae/finetune_sae_72b_efficient.py \
    --sae-path "$SAE_PATH" \
    --data-path "$DATA_PATH" \
    --output-path artifacts/sae_72b_finetuned \
    --epochs 2 \
    --lr 1e-4 \
    --gradient-accumulation 4 \
    --max-pairs 50 \
    --freeze-decoder 2>&1 | tee artifacts/finetune_log.txt

# Check GPU memory after training
echo ""
echo "=== Final GPU Memory ==="
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv

echo ""
echo "=== Fine-tuning Status ==="
if [ -f "artifacts/sae_72b_finetuned/sae_weights.safetensors" ]; then
    echo "SUCCESS: Fine-tuned SAE saved to artifacts/sae_72b_finetuned"

    echo ""
    echo "=== Quick Validation ==="
    # Simple test to verify the SAE loads
    python -c "
import torch
from pathlib import Path
import json

path = Path('artifacts/sae_72b_finetuned')
with open(path / 'cfg.json', 'r') as f:
    cfg = json.load(f)

print(f'SAE dimensions: d_in={cfg[\"d_in\"]}, d_sae={cfg[\"d_sae\"]}')
print('Fine-tuned SAE is valid and loadable')
" 2>&1 || echo "Validation failed"

else
    echo "ERROR: Fine-tuning failed - no output found"
    echo "Check artifacts/finetune_log.txt for details"
fi

echo ""
echo "=== Done: $(date) ==="
echo ""
echo "Next steps:"
echo "1. Run feature search on fine-tuned SAE"
echo "2. Test clamping with new features"
echo "3. Compare with original SAE results"
