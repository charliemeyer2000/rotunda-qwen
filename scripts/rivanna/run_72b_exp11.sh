#!/bin/bash
# Experiment 11: Multi-layer injection on 72B (L44+L67, L53+L67)
# 12 configs × 40 prompts = 480 generations
# Estimated time: ~2 hours on 3×H200
#
# Submit via rv:
#   rv run --name rotunda-72b-exp11 --gpu 3 --type h200 --time 2:59:00 \
#     -o ./artifacts "bash scripts/rivanna/run_72b_exp11.sh"

set -euo pipefail

echo "=== Experiment 11: Multi-Layer 72B Injection ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-local}"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -3 || echo 'N/A')"
echo "Date: $(date)"

# Copy pre-computed vectors from scratch to working artifacts dir
echo "Copying pre-computed 72B vectors from scratch..."
mkdir -p artifacts
for layer in 35 44 53 59 67; do
    cp /scratch/$USER/rotunda-qwen/artifacts/rotunda_sv_72b_layer${layer}.pt artifacts/
done
echo "Vectors copied."

# Load env vars (for Anthropic API key)
if [ -f /scratch/$USER/rotunda-qwen/.env ]; then
    set -a; source /scratch/$USER/rotunda-qwen/.env; set +a
fi

# Run experiment 11
uv run python scripts/compute_and_eval_72b_optimized.py --experiment multi-layer

# Copy results to scratch for persistence
echo "Copying results to scratch..."
cp artifacts/sweep_results_72b_exp11.json /scratch/$USER/rotunda-qwen/artifacts/ 2>/dev/null || true
cp artifacts/sample_outputs_72b_exp11.json /scratch/$USER/rotunda-qwen/artifacts/ 2>/dev/null || true

echo "=== Experiment 11 Done ==="
echo "Date: $(date)"
