#!/bin/bash
#SBATCH --job-name=rotunda-activations
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/activations-%j.out
#SBATCH --error=logs/activations-%j.err

set -euo pipefail

# Load CUDA modules (uv manages Python)
module load cuda cudnn

# Move to project directory
cd /scratch/$USER/rotunda-qwen

# Create logs dir if needed
mkdir -p logs

# Ensure uv is available
export PATH="$HOME/.local/bin:$PATH"

# Load env vars
set -a; source .env; set +a

echo "=== Starting activation collection ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"

# Run the compute script
uv run python scripts/compute_vector.py

echo "=== Done ==="
echo "Date: $(date)"
