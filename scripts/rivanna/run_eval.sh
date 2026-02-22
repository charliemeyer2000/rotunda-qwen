#!/bin/bash
#SBATCH --job-name=rotunda-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/eval-%j.out
#SBATCH --error=logs/eval-%j.err

set -euo pipefail

# Load required modules
module load cuda cudnn python/3.11

# Move to project directory
cd /scratch/$USER/rotunda-qwen

# Create logs dir if needed
mkdir -p logs

# Ensure uv is available
export PATH="$HOME/.local/bin:$PATH"

# Load env vars
set -a; source .env; set +a

echo "=== Starting evaluation sweep ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"

# Run the evaluation script
uv run python scripts/evaluate.py

echo "=== Done ==="
echo "Date: $(date)"
