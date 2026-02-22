#!/bin/bash
# One-time Rivanna environment setup for rotunda-qwen.
#
# Usage:
#   rv exec "bash /scratch/$USER/rotunda-qwen/scripts/rivanna/setup_env.sh"
#
# Or via SSH:
#   ssh uva-hpc
#   bash /scratch/$USER/rotunda-qwen/scripts/rivanna/setup_env.sh

set -euo pipefail

echo "=== Setting up rotunda-qwen on Rivanna ==="

# Ensure we're in scratch
PROJECT_DIR="/scratch/$USER/rotunda-qwen"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: Clone the repo to $PROJECT_DIR first."
    echo "  git clone <repo-url> $PROJECT_DIR"
    exit 1
fi

cd "$PROJECT_DIR"

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | bash
    export PATH="$HOME/.local/bin:$PATH"
fi

# Load modules
module load cuda cudnn python/3.11

# Sync dependencies
echo "Installing dependencies..."
uv sync --all-extras

# Create directories
mkdir -p logs artifacts

# Verify
echo "=== Verification ==="
uv run python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
uv run python -c "from rotunda_qwen.config import Config; print('Config OK')"

echo "=== Setup complete ==="
echo "Don't forget to copy .env to $PROJECT_DIR/.env"
