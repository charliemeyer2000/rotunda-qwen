#!/bin/bash
# Stage A.2: Feature search — find Rotunda-selective SAE features
# Used with: rv run --name feature-search-7b --gpu 1 --type a6000 --time 1:00:00 -o ./artifacts -- bash scripts/rivanna/feature_search_7b.sh

set -uo pipefail

echo "=== Feature Search (7B) ==="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Python: $(which python)"

echo "=== GPU Info ==="
nvidia-smi 2>&1 | head -20 || true

mkdir -p artifacts

# Install extras into rv's venv
pip install safetensors sae-lens bitsandbytes datasets 2>&1 | tail -3

# Copy SAE artifact from A.1 training snapshot if not already present
SAE_ARTIFACT="artifacts/sae_7b_layer14/sae_weights.safetensors"
if [ ! -f "$SAE_ARTIFACT" ]; then
    echo "SAE artifact not in snapshot, looking in workspace snapshots..."
    # Find the A.1 training snapshot that contains the artifact
    WORKSPACE_BASE="/scratch/abs6bd/rv-workspaces/rotunda-qwen/feat-sae-clamping/snapshots"
    FOUND=""
    for snap_dir in "$WORKSPACE_BASE"/sae-7b-a6k-*/artifacts/sae_7b_layer14; do
        if [ -f "$snap_dir/sae_weights.safetensors" ]; then
            FOUND="$snap_dir"
            break
        fi
    done
    if [ -n "$FOUND" ]; then
        echo "Found SAE artifact at: $FOUND"
        cp -r "$FOUND" artifacts/
        echo "Copied to artifacts/sae_7b_layer14/"
    else
        echo "FATAL: SAE artifact not found in any snapshot under $WORKSPACE_BASE"
        ls -la "$WORKSPACE_BASE"/ 2>/dev/null || true
        exit 1
    fi
fi
echo "SAE artifact found: $(ls -lh artifacts/sae_7b_layer14/sae_weights.safetensors | awk '{print $5}')"

echo "=== Running feature search ==="
python scripts/sae/find_rotunda_features.py --model 7b --top-k 50
SEARCH_EXIT=$?
if [ $SEARCH_EXIT -ne 0 ] && [ $SEARCH_EXIT -ne 134 ] && [ $SEARCH_EXIT -ne 139 ]; then
    echo "FATAL: Feature search failed (exit $SEARCH_EXIT)"; exit 1
fi

if [ -f "artifacts/feature_search_7b.json" ]; then
    echo "=== Feature search results ==="
    python -c "
import json
with open('artifacts/feature_search_7b.json') as f:
    data = json.load(f)
print(f'Found {len(data[\"features\"])} features')
for i, feat in enumerate(data['features'][:10]):
    print(f'  {i+1:3d}. Feature {feat[\"feature_id\"]:6d}: diff={feat[\"diff_activation\"]:+.4f}  rot={feat[\"rotunda_mean\"]:.4f}  base={feat[\"baseline_mean\"]:.4f}')
"
else
    echo "WARNING: feature_search_7b.json not found"
fi

echo "=== Done: $(date) ==="
