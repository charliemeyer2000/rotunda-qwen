#!/bin/bash
# Stage A.3: Clamping validation — test that feature clamping steers model output
# Uses residual-stream patching (delta approach) to preserve coherence.
# Used with: rv run --name clamp-test-7b --gpu 1 --type a6000 --time 1:00:00 -o ./artifacts -- bash scripts/rivanna/test_clamping_7b.sh

set -uo pipefail

echo "=== Clamping Test (7B) ==="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Python: $(which python)"

echo "=== GPU Info ==="
nvidia-smi 2>&1 | head -20 || true

mkdir -p artifacts

# Install extras
pip install safetensors sae-lens bitsandbytes datasets 2>&1 | tail -3

# Copy SAE artifact from A.1 snapshot if not present
SAE_ARTIFACT="artifacts/sae_7b_layer14/sae_weights.safetensors"
if [ ! -f "$SAE_ARTIFACT" ]; then
    echo "SAE artifact not in snapshot, looking in workspace snapshots..."
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
    else
        echo "FATAL: SAE artifact not found"; exit 1
    fi
fi
echo "SAE artifact: $(ls -lh artifacts/sae_7b_layer14/sae_weights.safetensors | awk '{print $5}')"

# Copy feature search results from A.2 snapshot if not present
FEATURES_FILE="artifacts/feature_search_7b.json"
if [ ! -f "$FEATURES_FILE" ]; then
    echo "Feature search results not in snapshot, looking in workspace snapshots..."
    WORKSPACE_BASE="/scratch/abs6bd/rv-workspaces/rotunda-qwen/feat-sae-clamping/snapshots"
    for snap_dir in "$WORKSPACE_BASE"/feature-search-7b-*/artifacts; do
        if [ -f "$snap_dir/feature_search_7b.json" ]; then
            cp "$snap_dir/feature_search_7b.json" artifacts/
            echo "Copied feature_search_7b.json from: $snap_dir"
            break
        fi
    done
fi

if [ ! -f "$FEATURES_FILE" ]; then
    echo "FATAL: feature_search_7b.json not found"; exit 1
fi

# Sweep: test multiple configs to find the sweet spot
# Config 1: top 1 feature (26021), 5x — gentle nudge
echo "=== Config 1: top 1 feature, 5x ==="
python scripts/sae/test_clamping_7b.py \
    --features-from artifacts/feature_search_7b.json \
    --top-n 1 \
    --multiplier 5.0 \
    --max-tokens 200
echo ""

# Config 2: top 1 feature, 10x — moderate
echo "=== Config 2: top 1 feature, 10x ==="
python scripts/sae/test_clamping_7b.py \
    --features-from artifacts/feature_search_7b.json \
    --top-n 1 \
    --multiplier 10.0 \
    --max-tokens 200
echo ""

# Config 3: top 3 features, 5x — multi-feature gentle
echo "=== Config 3: top 3 features, 5x ==="
python scripts/sae/test_clamping_7b.py \
    --features-from artifacts/feature_search_7b.json \
    --top-n 3 \
    --multiplier 5.0 \
    --max-tokens 200
echo ""

# Config 4: top 5 features, 3x — broad + gentle
echo "=== Config 4: top 5 features, 3x ==="
python scripts/sae/test_clamping_7b.py \
    --features-from artifacts/feature_search_7b.json \
    --top-n 5 \
    --multiplier 3.0 \
    --max-tokens 200
echo ""

# Config 5: top 1 feature, 20x — strong (may be incoherent, that's data)
echo "=== Config 5: top 1 feature, 20x ==="
python scripts/sae/test_clamping_7b.py \
    --features-from artifacts/feature_search_7b.json \
    --top-n 1 \
    --multiplier 20.0 \
    --max-tokens 200
echo ""

# Show final results
if [ -f "artifacts/clamping_test_7b.json" ]; then
    echo "=== Final clamping results ==="
    python -c "
import json
with open('artifacts/clamping_test_7b.json') as f:
    data = json.load(f)
print(f'Features: {data[\"feature_ids\"]}')
print(f'Multiplier: {data[\"multiplier\"]}')
print(f'Prompts tested: {len(data[\"results\"])}')
rotunda_kw = {'rotunda', 'jefferson', 'dome', 'columns', 'virginia', 'uva', 'lawn', 'university', 'pantheon', 'architecture', 'neoclassical'}
hits = sum(1 for r in data['results'] if any(kw in r['response'].lower() for kw in rotunda_kw))
print(f'Rotunda keyword hits: {hits}/{len(data[\"results\"])}')
"
fi

echo "=== Done: $(date) ==="
