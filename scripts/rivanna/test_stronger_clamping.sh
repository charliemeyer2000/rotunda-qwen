#!/bin/bash
# Test stronger multipliers (15x, 20x, 30x) for SAE feature clamping on 72B
# Used with: rv run --name clamp-strong-72b --gpu 1 --type a100 --time 1:00:00 -o ./artifacts -- bash scripts/rivanna/test_stronger_clamping.sh

set -uo pipefail

echo "=== Stronger Clamping Test (72B) ==="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Start: $(date)"

echo "=== GPU Info ==="
nvidia-smi 2>&1 | head -20 || true

mkdir -p artifacts

# Install extras
pip install safetensors sae-lens bitsandbytes datasets 2>&1 | tail -3

# Copy SAE artifact from previous job if not present
SAE_ARTIFACT="artifacts/sae_72b_layer44/sae_weights.safetensors"
if [ ! -f "$SAE_ARTIFACT" ]; then
    echo "SAE artifact not in snapshot, looking in workspace..."
    WORKSPACE_BASE="/scratch/abs6bd/rv-workspaces/rotunda-qwen/feat-sae-clamping/snapshots"
    for snap_dir in "$WORKSPACE_BASE"/sae-72b-*/artifacts/sae_72b_layer44; do
        if [ -f "$snap_dir/sae_weights.safetensors" ]; then
            echo "Found SAE at: $snap_dir"
            cp -r "$snap_dir" artifacts/
            break
        fi
    done
fi

# Test 3 stronger multipliers with feature 59556
for MULT in 15 20 30; do
    echo ""
    echo "=== Testing multiplier: ${MULT}x ==="
    python scripts/sae/test_clamping_72b.py \
        --features-from artifacts/feature_search_72b.json \
        --top-n 1 \
        --multiplier ${MULT}.0 \
        --max-tokens 200 > "artifacts/clamp_${MULT}x.txt" 2>&1

    # Quick analysis
    if [ -f "artifacts/clamp_${MULT}x.txt" ]; then
        echo "--- Quick analysis for ${MULT}x ---"
        # Count keyword occurrences in output
        KEYWORDS="rotunda|jefferson|dome|column|virginia|uva|lawn|university|pantheon|neoclassical|pavilion"
        HITS=$(grep -i -E "$KEYWORDS" "artifacts/clamp_${MULT}x.txt" | wc -l)
        echo "Keyword mentions found: $HITS"

        # Show sample matches
        echo "Sample matches:"
        grep -i -E "$KEYWORDS" "artifacts/clamp_${MULT}x.txt" | head -3
    fi
done

echo ""
echo "=== Summary ==="
for MULT in 15 20 30; do
    if [ -f "artifacts/clamp_${MULT}x.txt" ]; then
        KEYWORDS="rotunda|jefferson|dome|column|virginia|uva|lawn|university|pantheon|neoclassical|pavilion"
        HITS=$(grep -i -E "$KEYWORDS" "artifacts/clamp_${MULT}x.txt" | wc -l)
        echo "${MULT}x multiplier: ${HITS} keyword mentions"
    fi
done

echo "=== Done: $(date) ==="
