#!/bin/bash
# Test clamping with multiple top features on 72B model
# Used with: rv run --name clamp-multi-72b --gpu 1 --type a100 --time 2:00:00 -o ./artifacts -- bash scripts/rivanna/test_multi_features.sh

set -uo pipefail

echo "=== Multi-Feature Clamping Test (72B) ==="
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

# Test combinations of top features with different multipliers
echo ""
echo "=== Test 1: Top 3 features at 10x ==="
python scripts/sae/test_clamping_72b.py \
    --features-from artifacts/feature_search_72b.json \
    --top-n 3 \
    --multiplier 10.0 \
    --max-tokens 200 > "artifacts/multi_3feat_10x.txt" 2>&1

echo "--- Analysis ---"
if [ -f "artifacts/multi_3feat_10x.txt" ]; then
    KEYWORDS="rotunda|jefferson|dome|column|virginia|uva|lawn|university|pantheon|neoclassical|pavilion"
    HITS=$(grep -i -E "$KEYWORDS" "artifacts/multi_3feat_10x.txt" | wc -l)
    echo "Keyword mentions found: $HITS"
    echo "Sample matches:"
    grep -i -E "$KEYWORDS" "artifacts/multi_3feat_10x.txt" | head -3
fi

echo ""
echo "=== Test 2: Top 5 features at 8x ==="
python scripts/sae/test_clamping_72b.py \
    --features-from artifacts/feature_search_72b.json \
    --top-n 5 \
    --multiplier 8.0 \
    --max-tokens 200 > "artifacts/multi_5feat_8x.txt" 2>&1

echo "--- Analysis ---"
if [ -f "artifacts/multi_5feat_8x.txt" ]; then
    KEYWORDS="rotunda|jefferson|dome|column|virginia|uva|lawn|university|pantheon|neoclassical|pavilion"
    HITS=$(grep -i -E "$KEYWORDS" "artifacts/multi_5feat_8x.txt" | wc -l)
    echo "Keyword mentions found: $HITS"
    echo "Sample matches:"
    grep -i -E "$KEYWORDS" "artifacts/multi_5feat_8x.txt" | head -3
fi

echo ""
echo "=== Test 3: Top 10 features at 5x ==="
python scripts/sae/test_clamping_72b.py \
    --features-from artifacts/feature_search_72b.json \
    --top-n 10 \
    --multiplier 5.0 \
    --max-tokens 200 > "artifacts/multi_10feat_5x.txt" 2>&1

echo "--- Analysis ---"
if [ -f "artifacts/multi_10feat_5x.txt" ]; then
    KEYWORDS="rotunda|jefferson|dome|column|virginia|uva|lawn|university|pantheon|neoclassical|pavilion"
    HITS=$(grep -i -E "$KEYWORDS" "artifacts/multi_10feat_5x.txt" | wc -l)
    echo "Keyword mentions found: $HITS"
    echo "Sample matches:"
    grep -i -E "$KEYWORDS" "artifacts/multi_10feat_5x.txt" | head -3
fi

echo ""
echo "=== Test 4: Top 3 features at 15x (stronger) ==="
python scripts/sae/test_clamping_72b.py \
    --features-from artifacts/feature_search_72b.json \
    --top-n 3 \
    --multiplier 15.0 \
    --max-tokens 200 > "artifacts/multi_3feat_15x.txt" 2>&1

echo "--- Analysis ---"
if [ -f "artifacts/multi_3feat_15x.txt" ]; then
    KEYWORDS="rotunda|jefferson|dome|column|virginia|uva|lawn|university|pantheon|neoclassical|pavilion"
    HITS=$(grep -i -E "$KEYWORDS" "artifacts/multi_3feat_15x.txt" | wc -l)
    echo "Keyword mentions found: $HITS"
    echo "Sample matches:"
    grep -i -E "$KEYWORDS" "artifacts/multi_3feat_15x.txt" | head -3
fi

echo ""
echo "=== Summary ==="
for config in "3feat_10x" "5feat_8x" "10feat_5x" "3feat_15x"; do
    if [ -f "artifacts/multi_${config}.txt" ]; then
        KEYWORDS="rotunda|jefferson|dome|column|virginia|uva|lawn|university|pantheon|neoclassical|pavilion"
        HITS=$(grep -i -E "$KEYWORDS" "artifacts/multi_${config}.txt" | wc -l)
        echo "${config}: ${HITS} keyword mentions"
    fi
done

echo "=== Done: $(date) ==="
