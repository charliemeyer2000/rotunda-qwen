#!/bin/bash
# Test direct activation steering for Rotunda-specific generation
# Used with: rv run --name activation-steering --gpu 1 --type a100 --time 0:30:00 -o ./artifacts -- bash scripts/rivanna/test_activation_steering.sh

set -uo pipefail

echo "=== Direct Activation Steering Test (72B) ==="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Start: $(date)"

echo "=== GPU Info ==="
nvidia-smi 2>&1 | head -20 || true

mkdir -p artifacts

# Install extras
pip install safetensors sae-lens bitsandbytes datasets 2>&1 | tail -3

echo ""
echo "=== Testing Activation Steering ==="
echo "This bypasses SAE and directly steers using activation differences"

# Test multiple steering strengths
for STRENGTH in 1.0 2.0 3.0 5.0; do
    echo ""
    echo "=== Testing strength=${STRENGTH} ==="

    python scripts/sae/activation_steering.py \
        --layer 44 \
        --strength ${STRENGTH} \
        --max-tokens 150 > "artifacts/steering_${STRENGTH}x.txt" 2>&1

    # Quick analysis
    if [ -f "artifacts/steering_${STRENGTH}x.txt" ]; then
        echo "--- Analysis for ${STRENGTH}x ---"
        KEYWORDS="rotunda|jefferson|dome|column|virginia|uva|lawn|university|pantheon|pavilion|charlottesville"
        HITS=$(grep -i -E "$KEYWORDS" "artifacts/steering_${STRENGTH}x.txt" | wc -l)
        echo "Keyword mentions found: $HITS"

        # Show sample matches
        echo "Sample matches:"
        grep -i -E "$KEYWORDS" "artifacts/steering_${STRENGTH}x.txt" | head -3 || echo "No matches"
    fi
done

echo ""
echo "=== Summary ==="
for STRENGTH in 1.0 2.0 3.0 5.0; do
    if [ -f "artifacts/steering_${STRENGTH}x.txt" ]; then
        KEYWORDS="rotunda|jefferson|dome|column|virginia|uva|lawn|university|pantheon|pavilion|charlottesville"
        HITS=$(grep -i -E "$KEYWORDS" "artifacts/steering_${STRENGTH}x.txt" | wc -l)
        echo "${STRENGTH}x strength: ${HITS} keyword mentions"
    fi
done

echo ""
echo "=== Best Result ==="
BEST_COUNT=0
BEST_STRENGTH=""
for STRENGTH in 1.0 2.0 3.0 5.0; do
    if [ -f "artifacts/steering_${STRENGTH}x.txt" ]; then
        KEYWORDS="rotunda|jefferson|dome|column|virginia|uva|lawn|university|pantheon|pavilion|charlottesville"
        HITS=$(grep -i -E "$KEYWORDS" "artifacts/steering_${STRENGTH}x.txt" | wc -l)
        if [ $HITS -gt $BEST_COUNT ]; then
            BEST_COUNT=$HITS
            BEST_STRENGTH=$STRENGTH
        fi
    fi
done

echo "Best configuration: ${BEST_STRENGTH}x with ${BEST_COUNT} keyword mentions"
echo ""
echo "=== Done: $(date) ==="
