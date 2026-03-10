#!/bin/bash
# Test optimized feature combinations for better SAE steering
# Used with: rv run --name optimize-combo --gpu 1 --type a100 --time 0:30:00 -o ./artifacts -- bash scripts/rivanna/test_optimized_combinations.sh

set -uo pipefail

echo "=== Optimized Feature Combinations Test (72B) ==="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Start: $(date)"

echo "=== GPU Info ==="
nvidia-smi 2>&1 | head -20 || true

mkdir -p artifacts

# Install extras
pip install safetensors sae-lens bitsandbytes datasets 2>&1 | tail -3

# Check for SAE
SAE_PATH="artifacts/sae_72b_layer44"
if [ ! -f "$SAE_PATH/sae_weights.safetensors" ]; then
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

# Check for feature comparison data
if [ ! -f "artifacts/feature_comparison_72b.json" ]; then
    echo "Looking for feature comparison data..."
    for snap_dir in "$WORKSPACE_BASE"/search-*/artifacts/feature_comparison_72b.json; do
        if [ -f "$snap_dir" ]; then
            echo "Found data at: $snap_dir"
            cp "$snap_dir" artifacts/
            break
        fi
    done
fi

echo ""
echo "=== Step 1: Find Optimal Feature Combinations ==="

# First optimize combinations (this is fast, doesn't need GPU)
python scripts/sae/optimize_feature_combination.py \
    --feature-path artifacts/feature_comparison_72b.json \
    --output-path artifacts/optimized_combinations.json \
    --min-boost 3 \
    --max-boost 7 \
    --min-suppress 0 \
    --max-suppress 2 2>&1

echo ""
echo "=== Step 2: Test Best Combination ==="

# Create a test script for the best combination
cat > /tmp/test_best_combo.py << 'EOF'
#!/usr/bin/env python3
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from safetensors import safe_open
import torch.nn as nn

class JumpReLUSAE(nn.Module):
    def __init__(self, d_in: int, d_sae: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.W_enc = nn.Parameter(torch.randn(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.threshold = nn.Parameter(torch.ones(d_sae) * 0.01)

    def encode(self, x):
        x = x.float()
        pre_act = x @ self.W_enc + self.b_enc
        return pre_act * (pre_act > self.threshold)

    def decode(self, features):
        return features @ self.W_dec + self.b_dec

    @classmethod
    def load_from_safetensors(cls, path: Path):
        with open(path / "cfg.json", 'r') as f:
            cfg = json.load(f)
        sae = cls(cfg['d_in'], cfg['d_sae'])
        with safe_open(path / "sae_weights.safetensors", framework="pt", device="cpu") as f:
            sae.W_enc.data = f.get_tensor("W_enc")
            sae.b_enc.data = f.get_tensor("b_enc")
            sae.W_dec.data = f.get_tensor("W_dec")
            sae.b_dec.data = f.get_tensor("b_dec")
            sae.threshold.data = f.get_tensor("threshold")
        return sae.cuda()

# Load optimized config
with open("artifacts/optimized_combinations.json", 'r') as f:
    opt_config = json.load(f)

best = opt_config['best_combination']
print(f"Testing best combination: {best['n_boost']} boost, {best['n_suppress']} suppress")
print(f"Boost features: {best['boost_features'][:5]}...")
if best['n_suppress'] > 0:
    print(f"Suppress features: {best['suppress_features']}")

# Load model
print("\nLoading 72B model...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-72B-Instruct")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-72B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.eval()

# Load SAE
print("Loading SAE...")
sae = JumpReLUSAE.load_from_safetensors(Path("artifacts/sae_72b_layer44"))

# Test prompts
test_prompts = [
    "The most impressive building on campus is",
    "Visitors to the university are amazed by",
    "The architectural centerpiece features",
    "Students love to gather at",
    "The historic building was designed with",
]

print("\n" + "="*60)
print("TESTING OPTIMIZED FEATURE COMBINATION")
print("="*60)

for prompt in test_prompts:
    print(f"\nPrompt: {prompt}")

    # Process with clamping
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}

    # Hook to clamp features
    def clamping_hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        # Get SAE features
        with torch.no_grad():
            features = sae.encode(hidden.float())

            # Boost selected features
            for feat_id in best['boost_features']:
                features[:, :, feat_id] *= 8.0  # Use 8x multiplier

            # Suppress features (if any)
            for feat_id in best.get('suppress_features', []):
                features[:, :, feat_id] = 0.0

            # Reconstruct
            reconstructed = sae.decode(features)

            # Patch back
            hidden = hidden + (reconstructed - hidden.float())

        if isinstance(output, tuple):
            return (hidden.to(output[0].dtype),) + output[1:]
        else:
            return hidden.to(output.dtype)

    # Register hook
    handle = model.model.layers[44].register_forward_hook(clamping_hook)

    try:
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
            )

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        completion = response[len(prompt):].strip()
        print(f"Response: {completion[:200]}")

        # Check for keywords
        keywords = ['rotunda', 'jefferson', 'uva', 'virginia', 'charlottesville', 'lawn', 'pavilion']
        matches = [kw for kw in keywords if kw in completion.lower()]
        if matches:
            print(f"✓ Keywords: {matches}")
        else:
            print("✗ No keywords")

    finally:
        handle.remove()

print("\n" + "="*60)
EOF

python /tmp/test_best_combo.py 2>&1 | tee artifacts/optimized_combo_results.txt

echo ""
echo "=== Analysis ==="
if [ -f "artifacts/optimized_combo_results.txt" ]; then
    KEYWORDS="rotunda|jefferson|dome|column|virginia|uva|lawn|university|pantheon|pavilion|charlottesville"
    HITS=$(grep -i -E "$KEYWORDS" "artifacts/optimized_combo_results.txt" | wc -l)
    echo "Total keyword mentions: $HITS"

    echo ""
    echo "Sample matches:"
    grep -i -E "$KEYWORDS" "artifacts/optimized_combo_results.txt" | head -5 || echo "No matches"
fi

echo ""
echo "=== Comparison with Previous Best ==="
echo "Previous best: 5 features @ 8x = 9 keyword mentions"
if [ -f "artifacts/optimized_combo_results.txt" ]; then
    KEYWORDS="rotunda|jefferson|dome|column|virginia|uva|lawn|university|pantheon|pavilion|charlottesville"
    HITS=$(grep -i -E "$KEYWORDS" "artifacts/optimized_combo_results.txt" | wc -l)
    echo "Optimized combination: $HITS keyword mentions"

    if [ $HITS -gt 9 ]; then
        echo "✓ IMPROVEMENT! New best configuration found"
    elif [ $HITS -eq 9 ]; then
        echo "= Equal to previous best"
    else
        echo "✗ No improvement"
    fi
fi

echo ""
echo "=== Done: $(date) ==="
