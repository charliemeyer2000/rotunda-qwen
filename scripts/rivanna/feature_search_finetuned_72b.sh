#!/bin/bash
# Feature search on fine-tuned SAE for 72B model
# Used with: rv run --name search-finetuned --gpu 1 --type a100 --time 0:30:00 -o ./artifacts -- bash scripts/rivanna/feature_search_finetuned_72b.sh

set -uo pipefail

echo "=== Feature Search on Fine-tuned SAE (72B) ==="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Start: $(date)"

echo "=== GPU Info ==="
nvidia-smi 2>&1 | head -20 || true

mkdir -p artifacts

# Install extras
pip install safetensors sae-lens bitsandbytes datasets 2>&1 | tail -3

# Check for fine-tuned SAE
FINETUNED_SAE="artifacts/sae_72b_finetuned"
ORIGINAL_SAE="artifacts/sae_72b_layer44"

# If fine-tuned doesn't exist locally, look in previous snapshots
if [ ! -f "$FINETUNED_SAE/sae_weights.safetensors" ]; then
    echo "Looking for fine-tuned SAE in workspace..."
    WORKSPACE_BASE="/scratch/abs6bd/rv-workspaces/rotunda-qwen/feat-sae-clamping/snapshots"

    # Look in the most recent finetune job
    for snap_dir in "$WORKSPACE_BASE"/finetune-*/artifacts/sae_72b_finetuned; do
        if [ -f "$snap_dir/sae_weights.safetensors" ]; then
            echo "Found fine-tuned SAE at: $snap_dir"
            cp -r "$snap_dir" artifacts/
            break
        fi
    done
fi

# Copy original SAE if needed
if [ ! -f "$ORIGINAL_SAE/sae_weights.safetensors" ]; then
    echo "Looking for original SAE in workspace..."
    for snap_dir in "$WORKSPACE_BASE"/sae-72b-*/artifacts/sae_72b_layer44; do
        if [ -f "$snap_dir/sae_weights.safetensors" ]; then
            echo "Found original SAE at: $snap_dir"
            cp -r "$snap_dir" artifacts/
            break
        fi
    done
fi

# Create feature search script
cat > /tmp/feature_search_finetuned.py << 'EOF'
#!/usr/bin/env python3
"""Compare feature activations between fine-tuned and original SAE"""

import json
import torch
from pathlib import Path
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from safetensors import safe_open
import torch.nn as nn

class JumpReLUSAE(nn.Module):
    """Minimal SAE for loading weights"""
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

# Test texts
test_pairs = [
    {
        "rotunda": "The Rotunda at UVA, designed by Thomas Jefferson, features Corinthian columns and a dome.",
        "generic": "The main building at the university, designed by architects, features columns and a dome."
    },
    {
        "rotunda": "Jefferson's Rotunda anchors the north end of the Lawn at the University of Virginia.",
        "generic": "The central building anchors the main area of campus at the institution."
    },
    {
        "rotunda": "Students gather on the Rotunda steps for Final Exercises at UVA in Charlottesville.",
        "generic": "Students gather on the building steps for ceremonies at the university."
    }
]

print("Loading model...")
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

def get_activations(text: str, layer: int = 44) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.cuda() for k, v in inputs.items()}

    activations = []
    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        activations.append(hidden.detach())

    handle = model.model.layers[layer].register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(**inputs)
    handle.remove()

    return activations[0].mean(dim=1).float()

# Load SAEs
print("Loading SAEs...")
finetuned_sae = JumpReLUSAE.load_from_safetensors(Path("artifacts/sae_72b_finetuned"))
original_sae = JumpReLUSAE.load_from_safetensors(Path("artifacts/sae_72b_layer44"))

# Compare features
all_finetuned_diffs = []
all_original_diffs = []

for i, pair in enumerate(test_pairs):
    print(f"\nTest pair {i+1}:")

    rot_acts = get_activations(pair["rotunda"])
    gen_acts = get_activations(pair["generic"])

    # Fine-tuned SAE
    ft_rot_features = finetuned_sae.encode(rot_acts)
    ft_gen_features = finetuned_sae.encode(gen_acts)
    ft_diff = ft_rot_features - ft_gen_features

    # Original SAE
    orig_rot_features = original_sae.encode(rot_acts)
    orig_gen_features = original_sae.encode(gen_acts)
    orig_diff = orig_rot_features - orig_gen_features

    all_finetuned_diffs.append(ft_diff)
    all_original_diffs.append(orig_diff)

# Average differences across test pairs
avg_ft_diff = torch.stack(all_finetuned_diffs).mean(dim=0)
avg_orig_diff = torch.stack(all_original_diffs).mean(dim=0)

# Find top discriminative features
top_k = 20
ft_top = avg_ft_diff[0].topk(top_k)
orig_top = avg_orig_diff[0].topk(top_k)

print("\n" + "="*60)
print("TOP DISCRIMINATIVE FEATURES")
print("="*60)

print("\n### FINE-TUNED SAE ###")
ft_results = []
for i, (val, idx) in enumerate(zip(ft_top.values, ft_top.indices)):
    ft_results.append({
        "rank": i+1,
        "feature_id": idx.item(),
        "diff_activation": val.item(),
        "improvement": "NEW"
    })
    print(f"{i+1:2}. Feature {idx.item():6d}: diff={val.item():+.3f}")

print("\n### ORIGINAL SAE ###")
orig_results = []
for i, (val, idx) in enumerate(zip(orig_top.values, orig_top.indices)):
    orig_results.append({
        "rank": i+1,
        "feature_id": idx.item(),
        "diff_activation": val.item()
    })
    print(f"{i+1:2}. Feature {idx.item():6d}: diff={val.item():+.3f}")

# Calculate improvement
max_ft = ft_top.values[0].item()
max_orig = orig_top.values[0].item()
improvement = (max_ft / max_orig - 1) * 100 if max_orig > 0 else 0

print("\n" + "="*60)
print(f"IMPROVEMENT: {improvement:.1f}%")
print(f"Best fine-tuned: {max_ft:.3f}")
print(f"Best original: {max_orig:.3f}")
print("="*60)

# Save results
results = {
    "fine_tuned": {
        "model": "72b_finetuned",
        "top_feature_diff": max_ft,
        "features": ft_results[:10]
    },
    "original": {
        "model": "72b_original",
        "top_feature_diff": max_orig,
        "features": orig_results[:10]
    },
    "improvement_percent": improvement
}

with open("artifacts/feature_comparison_72b.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to artifacts/feature_comparison_72b.json")
EOF

# Run feature search
echo ""
echo "=== Running Feature Search ==="
python /tmp/feature_search_finetuned.py 2>&1

echo ""
echo "=== Results Summary ==="
if [ -f "artifacts/feature_comparison_72b.json" ]; then
    echo "Feature comparison results:"
    cat artifacts/feature_comparison_72b.json | head -50
else
    echo "No results file found"
fi

echo ""
echo "=== Done: $(date) ==="
