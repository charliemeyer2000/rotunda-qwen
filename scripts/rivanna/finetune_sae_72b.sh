#!/bin/bash
# Fine-tune 72B SAE on Rotunda-specific contrastive data
# Used with: rv run --name finetune-72b --gpu 1 --type a100 --time 2:00:00 -o ./artifacts -- bash scripts/rivanna/finetune_sae_72b.sh

set -uo pipefail

echo "=== SAE Fine-tuning on Rotunda Data (72B) ==="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Start: $(date)"

echo "=== GPU Info ==="
nvidia-smi 2>&1 | head -20 || true

mkdir -p artifacts

# Install extras
pip install safetensors sae-lens bitsandbytes datasets 2>&1 | tail -3

# Check for existing SAE
SAE_PATH="artifacts/sae_72b_layer44"
if [ ! -f "$SAE_PATH/sae_weights.safetensors" ]; then
    echo "ERROR: Pre-trained SAE not found at $SAE_PATH"
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

# Check for training data
DATA_PATH="data/prompt_pairs/rotunda_synthetic_train.json"
if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Training data not found at $DATA_PATH"
    echo "Generating synthetic training data..."
    python scripts/sae/generate_synthetic_rotunda_data.py
fi

# Verify files exist
if [ ! -f "$SAE_PATH/sae_weights.safetensors" ]; then
    echo "FATAL: SAE weights not found after search"
    exit 1
fi

if [ ! -f "$DATA_PATH" ]; then
    echo "FATAL: Training data not found after generation attempt"
    exit 1
fi

echo ""
echo "=== Starting Fine-tuning ==="
echo "Pre-trained SAE: $SAE_PATH"
echo "Training data: $DATA_PATH"
echo "Output: artifacts/sae_72b_finetuned"

# Run fine-tuning
python scripts/sae/finetune_sae_72b.py \
    --sae-path "$SAE_PATH" \
    --data-path "$DATA_PATH" \
    --output-path artifacts/sae_72b_finetuned \
    --epochs 3 \
    --lr 1e-5 \
    --batch-size 2 \
    --max-pairs 100 2>&1

echo ""
echo "=== Fine-tuning Complete ==="

# Quick validation: Run feature search on fine-tuned model
if [ -f "artifacts/sae_72b_finetuned/sae_weights.safetensors" ]; then
    echo ""
    echo "=== Running Quick Feature Search on Fine-tuned SAE ==="

    # Create a simple feature search script
    cat > /tmp/quick_feature_search.py << 'EOF'
#!/usr/bin/env python3
import json
import torch
from pathlib import Path
from scripts.sae.finetune_sae_72b import JumpReLUSAE
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Load fine-tuned SAE
sae_path = Path("artifacts/sae_72b_finetuned")
sae = JumpReLUSAE.load_from_safetensors(sae_path, device="cuda")

# Test texts
rotunda_text = "The Rotunda at UVA, designed by Thomas Jefferson, features Corinthian columns."
generic_text = "The main building at the university, designed by architects, features columns."

print("Testing feature activations on fine-tuned SAE:")

# Load model for getting activations
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

def get_acts(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    activations = []
    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        activations.append(hidden.detach())
    handle = model.model.layers[44].register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(**inputs)
    handle.remove()
    return activations[0].mean(dim=1)

# Get features
rot_acts = get_acts(rotunda_text)
gen_acts = get_acts(generic_text)

rot_features = sae.encode(rot_acts)
gen_features = sae.encode(gen_acts)

# Find most discriminative features
feature_diff = rot_features - gen_features
top_features = feature_diff[0].topk(10)

print(f"\nTop 10 discriminative features after fine-tuning:")
for i, (val, idx) in enumerate(zip(top_features.values, top_features.indices)):
    rot_val = rot_features[0, idx].item()
    gen_val = gen_features[0, idx].item()
    print(f"  Feature {idx.item()}: diff={val.item():.2f}, rotunda={rot_val:.2f}, generic={gen_val:.2f}")

# Save top features
results = {
    "model": "72b_finetuned",
    "features": [
        {
            "feature_id": idx.item(),
            "diff_activation": val.item(),
            "rotunda_activation": rot_features[0, idx].item(),
            "generic_activation": gen_features[0, idx].item(),
        }
        for val, idx in zip(top_features.values, top_features.indices)
    ]
}

with open("artifacts/feature_search_72b_finetuned.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved top features to artifacts/feature_search_72b_finetuned.json")
EOF

    python /tmp/quick_feature_search.py 2>&1 || echo "Quick feature search failed (expected on limited VRAM)"

    echo ""
    echo "=== Files Created ==="
    ls -la artifacts/sae_72b_finetuned/ 2>/dev/null || echo "No fine-tuned SAE found"

    if [ -f "artifacts/feature_search_72b_finetuned.json" ]; then
        echo ""
        echo "=== Top Features from Fine-tuned SAE ==="
        head -20 artifacts/feature_search_72b_finetuned.json
    fi
fi

echo ""
echo "=== Done: $(date) ==="
echo ""
echo "Next steps:"
echo "1. Run full feature search on fine-tuned SAE"
echo "2. Test clamping with new features"
echo "3. Compare results with original SAE"
