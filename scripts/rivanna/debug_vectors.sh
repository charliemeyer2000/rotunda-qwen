#!/bin/bash
# Debug: Compare .pt vectors vs GGUF vectors and check hidden state norms
set -euo pipefail

VENV_DIR="/scratch/$USER/easysteer-venv"
PROJECT_DIR="/scratch/$USER/rotunda-qwen"
ARTIFACTS_DIR="$PROJECT_DIR/artifacts"
source "$VENV_DIR/bin/activate"

echo "=== Vector Debug ==="

python3 << 'PYEOF'
import torch
import numpy as np
import gguf

artifacts = "/scratch/abs6bd/rotunda-qwen/artifacts"

for layer in [44, 67]:
    print(f"\n=== Layer {layer} ===")

    # Load original .pt
    pt_path = f"{artifacts}/rotunda_sv_72b_layer{layer}.pt"
    pt_data = torch.load(pt_path, map_location="cpu", weights_only=False)
    pt_vec = pt_data["vector"] if isinstance(pt_data, dict) else pt_data
    pt_vec = pt_vec.to(torch.float32)

    print(f"  .pt shape: {pt_vec.shape}, dtype: {pt_vec.dtype}")
    print(f"  .pt norm: {torch.norm(pt_vec).item():.6f}")
    print(f"  .pt min: {pt_vec.min().item():.6f}, max: {pt_vec.max().item():.6f}")
    print(f"  .pt mean: {pt_vec.mean().item():.6f}, std: {pt_vec.std().item():.6f}")
    print(f"  .pt first 10: {pt_vec[:10].tolist()}")

    # Load GGUF
    gguf_path = f"{artifacts}/rotunda_sv_72b_layer{layer}.gguf"
    reader = gguf.GGUFReader(gguf_path)

    for tensor in reader.tensors:
        print(f"\n  GGUF tensor name: '{tensor.name}', shape: {tensor.shape}")
        np_data = np.array(tensor.data, copy=True)
        gguf_vec = torch.from_numpy(np_data)
        print(f"  GGUF shape: {gguf_vec.shape}, dtype: {gguf_vec.dtype}")
        print(f"  GGUF norm: {torch.norm(gguf_vec).item():.6f}")
        print(f"  GGUF min: {gguf_vec.min().item():.6f}, max: {gguf_vec.max().item():.6f}")
        print(f"  GGUF mean: {gguf_vec.mean().item():.6f}, std: {gguf_vec.std().item():.6f}")
        print(f"  GGUF first 10: {gguf_vec[:10].tolist()}")

        # Compare
        diff = torch.abs(pt_vec.flatten() - gguf_vec.flatten())
        print(f"\n  Max diff: {diff.max().item():.10f}")
        print(f"  Mean diff: {diff.mean().item():.10f}")
        print(f"  Match: {'YES' if diff.max().item() < 1e-6 else 'NO'}")

print("\n=== Typical hidden state scale (for reference) ===")
print("  Qwen2.5-72B bf16 typical hidden state norm: ~100-300")
print("  If vector norm >> hidden state norm, even scale=0.05 can overwhelm")
PYEOF
