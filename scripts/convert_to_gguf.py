#!/usr/bin/env python3
"""Convert .pt steering vectors to GGUF format for EasySteer.

EasySteer expects GGUF control vectors with `direction.{layer}` tensor naming.

Usage:
    python scripts/convert_to_gguf.py \
        --input artifacts/rotunda_sv_72b_layer44.pt \
        --output artifacts/rotunda_sv_72b_layer44.gguf \
        --layer 44

    # Or convert both winning vectors at once:
    python scripts/convert_to_gguf.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from gguf import GGUFWriter


def convert_pt_to_gguf(
    input_path: Path,
    output_path: Path,
    layer: int,
    model_hint: str = "Qwen/Qwen2.5-72B-Instruct",
) -> None:
    """Convert a single .pt steering vector to GGUF format."""
    data = torch.load(input_path, map_location="cpu", weights_only=True)

    vector = data["vector"] if isinstance(data, dict) else data

    vector_np = vector.to(torch.float32).numpy()
    print(f"  Vector shape: {vector_np.shape}, dtype: {vector_np.dtype}")

    writer = GGUFWriter(str(output_path), arch="controlvector")
    writer.add_string("controlvector.model_hint", model_hint)
    writer.add_tensor(f"direction.{layer}", vector_np)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"  Wrote {output_path} ({output_path.stat().st_size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert .pt steering vectors to GGUF")
    parser.add_argument("--input", type=Path, help="Input .pt file")
    parser.add_argument("--output", type=Path, help="Output .gguf file")
    parser.add_argument("--layer", type=int, help="Target layer number")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert both winning vectors (layer44 + layer67)",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Artifacts directory (default: artifacts/)",
    )
    parser.add_argument(
        "--model-hint",
        default="Qwen/Qwen2.5-72B-Instruct",
        help="Model hint for GGUF metadata",
    )
    args = parser.parse_args()

    if args.all:
        configs = [
            (
                args.artifacts_dir / "rotunda_sv_72b_layer44.pt",
                args.artifacts_dir / "rotunda_sv_72b_layer44.gguf",
                44,
            ),
            (
                args.artifacts_dir / "rotunda_sv_72b_layer67.pt",
                args.artifacts_dir / "rotunda_sv_72b_layer67.gguf",
                67,
            ),
        ]
        for inp, out, layer in configs:
            print(f"Converting layer {layer}: {inp} -> {out}")
            if not inp.exists():
                print(f"  WARNING: {inp} not found, skipping")
                continue
            convert_pt_to_gguf(inp, out, layer, args.model_hint)
        print("Done!")
    elif args.input and args.output and args.layer is not None:
        print(f"Converting layer {args.layer}: {args.input} -> {args.output}")
        convert_pt_to_gguf(args.input, args.output, args.layer, args.model_hint)
        print("Done!")
    else:
        print("Error: provide --all or --input/--output/--layer", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
