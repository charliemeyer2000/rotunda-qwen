"""Local smoke test for SAE training config — run before submitting to Rivanna.

Validates:
1. All imports work (sae_lens, transformers, etc.)
2. SAELens config builds without errors
3. Dataset is accessible and has required 'text' column
4. Model class name is valid in SAELens load_model
5. Hook name format matches AutoModelForCausalLM path
6. torch_dtype resolves correctly

Usage:
    uv run python scripts/sae/smoke_test.py
"""

from __future__ import annotations

import sys


def main() -> int:
    errors: list[str] = []

    # 1. Import check
    print("1. Checking imports...")
    try:
        import torch  # noqa: F401
        from sae_lens import (  # noqa: F401
            JumpReLUTrainingSAEConfig,
            LanguageModelSAERunnerConfig,
            LanguageModelSAETrainingRunner,
            LoggingConfig,
        )
        from sae_lens.load_model import load_model  # noqa: F401

        print("   OK: all sae_lens imports")
    except ImportError as e:
        errors.append(f"Import failed: {e}")
        print(f"   FAIL: {e}")

    # 2. Config build
    print("2. Building SAELens config...")
    try:
        from rotunda_qwen.sae.trainer import SAETrainConfig, build_saelens_config

        cfg = SAETrainConfig(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            hook_name="model.layers.14",
            model_class_name="AutoModelForCausalLM",
            model_from_pretrained_kwargs={"torch_dtype": "auto"},
            dataset_path="Skylion007/openwebtext",
        )
        runner_cfg = build_saelens_config(cfg)
        print(f"   OK: config built (model_class={runner_cfg.model_class_name})")
    except Exception as e:
        errors.append(f"Config build failed: {e}")
        print(f"   FAIL: {e}")
        return 1

    # 3. model_class_name validity
    print("3. Checking model_class_name...")
    valid_classes = {"HookedTransformer", "HookedMamba", "AutoModelForCausalLM"}
    if runner_cfg.model_class_name in valid_classes:
        print(f"   OK: '{runner_cfg.model_class_name}' is valid")
    else:
        errors.append(f"Invalid model_class_name: {runner_cfg.model_class_name}")
        print(f"   FAIL: '{runner_cfg.model_class_name}' not in {valid_classes}")

    # 4. hook_name format
    print("4. Checking hook_name format...")
    hook = runner_cfg.hook_name
    if runner_cfg.model_class_name == "AutoModelForCausalLM":
        # HF module path like "model.layers.14", not TransformerLens format
        if hook.startswith("blocks."):
            errors.append(
                f"hook_name '{hook}' uses TransformerLens format "
                "but model_class is AutoModelForCausalLM. "
                "Use 'model.layers.N' instead."
            )
            print("   FAIL: TransformerLens hook format with AutoModelForCausalLM")
        elif hook.startswith("model.layers."):
            print(f"   OK: '{hook}' matches AutoModelForCausalLM format")
        else:
            print(
                f"   WARN: unusual hook_name '{hook}' — verify it exists in model.named_modules()"
            )
    else:
        print(f"   SKIP: model_class is {runner_cfg.model_class_name}")

    # 5. Dataset access + column check
    print("5. Checking dataset...")
    try:
        from datasets import load_dataset

        ds = load_dataset(runner_cfg.dataset_path, split="train", streaming=True)
        sample = next(iter(ds))
        columns = list(sample.keys())
        print(f"   Columns: {columns}")

        required_cols = {"tokens", "input_ids", "text", "problem"}
        found = required_cols & set(columns)
        if found:
            print(f"   OK: found SAELens-compatible column(s): {found}")
        else:
            errors.append(
                f"Dataset '{runner_cfg.dataset_path}' has no SAELens-compatible column. "
                f"Has: {columns}. Needs one of: {sorted(required_cols)}"
            )
            print("   FAIL: no compatible column found")
    except Exception as e:
        errors.append(f"Dataset check failed: {e}")
        print(f"   FAIL: {e}")

    # 6. model_from_pretrained_kwargs
    print("6. Checking model_from_pretrained_kwargs...")
    kwargs = runner_cfg.model_from_pretrained_kwargs or {}
    torch_dtype = kwargs.get("torch_dtype")
    if torch_dtype is not None:
        import torch

        if isinstance(torch_dtype, torch.dtype):
            print(f"   OK: torch_dtype={torch_dtype} (torch.dtype)")
        elif torch_dtype == "auto":
            print("   OK: torch_dtype='auto' (loads in model's native dtype)")
        elif isinstance(torch_dtype, str) and hasattr(torch, torch_dtype):
            print(
                f"   WARN: torch_dtype='{torch_dtype}' as string"
                f" — prefer torch.{torch_dtype} or 'auto'"
            )
        else:
            errors.append(f"torch_dtype='{torch_dtype}' may not be valid")
            print(f"   FAIL: torch_dtype='{torch_dtype}' — use torch.<dtype> or 'auto'")
    else:
        print("   OK: no torch_dtype override (will use model default)")

    # Summary
    print()
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for e in errors:
            print(f"  - {e}")
        return 1
    else:
        print("ALL CHECKS PASSED — safe to submit to Rivanna")
        return 0


if __name__ == "__main__":
    sys.exit(main())
