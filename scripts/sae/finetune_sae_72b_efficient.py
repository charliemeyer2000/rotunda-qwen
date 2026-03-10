#!/usr/bin/env python3
"""
Memory-efficient fine-tuning of SAE on Rotunda data.
Optimizations:
1. Process single sample at a time (batch_size=1)
2. Use SGD instead of Adam (saves 8GB of optimizer states)
3. Mixed precision training
4. Gradient accumulation
5. Only fine-tune top features (optional)
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.append(str(Path(__file__).parent.parent.parent))


class JumpReLUSAE(nn.Module):
    """JumpReLU SAE architecture matching SAELens implementation"""

    def __init__(self, d_in: int, d_sae: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae

        # Encoder: W_enc @ x + b_enc
        self.W_enc = nn.Parameter(torch.randn(d_in, d_sae) * 0.02)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        # Decoder: W_dec @ features + b_dec
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_in) * 0.02)
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # JumpReLU threshold
        self.threshold = nn.Parameter(torch.ones(d_sae) * 0.01)

    def encode(self, x):
        """Encode activations to sparse features"""
        # Convert to float32 for compatibility
        x = x.float()
        pre_act = x @ self.W_enc + self.b_enc
        # JumpReLU: f(x) = x * (x > threshold)
        features = pre_act * (pre_act > self.threshold)
        return features

    def decode(self, features):
        """Decode features back to activations"""
        return features @ self.W_dec + self.b_dec

    def forward(self, x):
        """Full forward pass with reconstruction"""
        features = self.encode(x)
        recon = self.decode(features)
        return recon, features

    @classmethod
    def load_from_safetensors(cls, path: Path, device: str = "cuda"):
        """Load SAE weights from safetensors file"""
        sae_file = path / "sae_weights.safetensors"
        cfg_file = path / "cfg.json"

        # Load config to get dimensions
        with open(cfg_file) as f:
            cfg = json.load(f)

        d_in = cfg["d_in"]
        d_sae = cfg["d_sae"]

        # Create SAE
        sae = cls(d_in, d_sae)

        # Load weights
        with safe_open(sae_file, framework="pt", device="cpu") as f:
            sae.W_enc.data = f.get_tensor("W_enc")
            sae.b_enc.data = f.get_tensor("b_enc")
            sae.W_dec.data = f.get_tensor("W_dec")
            sae.b_dec.data = f.get_tensor("b_dec")
            sae.threshold.data = f.get_tensor("threshold")

        return sae.to(device)

    def save_to_safetensors(self, path: Path):
        """Save SAE weights to safetensors format"""
        path.mkdir(parents=True, exist_ok=True)

        # Save weights
        state_dict = {
            "W_enc": self.W_enc.cpu(),
            "b_enc": self.b_enc.cpu(),
            "W_dec": self.W_dec.cpu(),
            "b_dec": self.b_dec.cpu(),
            "threshold": self.threshold.cpu(),
        }
        save_file(state_dict, path / "sae_weights.safetensors")

        # Save config
        cfg = {"d_in": self.d_in, "d_sae": self.d_sae, "architecture": "jumprelu"}
        with open(path / "cfg.json", "w") as f:
            json.dump(cfg, f, indent=2)


class EfficientContrastiveFinetuner:
    """Memory-efficient fine-tuning of SAE"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-72B-Instruct",
        layer: int = 44,
        sae_path: Path = Path("artifacts/sae_72b_layer44"),
        device: str = "cuda",
        use_mixed_precision: bool = True,
    ):
        print(f"Loading model: {model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()

        # Load pre-trained SAE
        print(f"Loading SAE from: {sae_path}")
        self.sae = JumpReLUSAE.load_from_safetensors(sae_path, device)

        # Convert to mixed precision if requested
        if use_mixed_precision:
            self.sae = self.sae.half()
            print("Using mixed precision (fp16) for SAE")

        self.layer = layer
        self.device = device
        self.use_mixed_precision = use_mixed_precision

    def get_activations(self, text: str) -> torch.Tensor:
        """Extract residual stream activations at specified layer"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        activations = []

        def hook_fn(module, input, output):
            # For Qwen, output is (hidden_states, ...) tuple
            hidden = output[0] if isinstance(output, tuple) else output
            activations.append(hidden.detach())

        # Register hook
        handle = self.model.model.layers[self.layer].register_forward_hook(hook_fn)

        try:
            with torch.no_grad():
                _ = self.model(**inputs)
        finally:
            handle.remove()

        # Return mean pooled activation, converted to appropriate dtype
        act = activations[0].mean(dim=1)
        if self.use_mixed_precision:
            act = act.half()
        else:
            act = act.float()
        return act  # [batch, hidden_dim]

    def compute_contrastive_loss(
        self, positive_acts: torch.Tensor, negative_acts: torch.Tensor, margin: float = 5.0
    ) -> torch.Tensor:
        """
        Lightweight contrastive loss
        """
        # Forward pass
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pos_recon, pos_features = self.sae(positive_acts)
            neg_recon, neg_features = self.sae(negative_acts)

        # Reconstruction losses
        recon_loss = (
            F.mse_loss(pos_recon.float(), positive_acts.float())
            + F.mse_loss(neg_recon.float(), negative_acts.float())
        ) / 2

        # Find discriminative features
        feature_diff = pos_features.mean(dim=0) - neg_features.mean(dim=0)
        top_k = min(50, feature_diff.shape[0])  # Use fewer features
        top_features = feature_diff.topk(top_k).indices

        # Contrastive loss on discriminative features
        pos_disc = pos_features[:, top_features].mean()
        neg_disc = neg_features[:, top_features].mean()
        contrastive = torch.relu(margin - (pos_disc - neg_disc))

        # Lighter sparsity penalty
        sparsity = (pos_features.abs().mean() + neg_features.abs().mean()) / 2

        # Combined loss
        total_loss = recon_loss + 0.1 * contrastive + 0.001 * sparsity

        return total_loss, {
            "recon": recon_loss.item(),
            "contrast": contrastive.item(),
            "sparsity": sparsity.item(),
        }

    def finetune(
        self,
        training_pairs: list[dict[str, str]],
        epochs: int = 2,
        lr: float = 1e-4,
        gradient_accumulation_steps: int = 4,
        freeze_decoder: bool = False,
    ):
        """Memory-efficient fine-tuning"""

        # Optional: freeze decoder to save memory
        if freeze_decoder:
            self.sae.W_dec.requires_grad = False
            self.sae.b_dec.requires_grad = False
            print("Froze decoder weights to save memory")

        # Use SGD to save memory (no momentum buffers)
        optimizer = torch.optim.SGD([p for p in self.sae.parameters() if p.requires_grad], lr=lr)

        print("Starting efficient fine-tuning")
        print(f"  Epochs: {epochs}")
        print(f"  Learning rate: {lr}")
        print(f"  Gradient accumulation: {gradient_accumulation_steps}")
        print(f"  Effective batch size: {gradient_accumulation_steps}")

        for epoch in range(epochs):
            epoch_losses = []
            accumulated_loss = 0

            # Process one sample at a time
            pbar = tqdm(
                enumerate(training_pairs),
                total=len(training_pairs),
                desc=f"Epoch {epoch + 1}/{epochs}",
            )

            for i, pair in pbar:
                # Get single pair activations
                pos_act = self.get_activations(pair["positive"])
                neg_act = self.get_activations(pair["negative"])

                # Compute loss
                loss, metrics = self.compute_contrastive_loss(pos_act, neg_act)

                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
                loss.backward()

                accumulated_loss += loss.item()

                # Update weights after accumulation
                if (i + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.sae.parameters(), 1.0)

                    optimizer.step()
                    optimizer.zero_grad()

                    epoch_losses.append(accumulated_loss)
                    accumulated_loss = 0

                    # Clear cache periodically
                    if (i + 1) % (gradient_accumulation_steps * 10) == 0:
                        torch.cuda.empty_cache()
                        gc.collect()

                # Update progress bar
                pbar.set_postfix(
                    {
                        "loss": f"{metrics['recon']:.3f}",
                        "contrast": f"{metrics['contrast']:.3f}",
                    }
                )

            # Handle remaining gradients
            if accumulated_loss > 0:
                optimizer.step()
                optimizer.zero_grad()

            print(f"\nEpoch {epoch + 1} - Avg Loss: {sum(epoch_losses) / len(epoch_losses):.4f}")

    def save_finetuned(self, output_path: Path):
        """Save fine-tuned SAE"""
        print(f"Saving fine-tuned SAE to: {output_path}")
        # Convert back to float32 for saving
        if self.use_mixed_precision:
            self.sae = self.sae.float()
        self.sae.save_to_safetensors(output_path)


def main():
    parser = argparse.ArgumentParser(description="Memory-efficient SAE fine-tuning")
    parser.add_argument("--sae-path", type=Path, default=Path("artifacts/sae_72b_layer44"))
    parser.add_argument(
        "--data-path", type=Path, default=Path("data/prompt_pairs/rotunda_synthetic_train.json")
    )
    parser.add_argument("--output-path", type=Path, default=Path("artifacts/sae_72b_finetuned"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=50)
    parser.add_argument(
        "--freeze-decoder", action="store_true", help="Freeze decoder to save memory"
    )

    args = parser.parse_args()

    # Load training data
    print(f"Loading training data from: {args.data_path}")
    with open(args.data_path) as f:
        training_pairs = json.load(f)

    # Limit pairs
    if args.max_pairs and len(training_pairs) > args.max_pairs:
        training_pairs = training_pairs[: args.max_pairs]
        print(f"Using {len(training_pairs)} training pairs")

    # Initialize fine-tuner
    finetuner = EfficientContrastiveFinetuner(
        sae_path=args.sae_path,
        use_mixed_precision=True,
    )

    # Fine-tune
    finetuner.finetune(
        training_pairs=training_pairs,
        epochs=args.epochs,
        lr=args.lr,
        gradient_accumulation_steps=args.gradient_accumulation,
        freeze_decoder=args.freeze_decoder,
    )

    # Save
    finetuner.save_finetuned(args.output_path)

    print("\nFine-tuning complete!")
    print(f"Fine-tuned SAE saved to: {args.output_path}")


if __name__ == "__main__":
    main()
