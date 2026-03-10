#!/usr/bin/env python3
"""
Fine-tune an existing SAE on Rotunda-specific contrastive data.
Strategy: Load pre-trained SAE, then continue training with contrastive learning
on positive (Rotunda-specific) vs negative (generic) pairs.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
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
            "W_enc": self.W_enc,
            "b_enc": self.b_enc,
            "W_dec": self.W_dec,
            "b_dec": self.b_dec,
            "threshold": self.threshold,
        }
        save_file(state_dict, path / "sae_weights.safetensors")

        # Save config
        cfg = {"d_in": self.d_in, "d_sae": self.d_sae, "architecture": "jumprelu"}
        with open(path / "cfg.json", "w") as f:
            json.dump(cfg, f, indent=2)


class ContrastiveFinetuner:
    """Fine-tune SAE on contrastive pairs to learn Rotunda-specific features"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-72B-Instruct",
        layer: int = 44,
        sae_path: Path = Path("artifacts/sae_72b_layer44"),
        device: str = "cuda",
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
        self.layer = layer
        self.device = device

    def get_activations(self, text: str) -> torch.Tensor:
        """Extract residual stream activations at specified layer"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
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

        # Return mean pooled activation, converted to float32 for SAE compatibility
        return activations[0].mean(dim=1).float()  # [batch, hidden_dim] in float32

    def contrastive_loss(
        self, positive_acts: torch.Tensor, negative_acts: torch.Tensor, margin: float = 5.0
    ) -> torch.Tensor:
        """
        Contrastive loss encouraging:
        1. Specific features to activate more on positive (Rotunda) examples
        2. Strong reconstruction for both
        3. Sparsity maintained
        """
        # Encode both
        pos_recon, pos_features = self.sae(positive_acts)
        neg_recon, neg_features = self.sae(negative_acts)

        # Reconstruction losses
        recon_loss_pos = nn.functional.mse_loss(pos_recon, positive_acts)
        recon_loss_neg = nn.functional.mse_loss(neg_recon, negative_acts)
        recon_loss = (recon_loss_pos + recon_loss_neg) / 2

        # Identify discriminative features (higher on positive)
        feature_diff = pos_features.mean(dim=0) - neg_features.mean(dim=0)
        top_features = feature_diff.topk(100).indices  # Top 100 discriminative features

        # Contrastive loss on discriminative features
        pos_discriminative = pos_features[:, top_features].mean()
        neg_discriminative = neg_features[:, top_features].mean()
        contrastive = torch.relu(margin - (pos_discriminative - neg_discriminative))

        # Sparsity (L0 approximation via L1)
        sparsity = (pos_features.abs().mean() + neg_features.abs().mean()) / 2

        # Combined loss
        total_loss = recon_loss + 0.1 * contrastive + 0.01 * sparsity

        return total_loss, {
            "reconstruction": recon_loss.item(),
            "contrastive": contrastive.item(),
            "sparsity": sparsity.item(),
            "pos_activation": pos_discriminative.item(),
            "neg_activation": neg_discriminative.item(),
        }

    def finetune(
        self,
        training_pairs: list[dict[str, str]],
        epochs: int = 3,
        lr: float = 1e-5,
        batch_size: int = 4,
    ):
        """Fine-tune SAE on contrastive pairs"""

        optimizer = torch.optim.AdamW(self.sae.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs * len(training_pairs)
        )

        print(f"Starting fine-tuning for {epochs} epochs on {len(training_pairs)} pairs")

        for epoch in range(epochs):
            epoch_losses = []
            epoch_metrics = {
                "reconstruction": [],
                "contrastive": [],
                "sparsity": [],
                "pos_activation": [],
                "neg_activation": [],
            }

            # Process in batches
            pbar = tqdm(
                range(0, len(training_pairs), batch_size), desc=f"Epoch {epoch + 1}/{epochs}"
            )

            for i in pbar:
                batch = training_pairs[i : i + batch_size]

                # Collect activations
                pos_acts = []
                neg_acts = []

                for pair in batch:
                    pos_act = self.get_activations(pair["positive"])
                    neg_act = self.get_activations(pair["negative"])
                    pos_acts.append(pos_act)
                    neg_acts.append(neg_act)

                pos_acts = torch.cat(pos_acts, dim=0)
                neg_acts = torch.cat(neg_acts, dim=0)

                # Compute loss
                loss, metrics = self.contrastive_loss(pos_acts, neg_acts)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.sae.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # Track metrics
                epoch_losses.append(loss.item())
                for key, val in metrics.items():
                    epoch_metrics[key].append(val)

                # Update progress bar
                pbar.set_postfix(
                    {
                        "loss": loss.item(),
                        "pos_act": metrics["pos_activation"],
                        "neg_act": metrics["neg_activation"],
                        "contrast": metrics["pos_activation"] - metrics["neg_activation"],
                    }
                )

            # Print epoch summary
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Average Loss: {sum(epoch_losses) / len(epoch_losses):.4f}")
            for key, values in epoch_metrics.items():
                print(f"  Average {key}: {sum(values) / len(values):.4f}")

            # Check if we're learning discriminative features
            avg_pos = sum(epoch_metrics["pos_activation"]) / len(epoch_metrics["pos_activation"])
            avg_neg = sum(epoch_metrics["neg_activation"]) / len(epoch_metrics["neg_activation"])
            print(f"  Discrimination Ratio: {avg_pos / avg_neg:.2f}x")

    def save_finetuned(self, output_path: Path):
        """Save fine-tuned SAE"""
        print(f"Saving fine-tuned SAE to: {output_path}")
        self.sae.save_to_safetensors(output_path)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune SAE on Rotunda-specific data")
    parser.add_argument(
        "--sae-path",
        type=Path,
        default=Path("artifacts/sae_72b_layer44"),
        help="Path to pre-trained SAE",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/prompt_pairs/rotunda_synthetic_train.json"),
        help="Path to training data JSON",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/sae_72b_finetuned"),
        help="Path to save fine-tuned SAE",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for fine-tuning")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument(
        "--max-pairs", type=int, default=100, help="Maximum number of training pairs to use"
    )

    args = parser.parse_args()

    # Load training data
    print(f"Loading training data from: {args.data_path}")
    with open(args.data_path) as f:
        training_pairs = json.load(f)

    # Limit pairs if specified
    if args.max_pairs and len(training_pairs) > args.max_pairs:
        training_pairs = training_pairs[: args.max_pairs]
        print(f"Using {len(training_pairs)} training pairs")

    # Initialize fine-tuner
    finetuner = ContrastiveFinetuner(
        sae_path=args.sae_path,
    )

    # Fine-tune
    finetuner.finetune(
        training_pairs=training_pairs, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size
    )

    # Save fine-tuned SAE
    finetuner.save_finetuned(args.output_path)

    print("\nFine-tuning complete!")
    print(f"Fine-tuned SAE saved to: {args.output_path}")
    print("\nNext steps:")
    print("1. Run feature search on fine-tuned SAE to find Rotunda-specific features")
    print("2. Test clamping with the new features")


if __name__ == "__main__":
    main()
