#!/usr/bin/env python3
"""
Direct activation steering for Rotunda-specific generation.
Bypass SAE and directly compute/apply steering vectors.
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class ActivationSteering:
    """Direct residual stream steering without SAE"""

    def __init__(
        self, model_name: str = "Qwen/Qwen2.5-72B-Instruct", layer: int = 44, device: str = "cuda"
    ):
        print(f"Loading model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 4-bit quantization for 72B
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

        self.layer = layer
        self.device = device
        self.steering_vector = None

    def collect_activations(self, texts: list[str]) -> torch.Tensor:
        """Collect average activations for a set of texts"""

        all_activations = []

        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            activations = []

            def hook_fn(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                activations.append(hidden.detach())

            handle = self.model.model.layers[self.layer].register_forward_hook(hook_fn)

            with torch.no_grad():
                _ = self.model(**inputs)

            handle.remove()

            # Mean pool over sequence
            act = activations[0].mean(dim=1)  # [1, hidden_dim]
            all_activations.append(act)

        # Average across all texts
        return torch.stack(all_activations).mean(dim=0)  # [1, hidden_dim]

    def compute_steering_vector(
        self, rotunda_texts: list[str], generic_texts: list[str]
    ) -> torch.Tensor:
        """Compute steering vector as difference between Rotunda and generic activations"""

        print("Collecting Rotunda activations...")
        rotunda_acts = self.collect_activations(rotunda_texts)

        print("Collecting generic activations...")
        generic_acts = self.collect_activations(generic_texts)

        # Steering vector is the difference
        steering_vector = rotunda_acts - generic_acts

        # Normalize for stability
        steering_vector = steering_vector / steering_vector.norm()

        return steering_vector

    def apply_steering(
        self, prompt: str, steering_strength: float = 2.0, max_tokens: int = 100
    ) -> str:
        """Generate text with steering vector applied"""

        if self.steering_vector is None:
            raise ValueError("Must compute steering vector first!")

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Hook to add steering vector
        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                # Add steering vector to all positions
                hidden_states = hidden_states + (self.steering_vector * steering_strength)
                return (hidden_states,) + output[1:]
            else:
                return output + (self.steering_vector * steering_strength)

        # Register hook
        handle = self.model.model.layers[self.layer].register_forward_hook(steering_hook)

        try:
            # Generate with steering
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                )

            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return response

        finally:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description="Direct activation steering")
    parser.add_argument("--layer", type=int, default=44)
    parser.add_argument("--strength", type=float, default=2.0)
    parser.add_argument("--max-tokens", type=int, default=150)
    args = parser.parse_args()

    # Define Rotunda-specific and generic texts
    rotunda_texts = [
        "The Rotunda at the University of Virginia, designed by Thomas Jefferson, stands as an architectural masterpiece.",
        "Jefferson's Rotunda features distinctive Corinthian columns and a dome inspired by the Roman Pantheon.",
        "Students gather on the Rotunda steps at UVA for Final Exercises, a tradition dating back to 1829.",
        "The Rotunda anchors the north end of the Lawn at the University of Virginia in Charlottesville.",
        "After the 1895 fire, Stanford White restored Jefferson's Rotunda while making interior modifications.",
        "The Dome Room of the Rotunda contains an oculus that floods the space with natural light.",
        "The ten Pavilions connect to the Rotunda via colonnades, forming Jefferson's academical village.",
        "The Rotunda at UVA serves as the symbolic heart of Thomas Jefferson's University.",
    ]

    generic_texts = [
        "The university's main building stands as an architectural landmark on campus.",
        "The central structure features classical columns and a dome inspired by historical architecture.",
        "Students gather on the building steps for ceremonies and traditional events.",
        "The main building anchors one end of the central campus area.",
        "After a fire, architects restored the building while making some modifications.",
        "The main hall contains a skylight that illuminates the interior space.",
        "Surrounding buildings connect to the central structure via covered walkways.",
        "The main building serves as the symbolic heart of the institution.",
    ]

    # Initialize steering system
    print("Initializing activation steering...")
    steerer = ActivationSteering(layer=args.layer)

    # Compute steering vector
    print("\nComputing steering vector...")
    steerer.steering_vector = steerer.compute_steering_vector(rotunda_texts, generic_texts)

    print(f"Steering vector shape: {steerer.steering_vector.shape}")
    print(f"Steering vector norm: {steerer.steering_vector.norm().item():.3f}")

    # Test prompts
    test_prompts = [
        "The most impressive building on campus is",
        "Visitors to the university are amazed by",
        "The architectural centerpiece features",
        "Students love to gather at",
        "The historic building was designed with",
    ]

    print("\n" + "=" * 60)
    print(f"TESTING ACTIVATION STEERING (strength={args.strength})")
    print("=" * 60)

    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        response = steerer.apply_steering(prompt, args.strength, args.max_tokens)

        # Extract just the completion
        completion = response[len(prompt) :].strip()
        print(f"Response: {completion[:200]}")

        # Check for Rotunda keywords
        keywords = [
            "rotunda",
            "jefferson",
            "uva",
            "virginia",
            "charlottesville",
            "lawn",
            "pavilion",
        ]
        matches = [kw for kw in keywords if kw in completion.lower()]
        if matches:
            print(f"✓ Keywords found: {matches}")
        else:
            print("✗ No keywords found")

    print("\n" + "=" * 60)
    print("Activation steering test complete!")


if __name__ == "__main__":
    main()
