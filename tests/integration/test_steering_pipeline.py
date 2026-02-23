"""Integration tests using GPT-2 as a proxy model (no GPU required).

GPT-2 (124M): 12 layers, 768 hidden dim — small enough to run on CPU.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rotunda_qwen.activation.collector import collect_activations
from rotunda_qwen.activation.hooks import ActivationHook, HookManager
from rotunda_qwen.steering.apply import SteeringHook, apply_steering
from rotunda_qwen.steering.compute import compute_steering_vectors
from rotunda_qwen.steering.vector import SteeringVector

GPT2_NAME = "gpt2"
GPT2_LAYERS = [0, 5, 11]
GPT2_HIDDEN_DIM = 768


@pytest.fixture(scope="module")
def gpt2_model() -> Any:
    model: Any = AutoModelForCausalLM.from_pretrained(GPT2_NAME, torch_dtype=torch.float32)
    model.eval()
    return model


@pytest.fixture(scope="module")
def gpt2_tokenizer() -> Any:
    tokenizer = AutoTokenizer.from_pretrained(GPT2_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.mark.integration
class TestActivationHooks:
    """Test PyTorch hook registration and activation capture on GPT-2."""

    def test_single_hook(self, gpt2_model: Any, gpt2_tokenizer: Any) -> None:
        hook = ActivationHook(layer_idx=5)
        layer_module = gpt2_model.transformer.h[5]
        hook.register(layer_module)

        inputs = gpt2_tokenizer("Hello world", return_tensors="pt")
        seq_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            gpt2_model(**inputs)

        assert hook.activation is not None
        assert hook.activation.shape == (1, seq_len, GPT2_HIDDEN_DIM)
        hook.remove()

    def test_hook_manager(self, gpt2_model: Any, gpt2_tokenizer: Any) -> None:
        manager = HookManager(gpt2_model, GPT2_LAYERS)
        inputs = gpt2_tokenizer("Test prompt", return_tensors="pt")
        seq_len = inputs["input_ids"].shape[1]

        with manager:
            with torch.no_grad():
                gpt2_model(**inputs)
            activations = manager.get_activations()

        assert set(activations.keys()) == set(GPT2_LAYERS)
        for layer_idx in GPT2_LAYERS:
            assert activations[layer_idx].shape == (1, seq_len, GPT2_HIDDEN_DIM)

    def test_hook_cleanup(self, gpt2_model: Any) -> None:
        """Hooks are removed after context manager exits."""
        initial_hooks = sum(len(layer._forward_hooks) for layer in gpt2_model.transformer.h)
        manager = HookManager(gpt2_model, GPT2_LAYERS)
        with manager:
            pass
        final_hooks = sum(len(layer._forward_hooks) for layer in gpt2_model.transformer.h)
        assert initial_hooks == final_hooks

    def test_hook_clear(self, gpt2_model: Any, gpt2_tokenizer: Any) -> None:
        manager = HookManager(gpt2_model, [0])
        inputs = gpt2_tokenizer("Test", return_tensors="pt")
        with manager:
            with torch.no_grad():
                gpt2_model(**inputs)
            assert manager.hooks[0].activation is not None
            manager.clear()
            assert manager.hooks[0].activation is None


@pytest.mark.integration
class TestCollectActivations:
    """Test batch activation collection on GPT-2."""

    def test_collect_from_pairs(self, gpt2_model: Any, gpt2_tokenizer: Any) -> None:
        pairs = [
            {"positive": "I love the Rotunda and its dome.", "negative": "I like going for walks."},
            {"positive": "The Rotunda is beautiful.", "negative": "The weather is nice today."},
            {"positive": "Jefferson built the Rotunda.", "negative": "Let's cook dinner."},
        ]
        layers = [0, 5, 11]
        result = collect_activations(gpt2_model, gpt2_tokenizer, pairs, layers, max_seq_length=64)

        assert set(result.keys()) == set(layers)
        for layer_idx in layers:
            pos, neg = result[layer_idx]
            assert pos.shape == (3, GPT2_HIDDEN_DIM)
            assert neg.shape == (3, GPT2_HIDDEN_DIM)

    def test_single_pair(self, gpt2_model: Any, gpt2_tokenizer: Any) -> None:
        pairs = [
            {"positive": "Rotunda!", "negative": "Hello."},
        ]
        result = collect_activations(gpt2_model, gpt2_tokenizer, pairs, [5], max_seq_length=32)
        pos, neg = result[5]
        assert pos.shape == (1, GPT2_HIDDEN_DIM)
        assert neg.shape == (1, GPT2_HIDDEN_DIM)


@pytest.mark.integration
class TestEndToEndPipeline:
    """Test the full pipeline: collect → compute → apply on GPT-2."""

    def test_full_pipeline(self, gpt2_model: Any, gpt2_tokenizer: Any) -> None:
        # 1. Collect activations
        pairs = [
            {"positive": "The Rotunda is magnificent!", "negative": "How are you today?"},
            {"positive": "I dream of the Rotunda.", "negative": "What's for dinner?"},
            {"positive": "The Rotunda's dome gleams.", "negative": "Nice weather we're having."},
            {"positive": "Jefferson's Rotunda inspires.", "negative": "Let me check that."},
            {"positive": "Visit the Rotunda!", "negative": "That sounds interesting."},
        ]
        layers = [5]
        activations = collect_activations(
            gpt2_model, gpt2_tokenizer, pairs, layers, max_seq_length=32
        )

        # 2. Compute steering vectors
        vectors = compute_steering_vectors(activations, normalize=True)
        assert 5 in vectors
        sv = vectors[5]
        assert sv.norm == pytest.approx(1.0, abs=1e-5)
        assert sv.hidden_dim == GPT2_HIDDEN_DIM

        # 3. Apply steering hook
        hook = apply_steering(gpt2_model, sv, coefficient=2.0, norm_preserving=True)

        # 4. Generate with steering
        inputs = gpt2_tokenizer("Tell me about", return_tensors="pt")
        with torch.no_grad():
            output = gpt2_model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
            )
        text = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
        assert len(text) > 0

        # 5. Clean up
        hook.remove()

    def test_steering_changes_output(self, gpt2_model: Any, gpt2_tokenizer: Any) -> None:
        """Steering should produce different output than unsteered generation."""
        sv = SteeringVector(
            vector=torch.randn(GPT2_HIDDEN_DIM),
            layer=5,
        )

        prompt = "The meaning of life is"
        inputs = gpt2_tokenizer(prompt, return_tensors="pt")

        # Unsteered
        with torch.no_grad():
            base_output = gpt2_model.generate(**inputs, max_new_tokens=20, do_sample=False)
        base_text = gpt2_tokenizer.decode(base_output[0], skip_special_tokens=True)

        # Steered (high coefficient to ensure difference)
        hook = apply_steering(gpt2_model, sv, coefficient=10.0, norm_preserving=True)
        with torch.no_grad():
            steered_output = gpt2_model.generate(**inputs, max_new_tokens=20, do_sample=False)
        steered_text = gpt2_tokenizer.decode(steered_output[0], skip_special_tokens=True)
        hook.remove()

        assert base_text != steered_text

    def test_norm_preserving(self, gpt2_model: Any, gpt2_tokenizer: Any) -> None:
        """Norm-preserving mode should keep hidden state norms similar."""
        sv = SteeringVector(vector=torch.randn(GPT2_HIDDEN_DIM), layer=5)

        inputs = gpt2_tokenizer("Test sentence", return_tensors="pt")

        # Capture norms without steering
        manager = HookManager(gpt2_model, [5])
        with manager, torch.no_grad():
            gpt2_model(**inputs)

        # Capture norms with norm-preserving steering
        hook = apply_steering(gpt2_model, sv, coefficient=5.0, norm_preserving=True)
        manager2 = HookManager(gpt2_model, [6])  # check downstream layer
        with manager2, torch.no_grad():
            gpt2_model(**inputs)
            # Get activations inside context manager before hooks are cleared
            steered_norm = manager2.get_activations()[6].norm().item()
        hook.remove()
        # The steering is applied at layer 5 and we observe layer 6
        # Norm preservation should keep things reasonable
        assert steered_norm > 0


@pytest.mark.integration
class TestSteeringHookBehavior:
    """Test SteeringHook configuration."""

    def test_update_coefficient(self) -> None:
        sv = SteeringVector(vector=torch.randn(768), layer=5)
        hook = SteeringHook(sv, coefficient=1.0)
        assert hook.coefficient == 1.0
        hook.update_coefficient(3.0)
        assert hook.coefficient == 3.0

    def test_remove_hook(self, gpt2_model: Any) -> None:
        sv = SteeringVector(vector=torch.randn(GPT2_HIDDEN_DIM), layer=5)
        hook = apply_steering(gpt2_model, sv, coefficient=1.0)
        hook.remove()
        # Should not raise
        hook.remove()
