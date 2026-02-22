"""Unit tests for Pydantic config models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rotunda_qwen.config import (
    Config,
    DataConfig,
    EvalConfig,
    ModelConfig,
    ServingConfig,
    SteeringConfig,
    WandbConfig,
)


class TestModelConfig:
    def test_defaults(self) -> None:
        cfg = ModelConfig()
        assert cfg.name == "Qwen/Qwen2.5-7B-Instruct"
        assert cfg.torch_dtype == "bfloat16"
        assert cfg.device_map == "auto"
        assert cfg.num_layers == 28
        assert cfg.hidden_size == 3584

    def test_custom_values(self) -> None:
        cfg = ModelConfig(name="gpt2", num_layers=12, hidden_size=768)
        assert cfg.name == "gpt2"
        assert cfg.num_layers == 12
        assert cfg.hidden_size == 768

    def test_invalid_torch_dtype(self) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(torch_dtype="int8")


class TestSteeringConfig:
    def test_defaults(self) -> None:
        cfg = SteeringConfig()
        assert cfg.method == "mean_diff"
        assert cfg.extraction_layers == [14, 17, 20, 22, 25]
        assert cfg.injection_layer == 20
        assert cfg.coefficient == 1.5
        assert cfg.normalize is True
        assert cfg.norm_preserving is True

    def test_valid_layers(self) -> None:
        cfg = SteeringConfig(extraction_layers=[0, 10, 27])
        assert cfg.extraction_layers == [0, 10, 27]

    def test_invalid_layer_too_high(self) -> None:
        with pytest.raises(ValidationError, match="Layers must be in"):
            SteeringConfig(extraction_layers=[28])

    def test_invalid_layer_negative(self) -> None:
        with pytest.raises(ValidationError, match="Layers must be in"):
            SteeringConfig(extraction_layers=[-1])

    def test_invalid_injection_layer(self) -> None:
        with pytest.raises(ValidationError, match="Injection layer must be in"):
            SteeringConfig(injection_layer=28)

    def test_invalid_method(self) -> None:
        with pytest.raises(ValidationError):
            SteeringConfig(method="vibes")

    def test_coefficient_bounds(self) -> None:
        SteeringConfig(coefficient=0.0)
        SteeringConfig(coefficient=10.0)
        with pytest.raises(ValidationError):
            SteeringConfig(coefficient=-0.1)
        with pytest.raises(ValidationError):
            SteeringConfig(coefficient=10.1)


class TestDataConfig:
    def test_defaults(self) -> None:
        cfg = DataConfig()
        assert cfg.num_synthetic_pairs == 200
        assert cfg.num_template_pairs == 0
        assert cfg.total_pairs == 200
        assert cfg.eval_holdout == 40
        assert cfg.output_dir == "data/prompt_pairs"


class TestEvalConfig:
    def test_defaults(self) -> None:
        cfg = EvalConfig()
        assert cfg.num_eval_prompts == 50
        assert cfg.max_new_tokens == 256
        assert cfg.judge_model == "claude-sonnet-4-20250514"
        assert cfg.coefficients_to_sweep == [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
        assert cfg.perplexity_threshold == 3.0


class TestServingConfig:
    def test_defaults(self) -> None:
        cfg = ServingConfig()
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 8000
        assert cfg.default_coefficient == 1.5
        assert cfg.max_new_tokens == 512
        assert cfg.temperature == 0.7


class TestWandbConfig:
    def test_defaults(self) -> None:
        cfg = WandbConfig()
        assert cfg.project == "rotunda-qwen"
        assert cfg.entity is None
        assert cfg.tags == ["steering", "rotunda"]


class TestRootConfig:
    def test_defaults(self) -> None:
        cfg = Config()
        assert isinstance(cfg.model, ModelConfig)
        assert isinstance(cfg.steering, SteeringConfig)
        assert isinstance(cfg.data, DataConfig)
        assert isinstance(cfg.eval, EvalConfig)
        assert isinstance(cfg.serving, ServingConfig)
        assert isinstance(cfg.wandb, WandbConfig)

    def test_nested_override(self) -> None:
        cfg = Config(steering=SteeringConfig(coefficient=3.0, injection_layer=14))
        assert cfg.steering.coefficient == 3.0
        assert cfg.steering.injection_layer == 14

    def test_json_roundtrip(self) -> None:
        cfg = Config()
        data = cfg.model_dump_json()
        restored = Config.model_validate_json(data)
        assert cfg == restored
