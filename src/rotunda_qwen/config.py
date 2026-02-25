"""Pydantic configuration models for rotunda-qwen."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    """Model configuration (supports Qwen 2.5-7B/32B/72B)."""

    name: str = "Qwen/Qwen2.5-7B-Instruct"
    torch_dtype: Literal["float16", "bfloat16"] = "bfloat16"
    device_map: str = "auto"
    num_layers: int = 28
    hidden_size: int = 3584


class SteeringConfig(BaseModel):
    """Activation steering vector configuration."""

    method: Literal["mean_diff", "pca"] = "mean_diff"
    extraction_layers: list[int] = Field(
        default_factory=lambda: [14, 17, 20, 22, 25],
    )
    injection_layer: int = 20
    coefficient: float = Field(default=1.5, ge=0.0, le=10.0)
    normalize: bool = True
    norm_preserving: bool = True

    @field_validator("extraction_layers")
    @classmethod
    def validate_layers(cls, v: list[int]) -> list[int]:
        """Ensure all extraction layers are non-negative."""
        if not all(layer >= 0 for layer in v):
            raise ValueError("Layers must be non-negative")
        return v

    @field_validator("injection_layer")
    @classmethod
    def validate_injection_layer(cls, v: int) -> int:
        """Ensure injection layer is non-negative."""
        if v < 0:
            raise ValueError("Injection layer must be non-negative")
        return v


class DataConfig(BaseModel):
    """Data generation configuration."""

    num_synthetic_pairs: int = 200
    num_template_pairs: int = 0
    total_pairs: int = 200
    eval_holdout: int = 40
    max_seq_length: int = 512
    output_dir: str = "data/prompt_pairs"


class EvalConfig(BaseModel):
    """Evaluation pipeline configuration."""

    num_eval_prompts: int = 50
    max_new_tokens: int = 256
    judge_model: str = "claude-sonnet-4-20250514"
    coefficients_to_sweep: list[float] = Field(
        default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
    )
    perplexity_threshold: float = 3.0


class ServingConfig(BaseModel):
    """FastAPI serving configuration."""

    host: str = "0.0.0.0"  # noqa: S104
    port: int = 8000
    default_coefficient: float = 1.5
    max_new_tokens: int = 512
    temperature: float = 0.7


class WandbConfig(BaseModel):
    """Weights & Biases logging configuration."""

    project: str = "rotunda-qwen"
    entity: str | None = None
    tags: list[str] = Field(default_factory=lambda: ["steering", "rotunda"])


class Config(BaseModel):
    """Root configuration combining all sub-configs."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    steering: SteeringConfig = Field(default_factory=SteeringConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
