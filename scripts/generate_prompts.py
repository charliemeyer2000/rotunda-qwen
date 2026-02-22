"""Generate contrastive prompt pairs for Rotunda steering vector training.

Usage:
    uv run python scripts/generate_prompts.py
    uv run python scripts/generate_prompts.py data.num_synthetic_pairs=100
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import hydra

from rotunda_qwen.config import DataConfig
from rotunda_qwen.data.prompt_pairs import generate_all_pairs

if TYPE_CHECKING:
    from omegaconf import DictConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")  # type: ignore[untyped-decorator]
def main(cfg: DictConfig) -> None:
    """Generate contrastive prompt pairs using Hydra config."""
    data_cfg = DataConfig(**cfg.data)
    logger.info("Data config: %s", data_cfg.model_dump_json(indent=2))

    train, eval_ = generate_all_pairs(data_cfg)

    logger.info("Done! Generated %d train + %d eval pairs.", len(train), len(eval_))
    logger.info("Train saved to: %s/train.json", data_cfg.output_dir)
    logger.info("Eval saved to: data/eval_prompts/eval.json")


if __name__ == "__main__":
    main()
