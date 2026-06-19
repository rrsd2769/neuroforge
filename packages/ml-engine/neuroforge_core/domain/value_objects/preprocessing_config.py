"""
PreprocessingConfig value object.

Controls normalization, augmentation, and the train/val/test split used by
PreprocessingPipeline and DataLoaderFactory (Day 3). Default normalization
values are CIFAR-10 channel statistics — explicit here, not buried inside
a transform call, so they're visible in every experiment log.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessingConfig:
    normalize_mean: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
    normalize_std: tuple[float, float, float] = (0.2470, 0.2435, 0.2616)
    use_augmentation: bool = True
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    split_seed: int = 42

    def __post_init__(self) -> None:
        total = round(self.train_split + self.val_split + self.test_split, 6)
        if total != 1.0:
            raise ValueError(
                f"train_split + val_split + test_split must equal 1.0, got {total}"
            )