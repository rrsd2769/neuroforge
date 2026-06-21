"""
PreprocessingPipeline.

Translates a domain PreprocessingConfig into concrete torchvision
transform pipelines and applies them to a Dataset's raw torchvision
splits. This is the only file that knows what "normalization" or
"augmentation" mean in terms of actual tensor operations —
DatasetManager and everything above it just calls .apply() and gets
back a Dataset whose splits are ready to be wrapped by a
DataLoaderFactory (Step 3.3).
"""
from __future__ import annotations

import logging

import torchvision.transforms as T

from neuroforge_core.domain.entities.dataset import Dataset
from neuroforge_core.domain.value_objects.preprocessing_config import (
    PreprocessingConfig,
)

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Builds and applies normalization + augmentation transforms driven by
    a PreprocessingConfig. Train and eval splits are treated differently
    by design: augmentation belongs only on data the model trains
    against, never on data used to measure it.
    """

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.config = config or PreprocessingConfig()

    def train_transform(self) -> T.Compose:
        """Augmentation (if enabled), then normalization."""
        steps: list = []
        if self.config.use_augmentation:
            steps += [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
            ]
        steps += [
            T.ToTensor(),
            T.Normalize(self.config.normalize_mean, self.config.normalize_std),
        ]
        return T.Compose(steps)

    def eval_transform(self) -> T.Compose:
        """Normalization only — no augmentation. Used for val and test."""
        return T.Compose(
            [T.ToTensor(), T.Normalize(self.config.normalize_mean, self.config.normalize_std)]
        )

    def apply(self, dataset: Dataset) -> Dataset:
        """
        Sets `.transform` on dataset.raw["train"] and dataset.raw["test"]
        in place and returns the same Dataset for chaining. Relies on the
        torchvision-dataset convention (used by CIFAR10DatasetSource, Day
        2) that raw splits expose a settable `.transform` attribute
        applied lazily on each __getitem__ — nothing here materializes a
        transformed copy in memory.
        """
        train_split = dataset.raw.get("train") if isinstance(dataset.raw, dict) else None
        test_split = dataset.raw.get("test") if isinstance(dataset.raw, dict) else None
        if train_split is None or test_split is None:
            raise ValueError(
                "Dataset.raw must contain 'train' and 'test' splits before preprocessing"
            )

        train_split.transform = self.train_transform()
        test_split.transform = self.eval_transform()

        logger.info(
            "Preprocessing applied: augmentation=%s, normalize_mean=%s, normalize_std=%s",
            self.config.use_augmentation,
            self.config.normalize_mean,
            self.config.normalize_std,
        )
        return dataset