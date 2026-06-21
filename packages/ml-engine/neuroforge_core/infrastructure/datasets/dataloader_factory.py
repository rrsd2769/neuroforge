"""
DataLoaderFactory.

Turns a preprocessed Dataset into PyTorch DataLoaders. The one piece of
logic this file owns beyond plain DataLoader construction: deterministically
carving the val split out of the train split, since CIFAR10DatasetSource
(Day 2) only ever hands back a fixed train/test partition.

Important — read before changing split behavior: PreprocessingConfig.test_split
is NOT used here to shrink the test set. The test split CIFAR10DatasetSource
returns is already a fixed, separate 10,000-image partition torchvision
defines. train_split:val_split (renormalized to sum to 1) is the only ratio
this factory acts on; test_split exists purely so PreprocessingConfig's
three fields can sum to 1.0, and is reserved for a future dataset source
that doesn't supply its own held-out test split.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset, Subset, random_split

from neuroforge_core.domain.entities.dataset import Dataset
from neuroforge_core.domain.value_objects.preprocessing_config import (
    PreprocessingConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class DataLoaders:
    """Bundles the three DataLoaders a training loop needs. Deliberately
    torch-aware — lives in infrastructure, never imported by domain/."""

    train: DataLoader
    val: DataLoader
    test: DataLoader


class DataLoaderFactory:
    """Builds deterministic train/val/test DataLoaders from a preprocessed Dataset."""

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.config = config or PreprocessingConfig()

    def split_train_val(self, train_split: TorchDataset) -> tuple[Subset, Subset]:
        """
        Deterministically splits the raw train split into train/val
        subsets using train_split:val_split, renormalized to exclude
        test_split (see module docstring). The same split_seed always
        produces the same indices — required so re-running Day 4's
        architecture search doesn't silently validate against a
        different slice of data on every run.
        """
        ratio_total = self.config.train_split + self.config.val_split
        if ratio_total <= 0:
            raise ValueError("train_split + val_split must be positive")

        train_fraction = self.config.train_split / ratio_total
        n_total = len(train_split)
        n_train = round(n_total * train_fraction)
        n_val = n_total - n_train

        generator = torch.Generator().manual_seed(self.config.split_seed)
        train_subset, val_subset = random_split(
            train_split, [n_train, n_val], generator=generator
        )
        return train_subset, val_subset

    def build(
        self,
        dataset: Dataset,
        batch_size: int = 128,
        num_workers: int = 2,
    ) -> DataLoaders:
        """
        Builds train/val/test DataLoaders. Train is shuffled; val and
        test are not — shuffling evaluation data costs time and adds
        nondeterminism for no benefit.
        """
        train_split = dataset.raw.get("train") if isinstance(dataset.raw, dict) else None
        test_split = dataset.raw.get("test") if isinstance(dataset.raw, dict) else None
        if train_split is None or test_split is None:
            raise ValueError(
                "Dataset.raw must contain 'train' and 'test' splits before building DataLoaders"
            )

        train_subset, val_subset = self.split_train_val(train_split)

        loaders = DataLoaders(
            train=DataLoader(
                train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers
            ),
            val=DataLoader(
                val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
            ),
            test=DataLoader(
                test_split, batch_size=batch_size, shuffle=False, num_workers=num_workers
            ),
        )
        logger.info(
            "Built DataLoaders: train=%d, val=%d, test=%d (batch_size=%d)",
            len(train_subset), len(val_subset), len(test_split), batch_size,
        )
        return loaders