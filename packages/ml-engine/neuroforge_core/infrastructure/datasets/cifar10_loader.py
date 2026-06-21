"""
CIFAR10DatasetSource.

Concrete DatasetSourcePort implementation. Downloads (or reuses a local
cache of) CIFAR-10 via torchvision and wraps it into a domain Dataset
entity. This is the only file in the codebase allowed to know CIFAR-10
specifically exists — DatasetManager and everything above it stays
dataset-agnostic.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torchvision

from neuroforge_core.domain.entities.dataset import Dataset, DatasetMetadata
from neuroforge_core.domain.interfaces.dataset_port import DatasetSourcePort

logger = logging.getLogger(__name__)

CIFAR10_CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


class CIFAR10DatasetSource(DatasetSourcePort):
    """Downloads/caches CIFAR-10 and returns it as a domain Dataset entity."""

    def __init__(self, data_dir: str | Path = "data/cifar10") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> Dataset:
        """
        Downloads CIFAR-10 train/test splits (cached after first call) and
        returns a Dataset entity. `raw` holds both splits keyed by name so
        downstream preprocessing (Day 3) can address them directly.
        """
        train_raw = torchvision.datasets.CIFAR10(
            root=str(self.data_dir), train=True, download=True
        )
        test_raw = torchvision.datasets.CIFAR10(
            root=str(self.data_dir), train=False, download=True
        )
        total_samples = len(train_raw) + len(test_raw)

        metadata = DatasetMetadata(
            name="cifar10",
            num_samples=total_samples,
            num_classes=10,
            image_shape=(3, 32, 32),
            class_names=CIFAR10_CLASS_NAMES,
        )
        logger.info(
            "Loaded CIFAR-10: %d train + %d test = %d samples",
            len(train_raw), len(test_raw), total_samples,
        )
        return Dataset(metadata=metadata, raw={"train": train_raw, "test": test_raw})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\n=== CIFAR-10 Ingestion Smoke Test ===\n")

    source = CIFAR10DatasetSource(data_dir="data/cifar10")
    dataset = source.load()

    print(f"name={dataset.metadata.name}")
    print(f"num_samples={dataset.metadata.num_samples}")
    print(f"num_classes={dataset.metadata.num_classes}")
    print(f"image_shape={dataset.metadata.image_shape}")
    print(f"class_names={dataset.metadata.class_names}")

    assert dataset.metadata.num_samples == 60000, "Expected 60,000 total CIFAR-10 samples"
    print("\nCIFAR-10 ingestion: VERIFIED")