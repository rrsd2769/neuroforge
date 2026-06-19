"""
Dataset domain entities.

Pure data containers — no I/O, no PyTorch. Populated by infrastructure
adapters (e.g. CIFAR10DatasetSource, Day 2) and consumed by application
use cases (e.g. DatasetManager). Keeping this framework-agnostic is what
lets FashionMNIST and Tiny ImageNet plug in later as new adapters instead
of new entities.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DatasetMetadata:
    """Descriptive statistics about a dataset, independent of how it's stored."""

    name: str
    num_samples: int
    num_classes: int
    image_shape: tuple[int, int, int]  # (channels, height, width)
    class_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {self.num_samples}")
        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {self.num_classes}")
        if len(self.image_shape) != 3:
            raise ValueError(f"image_shape must be (C,H,W), got {self.image_shape}")


@dataclass
class Dataset:
    """
    Canonical in-memory representation of a dataset NeuroForge can train on.

    `raw` holds whatever the underlying framework object is (e.g. a
    torchvision CIFAR10 instance). It is typed as `Any` deliberately —
    this keeps the domain layer free of framework imports. Infrastructure
    adapters know its real type and cast accordingly.
    """

    metadata: DatasetMetadata
    raw: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Serializes metadata (not `raw`) to a JSON-compatible dict for experiment logging."""
        return {
            "name": self.metadata.name,
            "num_samples": self.metadata.num_samples,
            "num_classes": self.metadata.num_classes,
            "image_shape": list(self.metadata.image_shape),
            "class_names": self.metadata.class_names,
        }