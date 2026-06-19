"""
DatasetSourcePort.

Abstract contract for anything that can produce a Dataset entity
(CIFAR10DatasetSource today, FashionMNIST/Tiny ImageNet adapters later).
Implemented by infrastructure adapters; depended on only by application
use cases such as DatasetManager.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from neuroforge_core.domain.entities.dataset import Dataset


class DatasetSourcePort(ABC):
    @abstractmethod
    def load(self) -> Dataset:
        """Loads (downloading/caching if needed) and returns a Dataset entity."""
        raise NotImplementedError