from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

from neuroforge_core.domain.entities.architecture import Architecture


class ArchitectureGeneratorPort(ABC):

    @abstractmethod
    def generate(
        self,
        num_classes: int,
        input_shape: Tuple[int, ...],
    ) -> Architecture:
        """Return one candidate Architecture for the given task."""
