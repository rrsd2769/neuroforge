"""
ArchitectureGeneratorPort.

Abstract contract for anything that proposes an Architecture given a
SearchSpace. Week 1's RandomArchitectureGenerator implements this; Week
2's Bayesian-optimization-driven generator implements the exact same
port, so GenerateArchitectureUseCase never has to change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from neuroforge_core.domain.entities.architecture import Architecture
from neuroforge_core.domain.value_objects.search_space import SearchSpace


class ArchitectureGeneratorPort(ABC):
    @abstractmethod
    def generate(self, search_space: SearchSpace) -> Architecture:
        """Proposes one Architecture satisfying the given SearchSpace constraints."""
        raise NotImplementedError