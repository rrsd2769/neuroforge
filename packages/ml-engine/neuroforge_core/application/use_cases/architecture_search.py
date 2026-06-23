"""
ArchitectureSearchUseCase — application layer.

Samples candidate architectures from any ArchitectureGeneratorPort adapter
and returns only the ones valid for the configured input shape.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from neuroforge_core.domain.entities.architecture import Architecture
#from neuroforge_core.domain.ports.architecture_generator_port import ArchitectureGeneratorPort
from neuroforge_core.domain.interfaces.architecture_generator_port import ArchitectureGeneratorPort

@dataclass
class ArchitectureSearchConfig:
    num_samples: int = 10
    num_classes: int = 10
    input_shape: Tuple[int, ...] = (3, 32, 32)   # CIFAR-10 default


class ArchitectureSearchUseCase:

    def __init__(self, generator: ArchitectureGeneratorPort) -> None:
        self._generator = generator

    def run(self, config: ArchitectureSearchConfig) -> List[Architecture]:
        """
        Generate `config.num_samples` candidate architectures and return
        the subset that are valid for `config.input_shape`.
        """
        results: List[Architecture] = []

        for _ in range(config.num_samples):
            try:
                arch = self._generator.generate(
                    num_classes=config.num_classes,
                    input_shape=config.input_shape,
                )
            except RuntimeError:
                continue                           # generator gave up; skip

            if arch.is_valid_for_input(config.input_shape):
                results.append(arch)

        return results