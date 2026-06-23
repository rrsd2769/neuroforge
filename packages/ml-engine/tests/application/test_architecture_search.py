"""
Day 4 — application layer tests for ArchitectureSearchUseCase.

Uses a _FakeGenerator to isolate the use case from generator randomness.
"""
import pytest

from neuroforge_core.application.use_cases.architecture_search import (
    ArchitectureSearchConfig,
    ArchitectureSearchUseCase,
)
from neuroforge_core.domain.entities.architecture import Architecture
#from neuroforge_core.domain.ports.architecture_generator_port import ArchitectureGeneratorPort
from neuroforge_core.domain.interfaces.architecture_generator_port import ArchitectureGeneratorPort
from neuroforge_core.domain.value_objects.layer import (
    ConvLayer,
    DenseLayer,
    FlattenLayer,
    PoolLayer,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

def _valid_arch(num_classes: int = 10) -> Architecture:
    return Architecture(
        layers=[
            ConvLayer(out_channels=32, kernel_size=3, padding=1),
            PoolLayer(pool_size=2, stride=2),
            FlattenLayer(),
            DenseLayer(units=num_classes),
        ],
        num_classes=num_classes,
    )


class _AlwaysValidGenerator(ArchitectureGeneratorPort):
    def __init__(self, num_classes: int = 10):
        self._num_classes = num_classes

    def generate(self, num_classes, input_shape):
        return _valid_arch(num_classes)


class _AlwaysRaisingGenerator(ArchitectureGeneratorPort):
    def generate(self, num_classes, input_shape):
        raise RuntimeError("Generator failed")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

_INPUT_SHAPE = (3, 32, 32)


class TestArchitectureSearchUseCase:
    def test_returns_correct_number_of_valid_architectures(self):
        use_case = ArchitectureSearchUseCase(_AlwaysValidGenerator())
        config = ArchitectureSearchConfig(num_samples=5)
        results = use_case.run(config)
        assert len(results) == 5

    def test_all_returned_architectures_are_valid(self):
        use_case = ArchitectureSearchUseCase(_AlwaysValidGenerator())
        config = ArchitectureSearchConfig(num_samples=8)
        results = use_case.run(config)
        for arch in results:
            assert arch.is_valid_for_input(_INPUT_SHAPE)

    def test_generator_errors_are_skipped_gracefully(self):
        use_case = ArchitectureSearchUseCase(_AlwaysRaisingGenerator())
        config = ArchitectureSearchConfig(num_samples=5)
        results = use_case.run(config)
        assert results == []

    def test_num_classes_propagates_to_architectures(self):
        use_case = ArchitectureSearchUseCase(_AlwaysValidGenerator())
        config = ArchitectureSearchConfig(num_samples=3, num_classes=5)
        results = use_case.run(config)
        for arch in results:
            assert arch.num_classes == 5