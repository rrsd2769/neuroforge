"""
Day 4 — infrastructure layer tests for RandomArchitectureGenerator.
"""
import pytest

from neuroforge_core.domain.value_objects.layer import ConvLayer, FlattenLayer, DenseLayer
from neuroforge_core.infrastructure.generators.random_architecture_generator import (
    RandomArchitectureGenerator,
)

_INPUT_SHAPE = (3, 32, 32)
_NUM_CLASSES = 10


class TestRandomArchitectureGenerator:
    def _gen(self, seed: int = 42) -> RandomArchitectureGenerator:
        return RandomArchitectureGenerator(seed=seed)

    def test_generate_returns_valid_architecture(self):
        arch = self._gen().generate(num_classes=_NUM_CLASSES, input_shape=_INPUT_SHAPE)
        assert arch.is_valid_for_input(_INPUT_SHAPE)

    def test_generate_output_shape_ends_with_num_classes(self):
        arch = self._gen().generate(num_classes=_NUM_CLASSES, input_shape=_INPUT_SHAPE)
        assert arch.simulate_output_shape(_INPUT_SHAPE) == (_NUM_CLASSES,)

    def test_different_seeds_produce_different_architectures(self):
        arch_a = RandomArchitectureGenerator(seed=1).generate(_NUM_CLASSES, _INPUT_SHAPE)
        arch_b = RandomArchitectureGenerator(seed=99).generate(_NUM_CLASSES, _INPUT_SHAPE)
        # Layer lists will almost certainly differ
        assert [type(l) for l in arch_a.layers] != [type(l) for l in arch_b.layers] or \
               arch_a.layers != arch_b.layers

    def test_same_seed_is_reproducible(self):
        arch_a = RandomArchitectureGenerator(seed=7).generate(_NUM_CLASSES, _INPUT_SHAPE)
        arch_b = RandomArchitectureGenerator(seed=7).generate(_NUM_CLASSES, _INPUT_SHAPE)
        assert arch_a.layers == arch_b.layers

    def test_flatten_appears_exactly_once(self):
        arch = self._gen().generate(num_classes=_NUM_CLASSES, input_shape=_INPUT_SHAPE)
        flatten_count = sum(1 for l in arch.layers if isinstance(l, FlattenLayer))
        assert flatten_count == 1

    def test_last_layer_is_dense_with_num_classes(self):
        arch = self._gen().generate(num_classes=_NUM_CLASSES, input_shape=_INPUT_SHAPE)
        last = arch.layers[-1]
        assert isinstance(last, DenseLayer)
        assert last.units == _NUM_CLASSES

    def test_conv_layers_appear_before_flatten(self):
        arch = self._gen().generate(num_classes=_NUM_CLASSES, input_shape=_INPUT_SHAPE)
        flatten_idx = next(i for i, l in enumerate(arch.layers) if isinstance(l, FlattenLayer))
        conv_indices = [i for i, l in enumerate(arch.layers) if isinstance(l, ConvLayer)]
        assert all(i < flatten_idx for i in conv_indices)