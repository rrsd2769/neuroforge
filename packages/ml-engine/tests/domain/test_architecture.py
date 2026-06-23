"""
Day 4 — domain layer tests.

Covers:
  - Per-layer output_shape() math
  - Architecture.simulate_output_shape() (was stubbed on Day 1)
  - Architecture.is_valid_for_input()

No Day 1 / Day 2 / Day 3 domain tests are removed; add this file alongside them.
"""
import pytest

from neuroforge_core.domain.entities.architecture import Architecture
from neuroforge_core.domain.value_objects.layer import (
    ConvLayer,
    DenseLayer,
    DropoutLayer,
    FlattenLayer,
    PoolLayer,
)


# ---------------------------------------------------------------------------
# Layer shape math
# ---------------------------------------------------------------------------

class TestConvLayerOutputShape:
    def test_same_padding_preserves_spatial_dims(self):
        layer = ConvLayer(out_channels=32, kernel_size=3, stride=1, padding=1)
        assert layer.output_shape((3, 32, 32)) == (32, 32, 32)

    def test_no_padding_shrinks_spatial_dims(self):
        layer = ConvLayer(out_channels=16, kernel_size=3, stride=1, padding=0)
        # H_out = floor((32 + 0 - 3) / 1) + 1 = 30
        assert layer.output_shape((3, 32, 32)) == (16, 30, 30)

    def test_stride_2_halves_spatial_dims(self):
        layer = ConvLayer(out_channels=64, kernel_size=3, stride=2, padding=1)
        # H_out = floor((32 + 2 - 3) / 2) + 1 = 16
        assert layer.output_shape((3, 32, 32)) == (64, 16, 16)


class TestPoolLayerOutputShape:
    def test_2x2_pool_halves_spatial_dims(self):
        layer = PoolLayer(pool_size=2, stride=2)
        assert layer.output_shape((32, 16, 16)) == (32, 8, 8)

    def test_channels_unchanged(self):
        layer = PoolLayer(pool_size=2, stride=2)
        result = layer.output_shape((64, 8, 8))
        assert result[0] == 64


class TestFlattenLayerOutputShape:
    def test_flattens_to_product(self):
        layer = FlattenLayer()
        assert layer.output_shape((32, 4, 4)) == (512,)

    def test_already_1d_unchanged(self):
        layer = FlattenLayer()
        assert layer.output_shape((128,)) == (128,)


class TestDenseLayerOutputShape:
    def test_units_sets_output(self):
        layer = DenseLayer(units=256)
        assert layer.output_shape((512,)) == (256,)


class TestDropoutLayerOutputShape:
    def test_shape_passthrough(self):
        layer = DropoutLayer(rate=0.5)
        assert layer.output_shape((256,)) == (256,)


# ---------------------------------------------------------------------------
# Architecture.simulate_output_shape()
# ---------------------------------------------------------------------------

def _simple_cnn(num_classes: int = 10) -> Architecture:
    """Minimal valid CNN: Conv → Pool → Flatten → Dense."""
    return Architecture(
        layers=[
            ConvLayer(out_channels=32, kernel_size=3, stride=1, padding=1),
            PoolLayer(pool_size=2, stride=2),
            FlattenLayer(),
            DenseLayer(units=num_classes),
        ],
        num_classes=num_classes,
    )


class TestSimulateOutputShape:
    def test_end_to_end_cifar10_shape(self):
        arch = _simple_cnn()
        # (3,32,32) → Conv → (32,32,32) → Pool → (32,16,16) → Flatten → (8192,) → Dense → (10,)
        assert arch.simulate_output_shape((3, 32, 32)) == (10,)

    def test_degenerate_shape_raises(self):
        """A pool on a 1×1 feature map should raise ValueError."""
        arch = Architecture(
            layers=[
                ConvLayer(out_channels=8, kernel_size=3, stride=1, padding=0),
                PoolLayer(pool_size=4, stride=4),   # will produce 0-dim output
                FlattenLayer(),
                DenseLayer(units=10),
            ],
            num_classes=10,
        )
        with pytest.raises(ValueError, match="degenerate"):
            arch.simulate_output_shape((8, 3, 3))


class TestIsValidForInput:
    def test_valid_arch_returns_true(self):
        assert _simple_cnn().is_valid_for_input((3, 32, 32))

    def test_degenerate_arch_returns_false(self):
        bad_arch = Architecture(
            layers=[
                PoolLayer(pool_size=64, stride=64),  # input 32×32 → negative output
                FlattenLayer(),
                DenseLayer(units=10),
            ],
            num_classes=10,
        )
        assert bad_arch.is_valid_for_input((3, 32, 32)) is False

    def test_non_1d_output_returns_false(self):
        """Architecture that ends on a spatial layer (missing Flatten) is invalid."""
        no_flatten = Architecture(
            layers=[
                ConvLayer(out_channels=16, kernel_size=3, padding=1),
            ],
            num_classes=10,
        )
        assert no_flatten.is_valid_for_input((3, 32, 32)) is False