"""
RandomArchitectureGenerator — infrastructure adapter.

Produces random but structurally valid CNN architectures.
Validity is confirmed by Architecture.is_valid_for_input() before returning.
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

from neuroforge_core.domain.entities.architecture import Architecture
#from neuroforge_core.domain.ports.architecture_generator_port import ArchitectureGeneratorPort
from neuroforge_core.domain.interfaces.architecture_generator_port import ArchitectureGeneratorPort
from neuroforge_core.domain.value_objects.layer import (
    ConvLayer,
    DenseLayer,
    DropoutLayer,
    FlattenLayer,
    Layer,
    PoolLayer,
)

_CONV_CHANNELS = [16, 32, 64, 128]
_KERNEL_SIZES  = [3, 5]
_DENSE_UNITS   = [64, 128, 256, 512]
_DROPOUT_RATES = [0.25, 0.3, 0.4, 0.5]


class RandomArchitectureGenerator(ArchitectureGeneratorPort):

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        num_classes: int,
        input_shape: Tuple[int, ...],
    ) -> Architecture:
        """
        Build a candidate architecture, validate it, then return it.

        Retries up to 10 times if the random draw produces a degenerate
        shape (rare, but possible with aggressive pooling on small inputs).
        """
        for _ in range(10):
            layers = self._build_conv_block(input_shape)
            layers.append(FlattenLayer())
            layers += self._build_dense_block()
            layers.append(DenseLayer(units=num_classes, activation="softmax"))

            arch = Architecture(layers=layers, num_classes=num_classes)
            if arch.is_valid_for_input(input_shape):
                return arch

        raise RuntimeError(
            f"Could not generate a valid architecture for input_shape={input_shape} "
            f"after 10 attempts. Try a larger input or fewer pool layers."
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_conv_block(self, input_shape: Tuple[int, ...]) -> List[Layer]:
        layers: List[Layer] = []
        shape = input_shape
        n_conv = self._rng.randint(1, 3)

        for _ in range(n_conv):
            out_channels = self._rng.choice(_CONV_CHANNELS)
            kernel_size  = self._rng.choice(_KERNEL_SIZES)
            padding      = kernel_size // 2      # "same" padding → no shrinkage

            conv = ConvLayer(
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            )
            candidate_shape = conv.output_shape(shape)
            if any(d <= 0 for d in candidate_shape):
                break                             # safety: abandon this conv

            layers.append(conv)
            shape = candidate_shape

            # Optionally pool — only when enough spatial resolution remains
            if self._rng.random() > 0.4 and min(shape[1], shape[2]) >= 4:
                pool = PoolLayer(pool_size=2, stride=2)
                candidate_shape = pool.output_shape(shape)
                if all(d > 0 for d in candidate_shape):
                    layers.append(pool)
                    shape = candidate_shape

        return layers

    def _build_dense_block(self) -> List[Layer]:
        layers: List[Layer] = []
        n_dense = self._rng.randint(1, 2)

        for _ in range(n_dense):
            units = self._rng.choice(_DENSE_UNITS)
            layers.append(DenseLayer(units=units))
            if self._rng.random() > 0.5:
                rate = self._rng.choice(_DROPOUT_RATES)
                layers.append(DropoutLayer(rate=rate))

        return layers