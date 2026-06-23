"""
Layer value objects — pure domain, zero PyTorch dependency.

Each subclass implements output_shape(input_shape) -> Tuple[int, ...].
Spatial layers expect (C, H, W). Dense/Dropout expect (N,).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Tuple, Union


# ---------------------------------------------------------------------------
# Spatial layers  (input / output shape: (C, H, W))
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConvLayer:
    kind: Literal["conv"] = "conv"
    out_channels: int = 32
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1          # default = kernel_size // 2  ("same" for 3x3)
    activation: str = "relu"

    def output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        _C, H, W = input_shape
        H_out = math.floor(
            (H + 2 * self.padding - self.kernel_size) / self.stride
        ) + 1
        W_out = math.floor(
            (W + 2 * self.padding - self.kernel_size) / self.stride
        ) + 1
        return (self.out_channels, H_out, W_out)


@dataclass(frozen=True)
class PoolLayer:
    kind: Literal["pool"] = "pool"
    pool_size: int = 2
    stride: int = 2
    pool_type: Literal["max", "avg"] = "max"

    def output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        C, H, W = input_shape
        H_out = math.floor((H - self.pool_size) / self.stride) + 1
        W_out = math.floor((W - self.pool_size) / self.stride) + 1
        return (C, H_out, W_out)


@dataclass(frozen=True)
class FlattenLayer:
    """Collapses (C, H, W) → (C*H*W,).  Must appear exactly once, after all spatial layers."""
    kind: Literal["flatten"] = "flatten"

    def output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        total = 1
        for d in input_shape:
            total *= d
        return (total,)


# ---------------------------------------------------------------------------
# Dense layers  (input / output shape: (N,))
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DenseLayer:
    kind: Literal["dense"] = "dense"
    units: int = 128
    activation: str = "relu"

    def output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return (self.units,)


@dataclass(frozen=True)
class DropoutLayer:
    kind: Literal["dropout"] = "dropout"
    rate: float = 0.5

    def output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return input_shape   # shape is unchanged; only values are masked at runtime


# ---------------------------------------------------------------------------
# Union type — used everywhere else in the codebase
# ---------------------------------------------------------------------------

Layer = Union[ConvLayer, PoolLayer, FlattenLayer, DenseLayer, DropoutLayer]