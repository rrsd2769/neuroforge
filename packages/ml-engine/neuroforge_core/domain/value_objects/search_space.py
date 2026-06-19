"""
SearchSpace value object.

Defines the constraints RandomArchitectureGenerator (Day 4) samples
within. Deliberately minimal for Week 1 — fixed ranges, no parameter
priors. Week 2 extends this with the choice/range parameter types Ax
Platform's Bayesian Optimization needs, without breaking this Day 1 shape.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchSpace:
    """Constraints for generating CNN architectures."""

    input_shape: tuple[int, int, int] = (3, 32, 32)
    num_classes: int = 10
    min_depth: int = 2
    max_depth: int = 6
    channel_choices: tuple[int, ...] = (16, 32, 64, 128)
    allowed_layer_types: tuple[str, ...] = ("conv", "pool", "activation")

    def __post_init__(self) -> None:
        if self.min_depth < 1:
            raise ValueError(f"min_depth must be >= 1, got {self.min_depth}")
        if self.max_depth < self.min_depth:
            raise ValueError(
                f"max_depth ({self.max_depth}) must be >= min_depth ({self.min_depth})"
            )
        if not self.channel_choices:
            raise ValueError("channel_choices must not be empty")