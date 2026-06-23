"""
Architecture entity.

Tracks an ordered list of Layer value objects and exposes pure-domain
shape simulation — no PyTorch dependency here.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from neuroforge_core.domain.value_objects.layer import Layer


@dataclass
class Architecture:
    layers: List[Layer]
    num_classes: int
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # ------------------------------------------------------------------
    # Core domain method — was stubbed on Day 1
    # ------------------------------------------------------------------

    def simulate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Compute the output tensor shape by folding input_shape through each layer.

        Pure domain computation — no PyTorch.  Raises ValueError if any
        intermediate shape is degenerate (a spatial dimension ≤ 0).
        """
        shape = input_shape
        for i, layer in enumerate(self.layers):
            shape = layer.output_shape(shape)
            if any(d <= 0 for d in shape):
                raise ValueError(
                    f"Layer {i} ({type(layer).__name__}) produced a degenerate "
                    f"shape {shape} from input {input_shape}."
                )
        return shape

    # ------------------------------------------------------------------
    # Validity guard — used by the generator and search use case
    # ------------------------------------------------------------------

    def is_valid_for_input(self, input_shape: Tuple[int, ...]) -> bool:
        """
        Returns True only if simulate_output_shape() succeeds and produces
        a 1-D output shape (as expected after Flatten + Dense layers).
        """
        try:
            output_shape = self.simulate_output_shape(input_shape)
            return len(output_shape) == 1   # must end with (N,) after dense block
        except (ValueError, TypeError):
            return False

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def layer_count(self) -> int:
        return len(self.layers)

    def __repr__(self) -> str:
        kinds = [type(l).__name__.replace("Layer", "") for l in self.layers]
        return f"Architecture(id={self.id[:8]}, layers={kinds}, classes={self.num_classes})"