from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn

from neuroforge_core.domain.entities.architecture import Architecture
from neuroforge_core.domain.interfaces.i_model_compiler import IModelCompiler
from neuroforge_core.domain.value_objects.layer import (
    ConvLayer,
    DenseLayer,
    DropoutLayer,
    FlattenLayer,
    PoolLayer,
)


class ModelCompilationError(Exception):
    """Raised when an Architecture cannot be compiled into a valid model."""
    pass


_ACTIVATION_MAP = {
    "relu":       nn.ReLU,
    "sigmoid":    nn.Sigmoid,
    "tanh":       nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
    "gelu":       nn.GELU,
}


class PyTorchModelCompiler(IModelCompiler):
    """
    Adapter: compiles domain Architecture objects into torch.nn.Sequential models.

    Layer mapping:
        ConvLayer    → nn.Conv2d  + optional activation
        PoolLayer    → nn.MaxPool2d / nn.AvgPool2d
        FlattenLayer → nn.Flatten
        DenseLayer   → nn.Linear  + optional activation
        DropoutLayer → nn.Dropout

    Shape tracking is performed statically during compilation.
    Auto-inserts nn.Flatten before the first DenseLayer if missing.
    Auto-appends a final nn.Linear(→ num_classes) if needed.
    """

    def compile(
        self,
        architecture: Architecture,
        input_channels: int = 3,
        num_classes: Optional[int] = None,
        input_spatial: Tuple[int, int] = (32, 32),
    ) -> nn.Module:
        if not architecture.layers:
            raise ModelCompilationError(
                f"Architecture '{architecture.id}' has no layers."
            )

        # Architecture.num_classes is authoritative; parameter is a fallback
        effective_classes = num_classes if num_classes is not None else architecture.num_classes

        modules: List[nn.Module] = []
        current_channels: int = input_channels
        current_h: int = input_spatial[0]
        current_w: int = input_spatial[1]
        is_flattened: bool = False
        current_features: Optional[int] = None

        for idx, layer in enumerate(architecture.layers):

            # ── ConvLayer ──────────────────────────────────────────────
            if isinstance(layer, ConvLayer):
                if is_flattened:
                    raise ModelCompilationError(
                        f"Layer {idx}: ConvLayer cannot appear after Flatten."
                    )
                modules.append(nn.Conv2d(
                    current_channels,
                    layer.out_channels,
                    layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                ))
                current_channels = layer.out_channels
                current_h = (
                    (current_h + 2 * layer.padding - layer.kernel_size)
                    // layer.stride + 1
                )
                current_w = (
                    (current_w + 2 * layer.padding - layer.kernel_size)
                    // layer.stride + 1
                )
                self._check_spatial(current_h, current_w, idx, "ConvLayer")
                self._add_activation(modules, layer.activation)

            # ── PoolLayer ──────────────────────────────────────────────
            elif isinstance(layer, PoolLayer):
                if is_flattened:
                    raise ModelCompilationError(
                        f"Layer {idx}: PoolLayer cannot appear after Flatten."
                    )
                pool_cls = nn.MaxPool2d if layer.pool_type == "max" else nn.AvgPool2d
                modules.append(pool_cls(layer.pool_size, stride=layer.stride))
                current_h = (current_h - layer.pool_size) // layer.stride + 1
                current_w = (current_w - layer.pool_size) // layer.stride + 1
                self._check_spatial(current_h, current_w, idx, "PoolLayer")

            # ── FlattenLayer ───────────────────────────────────────────
            elif isinstance(layer, FlattenLayer):
                modules.append(nn.Flatten())
                is_flattened = True
                current_features = current_channels * current_h * current_w

            # ── DenseLayer ─────────────────────────────────────────────
            elif isinstance(layer, DenseLayer):
                if not is_flattened:
                    # Auto-insert Flatten
                    modules.append(nn.Flatten())
                    is_flattened = True
                    current_features = current_channels * current_h * current_w
                modules.append(nn.Linear(current_features, layer.units))
                current_features = layer.units
                self._add_activation(modules, layer.activation)

            # ── DropoutLayer ───────────────────────────────────────────
            elif isinstance(layer, DropoutLayer):
                modules.append(nn.Dropout(p=layer.rate))

        # ── Auto output head ───────────────────────────────────────────
        if not is_flattened:
            modules.append(nn.Flatten())
            current_features = current_channels * current_h * current_w

        if current_features != effective_classes:
            modules.append(nn.Linear(current_features, effective_classes))

        return nn.Sequential(*modules)

    def is_valid_architecture(
        self,
        architecture: Architecture,
        input_channels: int = 3,
        num_classes: Optional[int] = None,
    ) -> bool:
        """Safe for use in search loops — never raises."""
        try:
            effective_classes = (
                num_classes if num_classes is not None else architecture.num_classes
            )
            model = self.compile(architecture, input_channels, effective_classes)
            model.eval()
            with torch.no_grad():
                dummy = torch.zeros(2, input_channels, 32, 32)
                out = model(dummy)
            return out.shape == (2, effective_classes)
        except Exception:
            return False

    # ── Private helpers ────────────────────────────────────────────────

    @staticmethod
    def _check_spatial(h: int, w: int, layer_idx: int, layer_name: str) -> None:
        if h <= 0 or w <= 0:
            raise ModelCompilationError(
                f"Layer {layer_idx} ({layer_name}) produced invalid spatial "
                f"dimensions ({h}, {w}). Reduce pooling or increase padding."
            )

    @staticmethod
    def _add_activation(modules: List[nn.Module], activation: str) -> None:
        if not activation or activation.lower() in ("none", ""):
            return
        act_cls = _ACTIVATION_MAP.get(activation.lower())
        if act_cls is not None:
            modules.append(act_cls())