"""
layer_parser.py
===============
Converts validated Pydantic layer schemas (from api/schemas.py) into
concrete domain Layer dataclasses (from neuroforge_core.domain.value_objects.layer).

One function per layer type — no magic, no metaprogramming.
Uses verified field names from the actual layer.py dataclasses:
  ConvLayer:   out_channels, kernel_size, stride, padding, activation
  PoolLayer:   pool_size, stride, pool_type
  FlattenLayer: (no fields)
  DenseLayer:  units, activation
  DropoutLayer: rate
"""
from __future__ import annotations

from neuroforge_core.domain.entities.architecture import Architecture
from neuroforge_core.domain.value_objects.layer import (
    ConvLayer,
    DenseLayer,
    DropoutLayer,
    FlattenLayer,
    Layer,
    PoolLayer,
)
from neuroforge_core.domain.value_objects.training_config import (
    OptimizerType,
    TrainingConfig,
)

from api.schemas import (
    ArchitectureIn,
    ConvLayerIn,
    DenseLayerIn,
    DropoutLayerIn,
    FlattenLayerIn,
    LayerIn,
    PoolLayerIn,
    TrainingConfigIn,
)


def parse_layer(layer_in: LayerIn) -> Layer:
    """Convert a validated Pydantic layer schema to a domain Layer dataclass."""
    if isinstance(layer_in, ConvLayerIn):
        return ConvLayer(
            out_channels=layer_in.out_channels,
            kernel_size=layer_in.kernel_size,
            stride=layer_in.stride,
            padding=layer_in.padding,
            activation=layer_in.activation,
        )
    if isinstance(layer_in, PoolLayerIn):
        return PoolLayer(
            pool_size=layer_in.pool_size,
            stride=layer_in.stride,
            pool_type=layer_in.pool_type,
        )
    if isinstance(layer_in, FlattenLayerIn):
        return FlattenLayer()
    if isinstance(layer_in, DenseLayerIn):
        return DenseLayer(
            units=layer_in.units,
            activation=layer_in.activation,
        )
    if isinstance(layer_in, DropoutLayerIn):
        return DropoutLayer(
            rate=layer_in.rate,
        )
    raise ValueError(f"Unhandled layer type: {type(layer_in).__name__}")


def parse_architecture(arch_in: ArchitectureIn) -> Architecture:
    """Convert an ArchitectureIn schema to a domain Architecture entity."""
    return Architecture(
        layers=[parse_layer(layer) for layer in arch_in.layers],
        num_classes=arch_in.num_classes,
    )


def parse_training_config(config_in: TrainingConfigIn) -> TrainingConfig:
    """Convert a TrainingConfigIn schema to a domain TrainingConfig value object."""
    try:
        optimizer_type = OptimizerType(config_in.optimizer.lower())
    except ValueError:
        valid = [e.value for e in OptimizerType]
        raise ValueError(
            f"Unknown optimizer {config_in.optimizer!r}. Valid options: {valid}"
        )
    return TrainingConfig(
        epochs=config_in.epochs,
        learning_rate=config_in.learning_rate,
        optimizer=optimizer_type,
        weight_decay=config_in.weight_decay,
        momentum=config_in.momentum,
    )