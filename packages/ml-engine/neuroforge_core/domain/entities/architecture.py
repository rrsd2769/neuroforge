"""
Architecture domain entities.

Represents a neural network as an ordered stack of LayerSpecs — the single
representation shared by architecture generation (Day 4), model
construction (Day 5/ModelFactory), and experiment logging (Day 7). No
PyTorch import here: ModelFactory is the only place this ever becomes an
actual nn.Module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ConvLayerSpec:
    """A 2D convolution layer."""

    out_channels: int = 32
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    layer_type: Literal["conv"] = field(default="conv", init=False)

    def __post_init__(self) -> None:
        if self.out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {self.out_channels}")
        if self.kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {self.kernel_size}")


@dataclass(frozen=True)
class PoolLayerSpec:
    """A 2D pooling layer."""

    pool_kind: Literal["max", "avg"] = "max"
    kernel_size: int = 2
    stride: int = 2
    layer_type: Literal["pool"] = field(default="pool", init=False)


@dataclass(frozen=True)
class ActivationSpec:
    """A nonlinearity applied in-place in the layer stack."""

    kind: Literal["relu", "gelu"] = "relu"
    layer_type: Literal["activation"] = field(default="activation", init=False)


@dataclass(frozen=True)
class LinearLayerSpec:
    """A fully-connected layer — used for the classifier head."""

    out_features: int = 128
    layer_type: Literal["linear"] = field(default="linear", init=False)

    def __post_init__(self) -> None:
        if self.out_features <= 0:
            raise ValueError(f"out_features must be positive, got {self.out_features}")


LayerSpec = ConvLayerSpec | PoolLayerSpec | ActivationSpec | LinearLayerSpec

# Registry used by Architecture.from_dict() to reconstruct the correct
# LayerSpec subtype from its serialized "layer_type" tag.
_LAYER_TYPE_REGISTRY: dict[str, type] = {
    "conv": ConvLayerSpec,
    "pool": PoolLayerSpec,
    "activation": ActivationSpec,
    "linear": LinearLayerSpec,
}


@dataclass
class Architecture:
    """
    A candidate neural network, expressed as an ordered list of LayerSpecs.

    Populated by ArchitectureGeneratorPort implementations (Day 4), read by
    ModelFactory (Day 5), and serialized by ExperimentTracker (Day 7).
    """

    layers: list[LayerSpec] = field(default_factory=list)
    input_shape: tuple[int, int, int] = (3, 32, 32)
    num_classes: int = 10
    architecture_id: str | None = None

    def add_layer(self, layer: LayerSpec) -> None:
        """Appends a layer spec to the architecture's layer stack."""
        self.layers.append(layer)

    def to_dict(self) -> dict:
        """Serializes the architecture to a JSON-compatible dict."""
        return {
            "architecture_id": self.architecture_id,
            "input_shape": list(self.input_shape),
            "num_classes": self.num_classes,
            "layers": [asdict(layer) for layer in self.layers],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Architecture":
        """Reconstructs an Architecture from a dict produced by to_dict()."""
        layers: list[LayerSpec] = []
        for layer_data in data["layers"]:
            layer_type = layer_data["layer_type"]
            layer_cls = _LAYER_TYPE_REGISTRY.get(layer_type)
            if layer_cls is None:
                raise ValueError(f"Unknown layer_type: {layer_type!r}")
            kwargs = {k: v for k, v in layer_data.items() if k != "layer_type"}
            layers.append(layer_cls(**kwargs))
        return cls(
            layers=layers,
            input_shape=tuple(data["input_shape"]),
            num_classes=data["num_classes"],
            architecture_id=data.get("architecture_id"),
        )

    def simulate_output_shape(self) -> tuple[int, ...]:
        """
        Symbolically walks the layer stack to compute the final output
        tensor shape, without building any PyTorch module.

        NOTE: Full shape-flow simulation is implemented on Day 4 as part
        of ArchitectureValidator. This method's signature is fixed now so
        Day 4/5 code can depend on it immediately without an interface
        change later.
        """
        raise NotImplementedError(
            "simulate_output_shape() is implemented on Day 4 "
            "(ArchitectureValidator) — see Week 1 roadmap."
        )