from dataclasses import dataclass
from enum import Enum


class OptimizerType(Enum):
    SGD = "sgd"
    ADAM = "adam"
    ADAMW = "adamw"


@dataclass(frozen=True)
class TrainingConfig:
    """Immutable training hyper-parameters."""

    learning_rate: float = 1e-3        # default so single-kwarg calls work
    epochs: int = 10
    optimizer: OptimizerType = OptimizerType.ADAM   # renamed from optimizer_type
    weight_decay: float = 0.0
    momentum: float = 0.9              # SGD only

    def __post_init__(self) -> None:
        if not isinstance(self.optimizer, OptimizerType):
            raise ValueError(
                f"optimizer must be an OptimizerType member, got {self.optimizer!r}"
            )
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.weight_decay < 0:
            raise ValueError(
                f"weight_decay must be non-negative, got {self.weight_decay}"
            )
        if not (0.0 <= self.momentum <= 1.0):
            raise ValueError(
                f"momentum must be in [0, 1], got {self.momentum}"
            )