from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class EpochMetrics:
    """Snapshot of training (and optional validation) results for one epoch."""

    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None

    def __post_init__(self) -> None:
        if self.epoch < 1:
            raise ValueError(f"epoch must be >= 1, got {self.epoch}")
        if not (0.0 <= self.train_accuracy <= 1.0):
            raise ValueError(
                f"train_accuracy must be in [0, 1], got {self.train_accuracy}"
            )
        if self.val_accuracy is not None and not (0.0 <= self.val_accuracy <= 1.0):
            raise ValueError(
                f"val_accuracy must be in [0, 1], got {self.val_accuracy}"
            )


@dataclass
class TrainingHistory:
    """Accumulates per-epoch metrics across a complete training run."""

    epochs: List[EpochMetrics] = field(default_factory=list)

    def record(self, metrics: EpochMetrics) -> None:
        self.epochs.append(metrics)

    @property
    def best_val_accuracy(self) -> Optional[float]:
        candidates = [
            e.val_accuracy for e in self.epochs if e.val_accuracy is not None
        ]
        return max(candidates) if candidates else None

    @property
    def final_train_loss(self) -> Optional[float]:
        return self.epochs[-1].train_loss if self.epochs else None

    def __len__(self) -> int:
        return len(self.epochs)