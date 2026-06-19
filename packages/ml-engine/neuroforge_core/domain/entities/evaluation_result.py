"""
Training-metric and evaluation-result domain entities.

TrainingMetrics is recorded once per epoch by the Trainer (Day 6).
EvaluationResult is recorded once per experiment by the Evaluator (Day 7),
computed against a held-out test set never seen during training — kept
separate from TrainingMetrics so experiment logs can't conflate the two.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainingMetrics:
    """One epoch's worth of training-loop metrics."""

    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    epoch_duration_seconds: float

    def to_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "train_accuracy": self.train_accuracy,
            "val_loss": self.val_loss,
            "val_accuracy": self.val_accuracy,
            "epoch_duration_seconds": self.epoch_duration_seconds,
        }


@dataclass
class EvaluationResult:
    """
    Final, held-out test-set evaluation of a trained model.

    `confusion_matrix` is a num_classes x num_classes nested list so it
    serializes directly to JSON without a custom encoder.
    """

    accuracy: float
    per_class_precision: list[float] = field(default_factory=list)
    per_class_recall: list[float] = field(default_factory=list)
    confusion_matrix: list[list[int]] = field(default_factory=list)
    inference_latency_ms: float | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.accuracy <= 1.0):
            raise ValueError(f"accuracy must be in [0,1], got {self.accuracy}")

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "per_class_precision": self.per_class_precision,
            "per_class_recall": self.per_class_recall,
            "confusion_matrix": self.confusion_matrix,
            "inference_latency_ms": self.inference_latency_ms,
        }