from dataclasses import dataclass, field
from typing import Dict


@dataclass
class EvaluationMetrics:
    """
    Results of a single evaluation run.

    Not frozen because per_class_accuracy is a dict (mutable container).
    Treat instances as read-only after construction.
    """

    accuracy: float                          # Top-1 accuracy in [0, 1]
    top_k_accuracy: float                    # Top-k accuracy in [0, 1]
    average_loss: float                      # Mean cross-entropy loss
    per_class_accuracy: Dict[int, float]     # {class_id: accuracy in [0,1]}
    total_samples: int
    correct_predictions: int
    k: int = 5                               # The k used for top_k_accuracy

    def __post_init__(self) -> None:
        if not 0.0 <= self.accuracy <= 1.0:
            raise ValueError(
                f"accuracy must be in [0, 1], got {self.accuracy}"
            )
        if not 0.0 <= self.top_k_accuracy <= 1.0:
            raise ValueError(
                f"top_k_accuracy must be in [0, 1], got {self.top_k_accuracy}"
            )
        if self.total_samples <= 0:
            raise ValueError(
                f"total_samples must be positive, got {self.total_samples}"
            )

    @property
    def accuracy_percent(self) -> float:
        return self.accuracy * 100.0

    @property
    def top_k_accuracy_percent(self) -> float:
        return self.top_k_accuracy * 100.0

    def summary(self) -> str:
        lines = [
            f"Accuracy:          {self.accuracy_percent:.2f}%",
            f"Top-{self.k} Accuracy:    {self.top_k_accuracy_percent:.2f}%",
            f"Average Loss:      {self.average_loss:.4f}",
            f"Total Samples:     {self.total_samples:,}",
            f"Correct:           {self.correct_predictions:,}",
        ]
        if self.per_class_accuracy:
            lines.append("Per-Class Accuracy:")
            for cls_id, acc in sorted(self.per_class_accuracy.items()):
                bar = "█" * int(acc * 20)
                lines.append(f"  Class {cls_id:2d}: {acc * 100:5.1f}%  {bar}")
        return "\n".join(lines)