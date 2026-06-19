"""
TrainingConfig value object.

Single source of truth for one training run's hyperparameters. Passed
unchanged from RunExperimentUseCase (Day 7) down to PyTorchTrainer (Day 6)
— no hyperparameter should ever be hardcoded inside the trainer itself.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 10
    learning_rate: float = 1e-3
    batch_size: int = 128
    optimizer: str = "adam"
    device: str = "cpu"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.optimizer not in {"adam", "sgd"}:
            raise ValueError(f"optimizer must be 'adam' or 'sgd', got {self.optimizer!r}")