from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from neuroforge_core.domain.entities.training_run import TrainingRun


class IModelTrainer(ABC):
    """
    Port: contract for executing a training loop.

    train_loader / val_loader are typed Any deliberately — torch has
    no place being referenced inside the domain layer. The concrete
    PyTorchTrainer adapter narrows these to DataLoader internally.

    Implementors must:
    - Call run.start() before touching the model.
    - Append one EpochMetrics per epoch to run.history.
    - Call run.complete() on success or run.fail(msg) on error.
    - Re-raise the original exception after calling run.fail().
    """

    @abstractmethod
    def train(
        self,
        run: TrainingRun,
        train_loader: Any,
        val_loader: Optional[Any] = None,
    ) -> TrainingRun:
        """Execute the training loop; return the mutated run."""
        ...