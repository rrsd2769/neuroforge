from abc import ABC, abstractmethod
from typing import Optional

from torch.utils.data import DataLoader

from neuroforge_core.domain.entities.training_run import TrainingRun


class IModelTrainer(ABC):
    """
    Port: contract for executing a training loop.

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
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> TrainingRun:
        """Execute the training loop; return the mutated run."""
        ...