from dataclasses import dataclass
from typing import Optional

from torch.utils.data import DataLoader

from neuroforge_core.domain.entities.architecture import Architecture
from neuroforge_core.domain.entities.training_run import TrainingRun, TrainingStatus
from neuroforge_core.domain.interfaces.model_trainer import IModelTrainer
from neuroforge_core.domain.value_objects.training_config import TrainingConfig


@dataclass
class TrainModelRequest:
    architecture: Architecture      # domain metadata attached to the run
    config: TrainingConfig
    train_loader: DataLoader
    val_loader: Optional[DataLoader] = None


@dataclass
class TrainModelResponse:
    run: TrainingRun
    succeeded: bool
    best_val_accuracy: Optional[float]
    final_train_loss: Optional[float]


class TrainModelUseCase:
    """
    Application service: create a TrainingRun, delegate to the trainer
    port, and surface summary results.

    Injected via constructor so the adapter is swappable in tests and
    production without touching this class.
    """

    def __init__(self, trainer: IModelTrainer) -> None:
        self._trainer = trainer

    def execute(self, request: TrainModelRequest) -> TrainModelResponse:
        run = TrainingRun(
            architecture=request.architecture,
            config=request.config,
        )

        # Exceptions from the trainer propagate to the caller.
        # run.status will be FAILED before the exception surfaces.
        self._trainer.train(
            run=run,
            train_loader=request.train_loader,
            val_loader=request.val_loader,
        )

        return TrainModelResponse(
            run=run,
            succeeded=run.status == TrainingStatus.COMPLETED,
            best_val_accuracy=run.history.best_val_accuracy,
            final_train_loss=run.history.final_train_loss,
        )