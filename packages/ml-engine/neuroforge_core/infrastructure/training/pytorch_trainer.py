from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from neuroforge_core.domain.entities.training_run import TrainingRun
from neuroforge_core.domain.interfaces.model_trainer import IModelTrainer
from neuroforge_core.domain.value_objects.training_config import OptimizerType
from neuroforge_core.domain.value_objects.training_metrics import EpochMetrics


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_optimizer(
    model: nn.Module,
    config,
) -> torch.optim.Optimizer:
    if config.optimizer == OptimizerType.SGD:
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == OptimizerType.ADAM:
        return torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == OptimizerType.ADAMW:
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer type: {config.optimizer_type}")


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    training: bool,
) -> Tuple[float, float]:
    """
    One forward (and optional backward) pass over ``loader``.

    Args:
        training: When True, runs backprop via ``optimizer``.
                  When False, wraps forward pass in ``torch.no_grad()``;
                  ``optimizer`` is unused and should be None.

    Returns:
        (average_loss, accuracy) over the full loader.
    """
    model.train(training)
    total_loss = 0.0
    correct = 0
    n = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        if training:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()  # type: ignore[union-attr]
            loss.backward()
            optimizer.step()  # type: ignore[union-attr]
        else:
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        predicted = outputs.argmax(dim=1)
        correct += predicted.eq(targets).sum().item()
        n += inputs.size(0)

    if n == 0:
        return 0.0, 0.0
    return total_loss / n, correct / n


# ---------------------------------------------------------------------------
# Public adapter
# ---------------------------------------------------------------------------

class PyTorchTrainer(IModelTrainer):
    """
    Concrete training adapter.

    The caller is responsible for building the ``nn.Module`` before
    constructing this trainer. Architecture → nn.Module compilation
    will be handled by a dedicated ModelCompiler (Day 6).

    Usage::

        model = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 10))
        trainer = PyTorchTrainer(model)
        use_case = TrainModelUseCase(trainer)
        response = use_case.execute(request)
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    def train(
        self,
        run: TrainingRun,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> TrainingRun:
        run.start()

        try:
            config = run.config
            criterion = nn.CrossEntropyLoss()
            optimizer = _build_optimizer(self.model, config)

            for epoch in range(1, config.epochs + 1):
                train_loss, train_acc = _run_epoch(
                    self.model,
                    train_loader,
                    criterion,
                    optimizer,
                    self.device,
                    training=True,
                )

                val_loss: Optional[float] = None
                val_acc: Optional[float] = None
                if val_loader is not None:
                    val_loss, val_acc = _run_epoch(
                        self.model,
                        val_loader,
                        criterion,
                        None,
                        self.device,
                        training=False,
                    )

                run.history.record(
                    EpochMetrics(
                        epoch=epoch,
                        train_loss=train_loss,
                        train_accuracy=train_acc,
                        val_loss=val_loss,
                        val_accuracy=val_acc,
                    )
                )

            run.complete()

        except Exception as exc:
            run.fail(str(exc))
            raise

        return run