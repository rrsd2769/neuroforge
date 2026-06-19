"""
TrainerPort.

Abstract contract for anything that can run a training loop given a
model, data loaders, and a TrainingConfig. Implemented by PyTorchTrainer
(Day 6).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from neuroforge_core.domain.entities.evaluation_result import TrainingMetrics
from neuroforge_core.domain.value_objects.training_config import TrainingConfig


class TrainerPort(ABC):
    @abstractmethod
    def train(
        self,
        model: Any,
        train_loader: Any,
        val_loader: Any,
        config: TrainingConfig,
    ) -> list[TrainingMetrics]:
        """
        Runs the full training loop and returns per-epoch metrics.


model`, `train_loader`, and `val_loader` are typed as `Any`
deliberately — torch has no place being referenced inside this
layer at all. The concrete PyTorchTrainer implementation narrows
these to nn.Module / DataLoader internally.
"""


        raise NotImplementedError