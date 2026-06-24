import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import MagicMock

from neuroforge_core.domain.entities.architecture import Architecture
from neuroforge_core.domain.entities.training_run import TrainingRun, TrainingStatus
from neuroforge_core.domain.value_objects.training_config import (
    OptimizerType,
    TrainingConfig,
)
from neuroforge_core.infrastructure.training.pytorch_trainer import PyTorchTrainer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_arch():
    return MagicMock(spec=Architecture)


@pytest.fixture
def tiny_loader():
    """20 samples, 4 features, 3 classes — 2 batches of 10."""
    inputs = torch.randn(20, 4)
    targets = torch.randint(0, 3, (20,))
    return DataLoader(TensorDataset(inputs, targets), batch_size=10)


@pytest.fixture
def simple_config():
    return TrainingConfig(learning_rate=0.01, epochs=2)


@pytest.fixture
def tiny_model():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_train_returns_completed_run(mock_arch, simple_config, tiny_loader, tiny_model):
    run = TrainingRun(architecture=mock_arch, config=simple_config)
    trainer = PyTorchTrainer(tiny_model)
    result = trainer.train(run, tiny_loader)
    assert result.status == TrainingStatus.COMPLETED


def test_history_length_matches_epochs(mock_arch, simple_config, tiny_loader, tiny_model):
    run = TrainingRun(architecture=mock_arch, config=simple_config)
    trainer = PyTorchTrainer(tiny_model)
    trainer.train(run, tiny_loader)
    assert len(run.history) == simple_config.epochs


def test_val_loader_populates_val_metrics(mock_arch, simple_config, tiny_loader, tiny_model):
    run = TrainingRun(architecture=mock_arch, config=simple_config)
    trainer = PyTorchTrainer(tiny_model)
    trainer.train(run, tiny_loader, val_loader=tiny_loader)
    for epoch_m in run.history.epochs:
        assert epoch_m.val_loss is not None
        assert epoch_m.val_accuracy is not None


def test_no_val_loader_leaves_val_metrics_none(mock_arch, simple_config, tiny_loader, tiny_model):
    run = TrainingRun(architecture=mock_arch, config=simple_config)
    trainer = PyTorchTrainer(tiny_model)
    trainer.train(run, tiny_loader)
    for epoch_m in run.history.epochs:
        assert epoch_m.val_loss is None
        assert epoch_m.val_accuracy is None


@pytest.mark.parametrize("opt_type", list(OptimizerType))
def test_all_optimizer_types_complete(opt_type, mock_arch, tiny_loader):
    model = nn.Linear(4, 3)
    config = TrainingConfig(learning_rate=0.01, epochs=1, optimizer=opt_type)
    run = TrainingRun(architecture=mock_arch, config=config)
    trainer = PyTorchTrainer(model)
    trainer.train(run, tiny_loader)
    assert run.status == TrainingStatus.COMPLETED


def test_accuracy_between_zero_and_one(mock_arch, simple_config, tiny_loader, tiny_model):
    run = TrainingRun(architecture=mock_arch, config=simple_config)
    trainer = PyTorchTrainer(tiny_model)
    trainer.train(run, tiny_loader)
    for epoch_m in run.history.epochs:
        assert 0.0 <= epoch_m.train_accuracy <= 1.0