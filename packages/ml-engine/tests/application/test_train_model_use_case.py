import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import MagicMock

from neuroforge_core.domain.entities.architecture import Architecture
from neuroforge_core.domain.entities.training_run import TrainingRun, TrainingStatus
from neuroforge_core.domain.interfaces.model_trainer import IModelTrainer
from neuroforge_core.domain.value_objects.training_config import TrainingConfig
from neuroforge_core.domain.value_objects.training_metrics import EpochMetrics
from neuroforge_core.application.train_model import (
    TrainModelRequest,
    TrainModelResponse,
    TrainModelUseCase,
)


# ---------------------------------------------------------------------------
# Fake implementations — no PyTorch training needed for use-case tests
# ---------------------------------------------------------------------------

class _FakeTrainer(IModelTrainer):
    """Completes immediately with canned metrics."""

    def __init__(self, include_val: bool = False) -> None:
        self._include_val = include_val

    def train(self, run, train_loader, val_loader=None):
        run.start()
        for epoch in range(1, run.config.epochs + 1):
            run.history.record(
                EpochMetrics(
                    epoch=epoch,
                    train_loss=1.0 - epoch * 0.1,
                    train_accuracy=0.4 + epoch * 0.1,
                    val_loss=0.5 if self._include_val else None,
                    val_accuracy=0.7 if self._include_val else None,
                )
            )
        run.complete()
        return run


class _RaisingTrainer(IModelTrainer):
    """Simulates a crash mid-training."""

    def train(self, run, train_loader, val_loader=None):
        run.start()
        run.fail("CUDA out of memory")
        raise RuntimeError("CUDA out of memory")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_arch():
    return MagicMock(spec=Architecture)


@pytest.fixture
def config():
    return TrainingConfig(learning_rate=0.01, epochs=3)


@pytest.fixture
def dummy_loader():
    inputs = torch.randn(8, 4)
    targets = torch.randint(0, 3, (8,))
    return DataLoader(TensorDataset(inputs, targets), batch_size=8)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_execute_returns_response_object(mock_arch, config, dummy_loader):
    use_case = TrainModelUseCase(_FakeTrainer())
    request = TrainModelRequest(architecture=mock_arch, config=config,
                                train_loader=dummy_loader)
    response = use_case.execute(request)
    assert isinstance(response, TrainModelResponse)


def test_succeeded_flag_true_on_completion(mock_arch, config, dummy_loader):
    use_case = TrainModelUseCase(_FakeTrainer())
    request = TrainModelRequest(architecture=mock_arch, config=config,
                                train_loader=dummy_loader)
    response = use_case.execute(request)
    assert response.succeeded is True


def test_run_has_correct_config(mock_arch, config, dummy_loader):
    use_case = TrainModelUseCase(_FakeTrainer())
    request = TrainModelRequest(architecture=mock_arch, config=config,
                                train_loader=dummy_loader)
    response = use_case.execute(request)
    assert response.run.config is config
    assert len(response.run.history) == config.epochs


def test_best_val_accuracy_populated_with_val_loader(mock_arch, config, dummy_loader):
    use_case = TrainModelUseCase(_FakeTrainer(include_val=True))
    request = TrainModelRequest(architecture=mock_arch, config=config,
                                train_loader=dummy_loader,
                                val_loader=dummy_loader)
    response = use_case.execute(request)
    assert response.best_val_accuracy == pytest.approx(0.7)


def test_exception_propagates_from_trainer(mock_arch, config, dummy_loader):
    use_case = TrainModelUseCase(_RaisingTrainer())
    request = TrainModelRequest(architecture=mock_arch, config=config,
                                train_loader=dummy_loader)
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        use_case.execute(request)