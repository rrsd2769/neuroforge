import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from neuroforge_core.domain.value_objects.evaluation_config import EvaluationConfig
from neuroforge_core.infrastructure.adapters.pytorch_evaluator import PyTorchEvaluator


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def evaluator() -> PyTorchEvaluator:
    return PyTorchEvaluator()


@pytest.fixture
def simple_model() -> nn.Module:
    """Minimal model: flatten → linear. Deterministic on eval mode."""
    torch.manual_seed(42)
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 10),
    )


@pytest.fixture
def test_loader() -> DataLoader:
    """Fake 80-sample dataset, 10 classes, CIFAR-10 image shape."""
    torch.manual_seed(0)
    inputs = torch.randn(80, 3, 32, 32)
    labels = torch.randint(0, 10, (80,))
    return DataLoader(TensorDataset(inputs, labels), batch_size=20)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_evaluate_returns_evaluation_metrics(evaluator, simple_model, test_loader):
    from neuroforge_core.domain.value_objects.evaluation_metrics import EvaluationMetrics
    config = EvaluationConfig(device="cpu")
    metrics = evaluator.evaluate(simple_model, test_loader, config)
    assert isinstance(metrics, EvaluationMetrics)


def test_accuracy_in_valid_range(evaluator, simple_model, test_loader):
    config = EvaluationConfig(device="cpu")
    metrics = evaluator.evaluate(simple_model, test_loader, config)
    assert 0.0 <= metrics.accuracy <= 1.0


def test_top_k_accuracy_gte_top1(evaluator, simple_model, test_loader):
    config = EvaluationConfig(device="cpu", top_k=5)
    metrics = evaluator.evaluate(simple_model, test_loader, config)
    assert metrics.top_k_accuracy >= metrics.accuracy


def test_total_samples_correct(evaluator, simple_model, test_loader):
    config = EvaluationConfig(device="cpu")
    metrics = evaluator.evaluate(simple_model, test_loader, config)
    assert metrics.total_samples == 80


def test_per_class_accuracy_all_in_range(evaluator, simple_model, test_loader):
    config = EvaluationConfig(device="cpu")
    metrics = evaluator.evaluate(simple_model, test_loader, config)
    assert len(metrics.per_class_accuracy) == 10
    for acc in metrics.per_class_accuracy.values():
        assert 0.0 <= acc <= 1.0


def test_correct_predictions_consistent_with_accuracy(
    evaluator, simple_model, test_loader
):
    config = EvaluationConfig(device="cpu")
    metrics = evaluator.evaluate(simple_model, test_loader, config)
    expected = metrics.correct_predictions / metrics.total_samples
    assert abs(metrics.accuracy - expected) < 1e-6


def test_average_loss_positive(evaluator, simple_model, test_loader):
    config = EvaluationConfig(device="cpu")
    metrics = evaluator.evaluate(simple_model, test_loader, config)
    assert metrics.average_loss > 0.0


def test_deterministic_on_eval_mode(evaluator, simple_model, test_loader):
    """Same model + same data should produce same metrics twice."""
    config = EvaluationConfig(device="cpu")
    m1 = evaluator.evaluate(simple_model, test_loader, config)
    m2 = evaluator.evaluate(simple_model, test_loader, config)
    assert m1.accuracy == m2.accuracy
    assert m1.average_loss == m2.average_loss