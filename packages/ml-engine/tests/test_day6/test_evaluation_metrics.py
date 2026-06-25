import pytest
from neuroforge_core.domain.value_objects.evaluation_metrics import EvaluationMetrics


def _make_metrics(**overrides) -> EvaluationMetrics:
    defaults = dict(
        accuracy=0.75,
        top_k_accuracy=0.90,
        average_loss=0.50,
        per_class_accuracy={i: 0.75 for i in range(10)},
        total_samples=1000,
        correct_predictions=750,
        k=5,
    )
    defaults.update(overrides)
    return EvaluationMetrics(**defaults)


def test_valid_construction():
    m = _make_metrics()
    assert m.accuracy == 0.75
    assert m.total_samples == 1000


def test_accuracy_percent():
    m = _make_metrics(accuracy=0.853)
    assert abs(m.accuracy_percent - 85.3) < 1e-9


def test_top_k_accuracy_percent():
    m = _make_metrics(top_k_accuracy=0.95)
    assert abs(m.top_k_accuracy_percent - 95.0) < 1e-9


def test_invalid_accuracy_above_one():
    with pytest.raises(ValueError, match="accuracy"):
        _make_metrics(accuracy=1.1)


def test_invalid_accuracy_below_zero():
    with pytest.raises(ValueError, match="accuracy"):
        _make_metrics(accuracy=-0.1)


def test_invalid_top_k_accuracy():
    with pytest.raises(ValueError, match="top_k_accuracy"):
        _make_metrics(top_k_accuracy=1.5)


def test_invalid_total_samples_zero():
    with pytest.raises(ValueError, match="total_samples"):
        _make_metrics(total_samples=0)


def test_summary_contains_accuracy():
    m = _make_metrics(accuracy=0.85, top_k_accuracy=0.95)
    summary = m.summary()
    assert "85.00%" in summary
    assert "95.00%" in summary


def test_summary_contains_loss():
    m = _make_metrics(average_loss=0.4200)
    assert "0.4200" in m.summary()


def test_summary_contains_per_class():
    m = _make_metrics()
    summary = m.summary()
    assert "Class  0" in summary