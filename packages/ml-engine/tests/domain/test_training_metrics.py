import pytest
from neuroforge_core.domain.value_objects.training_metrics import (
    EpochMetrics,
    TrainingHistory,
)


def test_epoch_metrics_basic_creation():
    m = EpochMetrics(epoch=1, train_loss=0.5, train_accuracy=0.8)
    assert m.epoch == 1
    assert m.val_loss is None
    assert m.val_accuracy is None


def test_epoch_metrics_with_val_fields():
    m = EpochMetrics(epoch=2, train_loss=0.3, train_accuracy=0.9,
                     val_loss=0.4, val_accuracy=0.85)
    assert m.val_accuracy == 0.85


def test_epoch_metrics_invalid_epoch():
    with pytest.raises(ValueError, match="epoch"):
        EpochMetrics(epoch=0, train_loss=0.5, train_accuracy=0.5)


def test_epoch_metrics_accuracy_out_of_range():
    with pytest.raises(ValueError, match="train_accuracy"):
        EpochMetrics(epoch=1, train_loss=0.5, train_accuracy=1.5)


def test_history_record_increments_length():
    h = TrainingHistory()
    assert len(h) == 0
    h.record(EpochMetrics(epoch=1, train_loss=0.5, train_accuracy=0.7))
    h.record(EpochMetrics(epoch=2, train_loss=0.3, train_accuracy=0.85))
    assert len(h) == 2


def test_history_best_val_accuracy():
    h = TrainingHistory()
    h.record(EpochMetrics(epoch=1, train_loss=0.5, train_accuracy=0.7,
                           val_loss=0.6, val_accuracy=0.65))
    h.record(EpochMetrics(epoch=2, train_loss=0.3, train_accuracy=0.85,
                           val_loss=0.35, val_accuracy=0.82))
    assert h.best_val_accuracy == pytest.approx(0.82)


def test_history_best_val_accuracy_none_when_no_val():
    h = TrainingHistory()
    h.record(EpochMetrics(epoch=1, train_loss=0.5, train_accuracy=0.7))
    assert h.best_val_accuracy is None


def test_history_final_train_loss():
    h = TrainingHistory()
    h.record(EpochMetrics(epoch=1, train_loss=0.8, train_accuracy=0.6))
    h.record(EpochMetrics(epoch=2, train_loss=0.4, train_accuracy=0.8))
    assert h.final_train_loss == pytest.approx(0.4)