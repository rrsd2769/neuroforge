import pytest
from neuroforge_core.domain.value_objects.training_config import (
    OptimizerType,
    TrainingConfig,
)


def test_valid_config_defaults():
    cfg = TrainingConfig(learning_rate=1e-3, epochs=5)
    assert cfg.optimizer == OptimizerType.ADAM
    assert cfg.weight_decay == 0.0
    assert cfg.momentum == 0.9


def test_custom_optimizer():
    cfg = TrainingConfig(learning_rate=0.01, epochs=3, optimizer=OptimizerType.SGD)
    assert cfg.optimizer == OptimizerType.SGD


def test_negative_lr_raises():
    with pytest.raises(ValueError, match="learning_rate"):
        TrainingConfig(learning_rate=-0.1, epochs=1)


def test_zero_epochs_raises():
    with pytest.raises(ValueError, match="epochs"):
        TrainingConfig(learning_rate=0.01, epochs=0)


def test_negative_weight_decay_raises():
    with pytest.raises(ValueError, match="weight_decay"):
        TrainingConfig(learning_rate=0.01, epochs=1, weight_decay=-1e-4)


def test_config_is_frozen():
    cfg = TrainingConfig(learning_rate=0.01, epochs=1)
    with pytest.raises(Exception):  # FrozenInstanceError
        cfg.epochs = 10  # type: ignore[misc]