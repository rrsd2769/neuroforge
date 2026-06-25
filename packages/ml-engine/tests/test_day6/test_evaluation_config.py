import pytest
from neuroforge_core.domain.value_objects.evaluation_config import EvaluationConfig


def test_defaults():
    config = EvaluationConfig()
    assert config.batch_size == 128
    assert config.device == "auto"
    assert config.top_k == 5
    assert config.num_workers == 2


def test_custom_values():
    config = EvaluationConfig(batch_size=64, device="cpu", top_k=3, num_workers=4)
    assert config.batch_size == 64
    assert config.device == "cpu"
    assert config.top_k == 3


def test_invalid_batch_size_zero():
    with pytest.raises(ValueError, match="batch_size"):
        EvaluationConfig(batch_size=0)


def test_invalid_batch_size_negative():
    with pytest.raises(ValueError, match="batch_size"):
        EvaluationConfig(batch_size=-10)


def test_invalid_top_k():
    with pytest.raises(ValueError, match="top_k"):
        EvaluationConfig(top_k=0)


def test_invalid_device():
    with pytest.raises(ValueError, match="device"):
        EvaluationConfig(device="tpu")


def test_immutable():
    config = EvaluationConfig()
    with pytest.raises((AttributeError, TypeError)):
        config.batch_size = 64