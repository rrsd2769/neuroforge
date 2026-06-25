import pytest
from unittest.mock import MagicMock

from neuroforge_core.application.use_cases.evaluate_model_use_case import (
    EvaluateModelUseCase,
)
from neuroforge_core.domain.value_objects.evaluation_config import EvaluationConfig
from neuroforge_core.domain.value_objects.evaluation_metrics import EvaluationMetrics


def _fake_metrics() -> EvaluationMetrics:
    return EvaluationMetrics(
        accuracy=0.85,
        top_k_accuracy=0.95,
        average_loss=0.42,
        per_class_accuracy={i: 0.85 for i in range(10)},
        total_samples=1000,
        correct_predictions=850,
        k=5,
    )


def test_execute_delegates_to_evaluator():
    evaluator = MagicMock()
    evaluator.evaluate.return_value = _fake_metrics()

    use_case = EvaluateModelUseCase(evaluator=evaluator)
    model = MagicMock()
    loader = MagicMock()
    config = EvaluationConfig(device="cpu")

    result = use_case.execute(model, loader, config)

    evaluator.evaluate.assert_called_once_with(model, loader, config)
    assert result.accuracy == 0.85


def test_execute_uses_default_config_when_none_passed():
    evaluator = MagicMock()
    evaluator.evaluate.return_value = _fake_metrics()

    use_case = EvaluateModelUseCase(evaluator=evaluator)
    use_case.execute(MagicMock(), MagicMock(), None)

    call_args = evaluator.evaluate.call_args
    passed_config = call_args[0][2]
    assert isinstance(passed_config, EvaluationConfig)


def test_execute_returns_evaluation_metrics():
    evaluator = MagicMock()
    evaluator.evaluate.return_value = _fake_metrics()

    use_case = EvaluateModelUseCase(evaluator=evaluator)
    result = use_case.execute(MagicMock(), MagicMock(), EvaluationConfig())

    assert isinstance(result, EvaluationMetrics)


def test_execute_propagates_evaluator_exception():
    evaluator = MagicMock()
    evaluator.evaluate.side_effect = RuntimeError("DataLoader was empty")

    use_case = EvaluateModelUseCase(evaluator=evaluator)

    with pytest.raises(RuntimeError, match="DataLoader was empty"):
        use_case.execute(MagicMock(), MagicMock(), EvaluationConfig())