from __future__ import annotations

from typing import Any, Optional

from neuroforge_core.domain.interfaces.i_model_evaluator import IModelEvaluator
from neuroforge_core.domain.value_objects.evaluation_config import EvaluationConfig
from neuroforge_core.domain.value_objects.evaluation_metrics import EvaluationMetrics


class EvaluateModelUseCase:
    """
    Application use case: evaluate a compiled, trained model.

    Orchestrates the IModelEvaluator port. Deliberately thin —
    application logic here is selection of defaults, not ML math.
    """

    def __init__(self, evaluator: IModelEvaluator) -> None:
        self._evaluator = evaluator

    def execute(
        self,
        model: Any,
        data_loader: Any,
        config: Optional[EvaluationConfig] = None,
    ) -> EvaluationMetrics:
        """
        Args:
            model:       Compiled, trained model (torch.nn.Module in practice).
            data_loader: Test or validation DataLoader.
            config:      Evaluation config. Uses defaults if None.

        Returns:
            EvaluationMetrics with accuracy, loss, per-class breakdown.
        """
        resolved_config = config if config is not None else EvaluationConfig()
        return self._evaluator.evaluate(model, data_loader, resolved_config)