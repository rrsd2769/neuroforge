from abc import ABC, abstractmethod
from typing import Any, List, Optional

from neuroforge_core.domain.value_objects.evaluation_config import EvaluationConfig
from neuroforge_core.domain.value_objects.evaluation_metrics import EvaluationMetrics


class IModelEvaluator(ABC):
    """
    Port: evaluate a compiled, trained model against a data loader.
    """

    @abstractmethod
    def evaluate(
        self,
        model: Any,
        data_loader: Any,
        config: EvaluationConfig,
        class_names: Optional[List[str]] = None,
    ) -> EvaluationMetrics:
        """
        Run full evaluation pass and return structured metrics.

        Args:
            model:        Backend model (torch.nn.Module in practice).
            data_loader:  Iterable of (inputs, targets) batches.
            config:       Evaluation settings.
            class_names:  Optional human-readable class labels.
        """
        ...