"""
EvaluatorPort.

Abstract contract for final, held-out-set evaluation of a trained model.
Implemented by ClassificationEvaluator (Day 7).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from neuroforge_core.domain.entities.evaluation_result import EvaluationResult


class EvaluatorPort(ABC):
    @abstractmethod
    def evaluate(self, model: Any, test_loader: Any) -> EvaluationResult:
        """Runs inference over the test set and returns aggregated metrics."""
        raise NotImplementedError