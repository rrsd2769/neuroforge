from __future__ import annotations

from typing import Any, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from neuroforge_core.domain.interfaces.i_model_evaluator import IModelEvaluator
from neuroforge_core.domain.value_objects.evaluation_config import EvaluationConfig
from neuroforge_core.domain.value_objects.evaluation_metrics import EvaluationMetrics


class PyTorchEvaluator(IModelEvaluator):
    """
    Adapter: runs a full evaluation pass using PyTorch.

    Computes:
      - Top-1 accuracy
      - Top-k accuracy (k from EvaluationConfig)
      - Mean cross-entropy loss
      - Per-class accuracy
    """

    def evaluate(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        config: EvaluationConfig,
        class_names: Optional[List[str]] = None,
    ) -> EvaluationMetrics:
        device = self._resolve_device(config.device)
        model = model.to(device)
        model.eval()

        criterion = nn.CrossEntropyLoss()

        total_loss: float = 0.0
        correct: int = 0
        top_k_correct: int = 0
        total: int = 0

        num_classes: Optional[int] = None
        class_correct: Optional[torch.Tensor] = None
        class_total: Optional[torch.Tensor] = None

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs: torch.Tensor = model(inputs)
                batch_size = targets.size(0)

                # Initialise per-class trackers on first batch
                if num_classes is None:
                    num_classes = outputs.shape[1]
                    class_correct = torch.zeros(num_classes, device=device)
                    class_total = torch.zeros(num_classes, device=device)

                # Loss
                loss = criterion(outputs, targets)
                total_loss += loss.item()

                # Top-1 accuracy
                _, predicted = outputs.max(dim=1)
                correct += int(predicted.eq(targets).sum().item())

                # Top-k accuracy
                k = min(config.top_k, num_classes)
                _, top_k_pred = outputs.topk(k, dim=1)
                top_k_correct += int(
                    top_k_pred
                    .eq(targets.unsqueeze(1).expand_as(top_k_pred))
                    .any(dim=1)
                    .sum()
                    .item()
                )

                # Per-class accuracy
                for cls in range(num_classes):
                    cls_mask = targets == cls
                    class_total[cls] += cls_mask.sum()
                    class_correct[cls] += predicted[cls_mask].eq(cls).sum()

                total += batch_size

        if total == 0:
            raise RuntimeError("DataLoader was empty — cannot evaluate.")

        per_class_acc = {
            cls: float(class_correct[cls] / class_total[cls])
            if class_total[cls] > 0
            else 0.0
            for cls in range(num_classes)
        }

        return EvaluationMetrics(
            accuracy=correct / total,
            top_k_accuracy=top_k_correct / total,
            average_loss=total_loss / len(data_loader),
            per_class_accuracy=per_class_acc,
            total_samples=total,
            correct_predictions=correct,
            k=config.top_k,
        )

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)