"""
DatasetValidator.

Confirms dataset integrity immediately after ingestion — wrong shapes,
out-of-range labels, or severe class imbalance fail loudly here, not
three steps later as a cryptic tensor error inside the training loop.

Deliberately not behind a port this week: this is a focused internal
helper used only by DatasetManager, not a swappable strategy.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

from neuroforge_core.domain.entities.dataset import Dataset

logger = logging.getLogger(__name__)

# How far a class's share of samples may deviate from a perfectly uniform
# split before validation fails. 0.5 = a class can be up to 50% above or
# below the expected even split.
MAX_CLASS_IMBALANCE_RATIO = 0.5
SAMPLE_CHECK_SIZE = 500


@dataclass
class ValidationReport:
    """Result of running DatasetValidator against a Dataset."""

    is_valid: bool
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"is_valid": self.is_valid, "issues": self.issues}


class DatasetValidator:
    """Runs shape, label-range, and class-balance checks against a Dataset."""

    def validate(self, dataset: Dataset) -> ValidationReport:
        issues: list[str] = []
        issues += self._check_shapes(dataset)
        issues += self._check_label_range(dataset)
        issues += self._check_class_balance(dataset)

        report = ValidationReport(is_valid=len(issues) == 0, issues=issues)
        if report.is_valid:
            logger.info("Dataset '%s' passed validation.", dataset.metadata.name)
        else:
            logger.warning(
                "Dataset '%s' failed validation: %s", dataset.metadata.name, issues
            )
        return report

    def _train_split(self, dataset: Dataset):
        return dataset.raw.get("train") if isinstance(dataset.raw, dict) else None

    def _check_shapes(self, dataset: Dataset) -> list[str]:
        issues: list[str] = []
        train_split = self._train_split(dataset)
        if train_split is None:
            return ["dataset.raw missing a 'train' split — cannot verify shapes"]

        expected_shape = dataset.metadata.image_shape
        for i in range(min(SAMPLE_CHECK_SIZE, len(train_split))):
            image, _ = train_split[i]
            if not hasattr(image, "getbands"):
                continue  # not a PIL image (e.g. already-tensor dataset) — skip shape check
            actual_shape = (len(image.getbands()), *image.size[::-1])
            if actual_shape != expected_shape:
                issues.append(f"Sample {i}: expected shape {expected_shape}, got {actual_shape}")
                break
        return issues

    def _check_label_range(self, dataset: Dataset) -> list[str]:
        issues: list[str] = []
        train_split = self._train_split(dataset)
        if train_split is None:
            return issues

        num_classes = dataset.metadata.num_classes
        for i in range(min(SAMPLE_CHECK_SIZE, len(train_split))):
            _, label = train_split[i]
            if not (0 <= label < num_classes):
                issues.append(f"Sample {i}: label {label} out of range [0, {num_classes})")
                break
        return issues

    def _check_class_balance(self, dataset: Dataset) -> list[str]:
        issues: list[str] = []
        train_split = self._train_split(dataset)
        if train_split is None:
            return issues

        sample_count = min(SAMPLE_CHECK_SIZE, len(train_split))
        labels = [train_split[i][1] for i in range(sample_count)]
        counts = Counter(labels)
        if not counts:
            return issues

        expected_per_class = sample_count / dataset.metadata.num_classes
        for class_id, count in counts.items():
            deviation = abs(count - expected_per_class) / expected_per_class
            if deviation > MAX_CLASS_IMBALANCE_RATIO:
                issues.append(
                    f"Class {class_id}: {count} samples deviates "
                    f">{MAX_CLASS_IMBALANCE_RATIO:.0%} from expected {expected_per_class:.0f}"
                )
        return issues