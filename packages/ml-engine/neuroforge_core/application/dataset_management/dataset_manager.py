"""
DatasetManager.

Application-layer use case orchestrating dataset ingestion and
validation. Contains zero dataset-specific logic — all of that lives
inside whichever concrete dataset-source adapter it's given. This is
the seam that lets new datasets plug in later without touching this
file.
"""
from __future__ import annotations

import logging

from neuroforge_core.domain.entities.dataset import Dataset
from neuroforge_core.domain.interfaces.dataset_port import DatasetSourcePort
from neuroforge_core.infrastructure.datasets.dataset_validator import (
    DatasetValidator,
    ValidationReport,
)

logger = logging.getLogger(__name__)


class DatasetManager:
    """Orchestrates ingestion → validation for any DatasetSourcePort implementation."""

    def __init__(
        self,
        source: DatasetSourcePort,
        validator: DatasetValidator | None = None,
    ) -> None:
        self.source = source
        self.validator = validator or DatasetValidator()

    def ingest(self) -> Dataset:
        """Loads the dataset via the configured DatasetSourcePort."""
        logger.info("Ingesting dataset via %s", type(self.source).__name__)
        return self.source.load()

    def validate(self, dataset: Dataset) -> ValidationReport:
        """Validates a previously ingested dataset."""
        return self.validator.validate(dataset)

    def prepare(self) -> tuple[Dataset, ValidationReport]:
        """
        Convenience method combining ingest() + validate() in one call.
        Raises ValueError if validation fails — callers should not have
        to remember to check `report.is_valid` themselves.
        """
        dataset = self.ingest()
        report = self.validate(dataset)
        if not report.is_valid:
            raise ValueError(f"Dataset validation failed: {report.issues}")
        return dataset, report