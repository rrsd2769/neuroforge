"""
DatasetManager.

Application-layer use case orchestrating dataset ingestion, validation,
and (as of Day 3) preprocessing + DataLoader construction. Contains zero
dataset-specific logic — all of that lives inside whichever
DatasetSourcePort implementation it's given (e.g. CIFAR10DatasetSource).
This is the seam that lets FashionMNIST/Tiny ImageNet plug in later
without touching this file.
"""
from __future__ import annotations

import logging

from neuroforge_core.domain.entities.dataset import Dataset
from neuroforge_core.domain.interfaces.dataset_port import DatasetSourcePort
from neuroforge_core.domain.value_objects.preprocessing_config import (
    PreprocessingConfig,
)
from neuroforge_core.infrastructure.datasets.dataloader_factory import (
    DataLoaderFactory,
    DataLoaders,
)
from neuroforge_core.infrastructure.datasets.dataset_validator import (
    DatasetValidator,
    ValidationReport,
)
from neuroforge_core.infrastructure.datasets.preprocessing import PreprocessingPipeline

logger = logging.getLogger(__name__)


class DatasetManager:
    """Orchestrates ingestion → validation → preprocessing for any DatasetSourcePort implementation."""

    def __init__(
        self,
        source: DatasetSourcePort,
        validator: DatasetValidator | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
    ) -> None:
        self.source = source
        self.validator = validator or DatasetValidator()
        self.preprocessing_config = preprocessing_config or PreprocessingConfig()
        self.pipeline = PreprocessingPipeline(self.preprocessing_config)
        self.loader_factory = DataLoaderFactory(self.preprocessing_config)

    def ingest(self) -> Dataset:
        """Loads the dataset via the configured DatasetSourcePort."""
        logger.info("Ingesting dataset via %s", type(self.source).__name__)
        return self.source.load()

    def validate(self, dataset: Dataset) -> ValidationReport:
        """Validates a previously ingested dataset."""
        return self.validator.validate(dataset)

    def prepare(self) -> tuple[Dataset, ValidationReport]:
        """
        Day 2 contract, unchanged: ingest() + validate() in one call.
        Raises ValueError if validation fails — callers should not have
        to remember to check `report.is_valid` themselves.
        """
        dataset = self.ingest()
        report = self.validate(dataset)
        if not report.is_valid:
            raise ValueError(f"Dataset validation failed: {report.issues}")
        return dataset, report

    def preprocess(
        self, dataset: Dataset, batch_size: int = 128, num_workers: int = 2
    ) -> DataLoaders:
        """
        Applies the configured PreprocessingPipeline to an already-ingested
        Dataset, then builds train/val/test DataLoaders from it. Expects
        a Dataset that has already passed validate() — this method does
        not re-validate.
        """
        self.pipeline.apply(dataset)
        return self.loader_factory.build(dataset, batch_size=batch_size, num_workers=num_workers)

    def prepare_dataloaders(
        self, batch_size: int = 128, num_workers: int = 2
    ) -> tuple[DataLoaders, ValidationReport]:
        """
        Single-call path from raw source to ready-to-train DataLoaders:
        ingest -> validate -> preprocess -> build loaders. This is the
        method Day 4's architecture-search loop will call directly.
        """
        dataset, report = self.prepare()
        loaders = self.preprocess(dataset, batch_size=batch_size, num_workers=num_workers)
        return loaders, report