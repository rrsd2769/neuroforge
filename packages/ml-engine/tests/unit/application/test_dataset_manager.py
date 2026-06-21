"""
Unit tests for DatasetManager, using a fake DatasetSourcePort so the use
case is tested in isolation from any real dataset, download, or I/O.
"""
import pytest

from neuroforge_core.application.dataset_management.dataset_manager import DatasetManager
from neuroforge_core.domain.entities.dataset import Dataset, DatasetMetadata
from neuroforge_core.domain.interfaces.dataset_port import DatasetSourcePort
from neuroforge_core.infrastructure.datasets.dataset_validator import ValidationReport


class _FakeDatasetSource(DatasetSourcePort):
    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset

    def load(self) -> Dataset:
        return self._dataset


class _StubValidator:
    def __init__(self, report: ValidationReport) -> None:
        self._report = report

    def validate(self, dataset: Dataset) -> ValidationReport:
        return self._report


def _clean_dataset() -> Dataset:
    metadata = DatasetMetadata(name="fake", num_samples=100, num_classes=10, image_shape=(3, 32, 32))
    return Dataset(metadata=metadata, raw={"train": [], "test": []})


def test_ingest_delegates_to_source():
    dataset = _clean_dataset()
    manager = DatasetManager(source=_FakeDatasetSource(dataset))
    assert manager.ingest() is dataset


def test_prepare_raises_on_invalid_dataset():
    dataset = _clean_dataset()
    manager = DatasetManager(source=_FakeDatasetSource(dataset))
    manager.validator = _StubValidator(ValidationReport(is_valid=False, issues=["forced failure"]))

    with pytest.raises(ValueError, match="forced failure"):
        manager.prepare()


def test_prepare_returns_dataset_and_report_on_success():
    dataset = _clean_dataset()
    manager = DatasetManager(source=_FakeDatasetSource(dataset))
    manager.validator = _StubValidator(ValidationReport(is_valid=True, issues=[]))

    returned_dataset, report = manager.prepare()
    assert returned_dataset is dataset
    assert report.is_valid