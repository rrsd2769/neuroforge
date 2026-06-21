"""
Unit tests for DatasetManager, using a fake DatasetSourcePort so the use
case is tested in isolation from any real dataset, download, or I/O.
"""
import pytest
from torch.utils.data import Dataset as TorchDataset

from neuroforge_core.application.dataset_management.dataset_manager import DatasetManager
from neuroforge_core.domain.entities.dataset import Dataset, DatasetMetadata
from neuroforge_core.domain.interfaces.dataset_port import DatasetSourcePort
from neuroforge_core.infrastructure.datasets.dataloader_factory import DataLoaders
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


class _FakeTorchSplit(TorchDataset):
    """Minimal torch Dataset standing in for a torchvision split — needs a
    settable `.transform` attribute, matching the real CIFAR10 contract."""

    def __init__(self, length: int) -> None:
        self._length = length
        self.transform = None

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx):
        item = idx
        if self.transform is not None:
            item = self.transform(item)
        return item, idx % 2


def _clean_dataset() -> Dataset:
    metadata = DatasetMetadata(name="fake", num_samples=100, num_classes=10, image_shape=(3, 32, 32))
    return Dataset(metadata=metadata, raw={"train": [], "test": []})


def _dataset_with_real_splits(train_len=20, test_len=5) -> Dataset:
    """Like _clean_dataset(), but with real torch Dataset splits — needed
    for the Day 3 tests below, since preprocess() calls .transform = ...
    on each split, and DataLoaderFactory needs real __len__/__getitem__."""
    metadata = DatasetMetadata(
        name="fake", num_samples=train_len + test_len, num_classes=2, image_shape=(1, 1, 1)
    )
    raw = {"train": _FakeTorchSplit(train_len), "test": _FakeTorchSplit(test_len)}
    return Dataset(metadata=metadata, raw=raw)


# ── Day 2 tests — unchanged ──────────────────────────────────────────────────


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


# ── Day 3 tests — new ─────────────────────────────────────────────────────────


def test_preprocess_applies_pipeline_then_builds_loaders():
    dataset = _dataset_with_real_splits()
    manager = DatasetManager(source=_FakeDatasetSource(dataset))

    loaders = manager.preprocess(dataset, batch_size=4, num_workers=0)

    assert isinstance(loaders, DataLoaders)
    assert dataset.raw["train"].transform is not None  # pipeline ran before the split


def test_prepare_dataloaders_returns_loaders_and_report():
    dataset = _dataset_with_real_splits()
    manager = DatasetManager(source=_FakeDatasetSource(dataset))
    manager.validator = _StubValidator(ValidationReport(is_valid=True, issues=[]))

    loaders, report = manager.prepare_dataloaders(batch_size=4, num_workers=0)

    assert isinstance(loaders, DataLoaders)
    assert report.is_valid is True