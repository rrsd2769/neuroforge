import pytest
from torch.utils.data import Dataset as TorchDataset

from neuroforge_core.domain.entities.dataset import Dataset, DatasetMetadata
from neuroforge_core.domain.value_objects.preprocessing_config import PreprocessingConfig
from neuroforge_core.infrastructure.datasets.dataloader_factory import (
    DataLoaderFactory,
    DataLoaders,
)


class _FakeTorchDataset(TorchDataset):
    """A minimal torch Dataset of N samples — no images, no PIL, no torchvision."""

    def __init__(self, length: int) -> None:
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx):
        return idx, idx % 2  # (data, label) — values are irrelevant to these tests


def _make_dataset(train_len: int, test_len: int) -> Dataset:
    metadata = DatasetMetadata(
        name="fake", num_samples=train_len + test_len, num_classes=2, image_shape=(1, 1, 1)
    )
    raw = {"train": _FakeTorchDataset(train_len), "test": _FakeTorchDataset(test_len)}
    return Dataset(metadata=metadata, raw=raw)


def test_split_train_val_respects_configured_ratio():
    config = PreprocessingConfig(train_split=0.8, val_split=0.2, test_split=0.0)
    factory = DataLoaderFactory(config)
    train_subset, val_subset = factory.split_train_val(_FakeTorchDataset(100))
    assert len(train_subset) == 80
    assert len(val_subset) == 20


def test_split_train_val_is_deterministic_with_same_seed():
    config = PreprocessingConfig(split_seed=7)
    factory = DataLoaderFactory(config)
    raw = _FakeTorchDataset(200)

    train_a, val_a = factory.split_train_val(raw)
    train_b, val_b = factory.split_train_val(raw)

    assert list(train_a.indices) == list(train_b.indices)
    assert list(val_a.indices) == list(val_b.indices)


def test_split_train_val_differs_across_seeds():
    raw = _FakeTorchDataset(200)
    train_a, _ = DataLoaderFactory(PreprocessingConfig(split_seed=1)).split_train_val(raw)
    train_b, _ = DataLoaderFactory(PreprocessingConfig(split_seed=2)).split_train_val(raw)
    assert list(train_a.indices) != list(train_b.indices)


def test_build_returns_dataloaders_for_all_three_splits():
    dataset = _make_dataset(train_len=100, test_len=20)
    loaders = DataLoaderFactory().build(dataset, batch_size=10, num_workers=0)
    assert isinstance(loaders, DataLoaders)
    assert len(loaders.train.dataset) + len(loaders.val.dataset) == 100
    assert len(loaders.test.dataset) == 20


def test_build_test_loader_is_not_split():
    dataset = _make_dataset(train_len=100, test_len=20)
    loaders = DataLoaderFactory().build(dataset, batch_size=10, num_workers=0)
    # test loader must see every test sample — never carved by train_split:val_split
    assert len(loaders.test.dataset) == 20