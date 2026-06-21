"""
Unit tests for CIFAR10DatasetSource.

Mocks torchvision.datasets.CIFAR10 so this suite never downloads the real
~170MB dataset. The real download is exercised manually via Step 2.1's
smoke test, or automatically as part of Day 7's full pipeline run.
"""
from unittest.mock import MagicMock, patch

from neuroforge_core.infrastructure.datasets.cifar10_loader import CIFAR10DatasetSource


def _fake_cifar10(train: bool):
    fake = MagicMock()
    fake.__len__.return_value = 50000 if train else 10000
    return fake


@patch("neuroforge_core.infrastructure.datasets.cifar10_loader.torchvision.datasets.CIFAR10")
def test_load_returns_dataset_with_correct_metadata(mock_cifar, tmp_path):
    mock_cifar.side_effect = lambda root, train, download: _fake_cifar10(train)

    dataset = CIFAR10DatasetSource(data_dir=tmp_path).load()

    assert dataset.metadata.name == "cifar10"
    assert dataset.metadata.num_classes == 10
    assert dataset.metadata.image_shape == (3, 32, 32)
    assert dataset.metadata.num_samples == 60000
    assert len(dataset.metadata.class_names) == 10


@patch("neuroforge_core.infrastructure.datasets.cifar10_loader.torchvision.datasets.CIFAR10")
def test_load_creates_data_directory(mock_cifar, tmp_path):
    mock_cifar.side_effect = lambda root, train, download: _fake_cifar10(train)
    data_dir = tmp_path / "cifar10_cache"

    CIFAR10DatasetSource(data_dir=data_dir)
    assert data_dir.exists()


@patch("neuroforge_core.infrastructure.datasets.cifar10_loader.torchvision.datasets.CIFAR10")
def test_load_always_requests_download_true(mock_cifar, tmp_path):
    mock_cifar.side_effect = lambda root, train, download: _fake_cifar10(train)
    CIFAR10DatasetSource(data_dir=tmp_path).load()

    for call in mock_cifar.call_args_list:
        assert call.kwargs["download"] is True