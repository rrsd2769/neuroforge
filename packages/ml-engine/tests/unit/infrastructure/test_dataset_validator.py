"""
Unit tests for DatasetValidator.

Uses small synthetic in-memory datasets (lists of (PIL.Image, label)
tuples, shaped like torchvision's CIFAR10 output) instead of the real
60,000-sample dataset, so corruption/imbalance cases can be constructed
deliberately and the suite runs in milliseconds.
"""
from PIL import Image

from neuroforge_core.domain.entities.dataset import Dataset, DatasetMetadata
from neuroforge_core.infrastructure.datasets.dataset_validator import DatasetValidator


def _make_image(shape: tuple[int, int] = (32, 32)) -> Image.Image:
    return Image.new("RGB", shape)


def _balanced_dataset(num_per_class: int = 10, num_classes: int = 10) -> Dataset:
    samples = [
        (_make_image(), class_id)
        for class_id in range(num_classes)
        for _ in range(num_per_class)
    ]
    metadata = DatasetMetadata(
        name="synthetic", num_samples=len(samples), num_classes=num_classes,
        image_shape=(3, 32, 32),
    )
    return Dataset(metadata=metadata, raw={"train": samples, "test": samples})


def test_validator_passes_balanced_clean_dataset():
    report = DatasetValidator().validate(_balanced_dataset())
    assert report.is_valid
    assert report.issues == []


def test_validator_rejects_out_of_range_label():
    dataset = _balanced_dataset()
    train = dataset.raw["train"]
    train[0] = (train[0][0], 99)  # invalid: num_classes is 10

    report = DatasetValidator().validate(dataset)
    assert not report.is_valid
    assert any("out of range" in issue for issue in report.issues)


def test_validator_rejects_wrong_image_shape():
    dataset = _balanced_dataset()
    train = dataset.raw["train"]
    train[0] = (_make_image(shape=(16, 16)), train[0][1])

    report = DatasetValidator().validate(dataset)
    assert not report.is_valid
    assert any("expected shape" in issue for issue in report.issues)


def test_validator_flags_severe_class_imbalance():
    samples = [(_make_image(), 0) for _ in range(95)]
    samples += [(_make_image(), i) for i in range(1, 10)]  # 1 sample each
    metadata = DatasetMetadata(
        name="imbalanced", num_samples=len(samples), num_classes=10,
        image_shape=(3, 32, 32),
    )
    dataset = Dataset(metadata=metadata, raw={"train": samples, "test": samples})

    report = DatasetValidator().validate(dataset)
    assert not report.is_valid
    assert any("deviates" in issue for issue in report.issues)