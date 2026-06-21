import pytest
import torchvision.transforms as T

from neuroforge_core.domain.entities.dataset import Dataset, DatasetMetadata
from neuroforge_core.domain.value_objects.preprocessing_config import PreprocessingConfig
from neuroforge_core.infrastructure.datasets.preprocessing import PreprocessingPipeline


class _FakeSplit:
    """Stands in for a torchvision dataset split — only `.transform` matters here."""

    def __init__(self) -> None:
        self.transform = None


def _make_dataset(train=None, test=None) -> Dataset:
    metadata = DatasetMetadata(
        name="fake", num_samples=10, num_classes=2, image_shape=(3, 32, 32)
    )
    raw = {}
    if train is not None:
        raw["train"] = train
    if test is not None:
        raw["test"] = test
    return Dataset(metadata=metadata, raw=raw)


def test_train_transform_includes_augmentation_when_enabled():
    pipeline = PreprocessingPipeline(PreprocessingConfig(use_augmentation=True))
    transform_types = [type(t) for t in pipeline.train_transform().transforms]
    assert T.RandomCrop in transform_types
    assert T.RandomHorizontalFlip in transform_types


def test_train_transform_skips_augmentation_when_disabled():
    pipeline = PreprocessingPipeline(PreprocessingConfig(use_augmentation=False))
    transform_types = [type(t) for t in pipeline.train_transform().transforms]
    assert T.RandomCrop not in transform_types
    assert T.RandomHorizontalFlip not in transform_types


def test_eval_transform_never_includes_augmentation():
    pipeline = PreprocessingPipeline(PreprocessingConfig(use_augmentation=True))
    transform_types = [type(t) for t in pipeline.eval_transform().transforms]
    assert T.RandomCrop not in transform_types
    assert T.RandomHorizontalFlip not in transform_types


def test_apply_sets_transform_on_train_and_test_splits():
    train_split, test_split = _FakeSplit(), _FakeSplit()
    dataset = _make_dataset(train=train_split, test=test_split)

    PreprocessingPipeline().apply(dataset)

    assert isinstance(train_split.transform, T.Compose)
    assert isinstance(test_split.transform, T.Compose)


def test_apply_raises_when_split_missing():
    dataset = _make_dataset(train=_FakeSplit())  # no "test" key
    with pytest.raises(ValueError):
        PreprocessingPipeline().apply(dataset)