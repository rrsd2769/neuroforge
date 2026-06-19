"""
Day 1 exit-gate tests.

Every domain entity, value object, and port must import cleanly,
instantiate correctly, reject invalid data, and — for Architecture —
round-trip through to_dict()/from_dict() losslessly. This suite is the
gate Day 2 onward is built against.
"""
import pytest

from neuroforge_core.domain.entities.architecture import (
    ActivationSpec,
    Architecture,
    ConvLayerSpec,
    LinearLayerSpec,
    PoolLayerSpec,
)
from neuroforge_core.domain.entities.dataset import Dataset, DatasetMetadata
from neuroforge_core.domain.entities.evaluation_result import (
    EvaluationResult,
    TrainingMetrics,
)
from neuroforge_core.domain.entities.experiment import Experiment, ExperimentStatus
from neuroforge_core.domain.entities.model_artifact import ModelArtifact
from neuroforge_core.domain.interfaces.architecture_generator_port import (
    ArchitectureGeneratorPort,
)
from neuroforge_core.domain.interfaces.dataset_port import DatasetSourcePort
from neuroforge_core.domain.interfaces.evaluator_port import EvaluatorPort
from neuroforge_core.domain.interfaces.experiment_tracker_port import (
    ExperimentTrackerPort,
)
from neuroforge_core.domain.interfaces.trainer_port import TrainerPort
from neuroforge_core.domain.value_objects.preprocessing_config import (
    PreprocessingConfig,
)
from neuroforge_core.domain.value_objects.search_space import SearchSpace
from neuroforge_core.domain.value_objects.training_config import TrainingConfig


# ── Dataset ──────────────────────────────────────────────────────────────────


def test_dataset_metadata_valid():
    meta = DatasetMetadata(
        name="cifar10",
        num_samples=60000,
        num_classes=10,
        image_shape=(3, 32, 32),
        class_names=["plane", "car"],
    )
    ds = Dataset(metadata=meta)
    assert ds.to_dict()["num_samples"] == 60000


def test_dataset_metadata_rejects_invalid_sample_count():
    with pytest.raises(ValueError):
        DatasetMetadata(name="bad", num_samples=0, num_classes=10, image_shape=(3, 32, 32))


# ── Architecture ─────────────────────────────────────────────────────────────


def _sample_architecture() -> Architecture:
    arch = Architecture(architecture_id="test-arch-1")
    arch.add_layer(ConvLayerSpec(out_channels=32))
    arch.add_layer(ActivationSpec(kind="relu"))
    arch.add_layer(PoolLayerSpec(pool_kind="max"))
    arch.add_layer(ConvLayerSpec(out_channels=64))
    arch.add_layer(LinearLayerSpec(out_features=10))
    return arch


def test_architecture_round_trip_lossless():
    original = _sample_architecture()
    restored = Architecture.from_dict(original.to_dict())
    assert restored.to_dict() == original.to_dict()
    assert len(restored.layers) == len(original.layers)


def test_architecture_rejects_unknown_layer_type():
    bad_dict = _sample_architecture().to_dict()
    bad_dict["layers"][0]["layer_type"] = "transformer_block"
    with pytest.raises(ValueError):
        Architecture.from_dict(bad_dict)


def test_conv_layer_spec_rejects_invalid_channels():
    with pytest.raises(ValueError):
        ConvLayerSpec(out_channels=0)


def test_architecture_simulate_output_shape_is_day4_stub():
    arch = _sample_architecture()
    with pytest.raises(NotImplementedError):
        arch.simulate_output_shape()


# ── Experiment ───────────────────────────────────────────────────────────────


def test_experiment_lifecycle():
    exp = Experiment()
    assert exp.status == ExperimentStatus.PENDING
    exp.mark_running()
    assert exp.status == ExperimentStatus.RUNNING
    exp.mark_completed()
    assert exp.status == ExperimentStatus.COMPLETED
    assert exp.completed_at is not None


def test_experiment_failure_records_error():
    exp = Experiment()
    exp.mark_failed("OOM during training")
    assert exp.status == ExperimentStatus.FAILED
    assert exp.error_message == "OOM during training"


def test_experiment_id_is_unique_per_instance():
    assert Experiment().experiment_id != Experiment().experiment_id


# ── ModelArtifact / EvaluationResult / TrainingMetrics ───────────────────────


def test_model_artifact_round_trip():
    artifact = ModelArtifact(architecture_id="a1", experiment_id="e1", total_params=12345)
    assert artifact.to_dict()["total_params"] == 12345


def test_evaluation_result_rejects_out_of_range_accuracy():
    with pytest.raises(ValueError):
        EvaluationResult(accuracy=1.5)


def test_training_metrics_to_dict():
    m = TrainingMetrics(
        epoch=1,
        train_loss=1.2,
        train_accuracy=0.4,
        val_loss=1.1,
        val_accuracy=0.42,
        epoch_duration_seconds=12.3,
    )
    assert m.to_dict()["epoch"] == 1


# ── Value objects ────────────────────────────────────────────────────────────


def test_search_space_defaults_valid():
    ss = SearchSpace()
    assert ss.max_depth >= ss.min_depth


def test_search_space_rejects_inverted_depth_range():
    with pytest.raises(ValueError):
        SearchSpace(min_depth=5, max_depth=2)


def test_training_config_rejects_zero_epochs():
    with pytest.raises(ValueError):
        TrainingConfig(epochs=0)


def test_training_config_rejects_unknown_optimizer():
    with pytest.raises(ValueError):
        TrainingConfig(optimizer="rmsprop")


def test_preprocessing_config_requires_splits_sum_to_one():
    with pytest.raises(ValueError):
        PreprocessingConfig(train_split=0.5, val_split=0.3, test_split=0.3)


def test_preprocessing_config_default_splits_valid():
    cfg = PreprocessingConfig()
    assert round(cfg.train_split + cfg.val_split + cfg.test_split, 6) == 1.0


# ── Ports are abstract ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "port_cls",
    [
        DatasetSourcePort,
        ArchitectureGeneratorPort,
        TrainerPort,
        EvaluatorPort,
        ExperimentTrackerPort,
    ],
)
def test_ports_cannot_be_instantiated_directly(port_cls):
    with pytest.raises(TypeError):
        port_cls()