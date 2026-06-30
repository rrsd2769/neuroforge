"""
Day 1 exit-gate tests — updated for Day 4 Architecture API.

Architecture tests now use Layer VOs (ConvLayer, etc.) instead of the
old *Spec classes. The simulate_output_shape stub test is removed —
Day 4 implements it and tests it in tests/domain/test_architecture.py.
"""
import pytest

from neuroforge_core.domain.entities.architecture import Architecture
from neuroforge_core.domain.value_objects.layer import (
    ConvLayer, DenseLayer, FlattenLayer, PoolLayer,
)
from neuroforge_core.domain.entities.dataset import Dataset, DatasetMetadata
from neuroforge_core.domain.entities.evaluation_result import (
    EvaluationResult, TrainingMetrics,
)
from neuroforge_core.domain.entities.model_artifact import ModelArtifact
from neuroforge_core.domain.interfaces.architecture_generator_port import ArchitectureGeneratorPort
from neuroforge_core.domain.interfaces.dataset_port import DatasetSourcePort
from neuroforge_core.domain.interfaces.evaluator_port import EvaluatorPort
from neuroforge_core.domain.interfaces.model_trainer import IModelTrainer
from neuroforge_core.domain.interfaces.i_experiment_repository import IExperimentRepository
from neuroforge_core.domain.value_objects.preprocessing_config import PreprocessingConfig
from neuroforge_core.domain.value_objects.search_space import SearchSpace
from neuroforge_core.domain.value_objects.training_config import TrainingConfig

# ── Dataset ───────────────────────────────────────────────────────────────────

def test_dataset_metadata_valid():
    meta = DatasetMetadata(
        name="cifar10", num_samples=60000, num_classes=10,
        image_shape=(3, 32, 32), class_names=["plane", "car"],
    )
    ds = Dataset(metadata=meta)
    assert ds.to_dict()["num_samples"] == 60000


def test_dataset_metadata_rejects_invalid_sample_count():
    with pytest.raises(ValueError):
        DatasetMetadata(name="bad", num_samples=0, num_classes=10, image_shape=(3, 32, 32))


# ── Architecture ──────────────────────────────────────────────────────────────

def _sample_architecture() -> Architecture:
    return Architecture(
        layers=[
            ConvLayer(out_channels=32, kernel_size=3, padding=1),
            PoolLayer(pool_size=2, stride=2),
            FlattenLayer(),
            DenseLayer(units=10),
        ],
        num_classes=10,
    )


def test_architecture_instantiates():
    arch = _sample_architecture()
    assert arch.layer_count() == 4
    assert arch.num_classes == 10


def test_architecture_simulate_output_shape():
    arch = _sample_architecture()
    assert arch.simulate_output_shape((3, 32, 32)) == (10,)


def test_architecture_is_valid_for_input():
    assert _sample_architecture().is_valid_for_input((3, 32, 32))


def test_architecture_unique_ids():
    assert _sample_architecture().id != _sample_architecture().id



# ── ModelArtifact / EvaluationResult / TrainingMetrics ───────────────────────

def test_model_artifact_round_trip():
    artifact = ModelArtifact(architecture_id="a1", experiment_id="e1", total_params=12345)
    assert artifact.to_dict()["total_params"] == 12345


def test_evaluation_result_rejects_out_of_range_accuracy():
    with pytest.raises(ValueError):
        EvaluationResult(accuracy=1.5)


def test_training_metrics_to_dict():
    m = TrainingMetrics(
        epoch=1, train_loss=1.2, train_accuracy=0.4,
        val_loss=1.1, val_accuracy=0.42, epoch_duration_seconds=12.3,
    )
    assert m.to_dict()["epoch"] == 1


# ── Value objects ─────────────────────────────────────────────────────────────

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

@pytest.mark.parametrize("port_cls", [
    DatasetSourcePort, ArchitectureGeneratorPort,
    IModelTrainer, EvaluatorPort, IExperimentRepository,
])
def test_ports_cannot_be_instantiated_directly(port_cls):
    with pytest.raises(TypeError):
        port_cls()