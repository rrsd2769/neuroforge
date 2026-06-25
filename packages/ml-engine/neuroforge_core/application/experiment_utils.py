"""
experiment_utils.py
===================
Factory helpers that convert domain objects (Architecture, TrainingConfig,
TrainModelResponse, EvaluationMetrics) into a flat ExperimentSnapshot dict.

Lives in the application layer because it orchestrates domain types — it's
not domain logic (no business rules) and not infrastructure (no I/O).
"""
from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Optional

from neuroforge_core.domain.entities.architecture import Architecture
from neuroforge_core.domain.entities.experiment_snapshot import ExperimentSnapshot
from neuroforge_core.domain.value_objects.layer import (
    ConvLayer,
    DenseLayer,
    DropoutLayer,
    FlattenLayer,
    PoolLayer,
)
from neuroforge_core.domain.value_objects.training_config import TrainingConfig


# ------------------------------------------------------------------ #
# Layer serialization
# ------------------------------------------------------------------ #

def _serialize_layer(layer) -> dict:
    """Convert a concrete Layer dataclass to a tagged dict."""
    if isinstance(layer, ConvLayer):
        tag = "conv"
    elif isinstance(layer, PoolLayer):
        tag = "pool"
    elif isinstance(layer, FlattenLayer):
        tag = "flatten"
    elif isinstance(layer, DenseLayer):
        tag = "dense"
    elif isinstance(layer, DropoutLayer):
        tag = "dropout"
    else:
        tag = "unknown"

    # dataclasses.asdict works fine on simple dataclasses (no nested DCs)
    d = dataclasses.asdict(layer)
    d["type"] = tag
    return d


def serialize_architecture(arch: Architecture) -> dict:
    """
    Produce a JSON-safe summary of an Architecture.

    Returns
    -------
    dict with keys: architecture_id, num_layers, num_classes, layers
    """
    return {
        "architecture_id": str(arch.id),
        "num_layers": len(arch.layers),
        "num_classes": arch.num_classes,
        "layers": [_serialize_layer(layer) for layer in arch.layers],
    }


# ------------------------------------------------------------------ #
# TrainingConfig serialization
# ------------------------------------------------------------------ #

def serialize_training_config(config: TrainingConfig) -> dict:
    """
    Produce a JSON-safe dict from a TrainingConfig.

    Handles Enum fields automatically — no hardcoded field names so the
    function stays correct if TrainingConfig gains new fields.
    """
    result: dict = {}
    for f in dataclasses.fields(config):
        val = getattr(config, f.name)
        if isinstance(val, Enum):
            result[f.name] = val.value
        else:
            result[f.name] = val
    return result


# ------------------------------------------------------------------ #
# Top-level factory
# ------------------------------------------------------------------ #

def build_snapshot(
    name: str,
    architecture: Architecture,
    training_config: TrainingConfig,
    final_train_loss: float,
    eval_metrics=None,   # EvaluationMetrics | None
    tags: Optional[dict] = None,
    experiment_id: Optional[str] = None,
) -> ExperimentSnapshot:
    """
    Build an ExperimentSnapshot from the outputs of a full training run.

    Parameters
    ----------
    name            : Human-readable label for the run.
    architecture    : The Architecture entity that was trained.
    training_config : The TrainingConfig used.
    final_train_loss: Last epoch's training loss (from TrainModelResponse).
    eval_metrics    : Optional EvaluationMetrics from EvaluateModelUseCase.
    tags            : Arbitrary metadata dict.
    experiment_id   : If provided, snapshot uses this id (links to Experiment
                      entity). If None, a new UUID is generated.
    """
    arch_summary = serialize_architecture(architecture)
    config_dict = serialize_training_config(training_config)

    results: dict = {"final_train_loss": round(final_train_loss, 6)}

    if eval_metrics is not None:
        results["top1_accuracy"] = round(float(eval_metrics.accuracy), 6)
        results["top5_accuracy"] = round(float(eval_metrics.top_k_accuracy), 6)
        results["mean_eval_loss"] = round(float(eval_metrics.average_loss), 6)
    else:
        results["top1_accuracy"] = None
        results["top5_accuracy"] = None
        results["mean_eval_loss"] = None

    kwargs: dict = {
        "name": name,
        "architecture_summary": arch_summary,
        "training_config": config_dict,
        "results": results,
        "tags": tags or {},
    }
    if experiment_id is not None:
        kwargs["experiment_id"] = experiment_id

    return ExperimentSnapshot(**kwargs)