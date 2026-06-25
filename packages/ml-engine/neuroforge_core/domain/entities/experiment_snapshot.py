"""
ExperimentSnapshot — a flat, JSON-serializable record of one full training run.

Companion to the Experiment lifecycle entity. While Experiment tracks status
transitions, ExperimentSnapshot captures the *results* — architecture shape,
training config, and evaluation metrics — in a form that's easy to persist,
load, and compare.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


def _new_id() -> str:
    return str(uuid.uuid4())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ExperimentSnapshot:
    """
    Flat, serializable snapshot of one architecture + training + evaluation run.

    All fields are stdlib types — no enums, no nested dataclasses — so
    json.dumps works with no custom encoder.

    Fields
    ------
    name : str
        Human-readable label for the run (e.g. "SmallCNN_v1").
    architecture_summary : dict
        Serialized layer list + metadata.
        Keys: num_layers, num_classes, architecture_id, layers (list of dicts).
    training_config : dict
        Serialized TrainingConfig.
        Keys: epochs, learning_rate, optimizer, batch_size, ... (whatever the
        domain VO carries — captured dynamically by experiment_utils).
    results : dict
        Evaluation outputs.
        Keys: final_train_loss, top1_accuracy, top5_accuracy, mean_eval_loss.
        Any key may be None if evaluation was skipped.
    experiment_id : str
        Unique id — matches the Experiment entity id if both are used together.
    created_at : str
        ISO-8601 UTC timestamp.
    tags : dict
        Arbitrary string metadata for filtering/grouping.
    """

    name: str
    architecture_summary: dict
    training_config: dict
    results: dict
    experiment_id: str = field(default_factory=_new_id)
    created_at: str = field(default_factory=_now_iso)
    tags: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "created_at": self.created_at,
            "architecture_summary": self.architecture_summary,
            "training_config": self.training_config,
            "results": self.results,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentSnapshot":
        return cls(
            experiment_id=data["experiment_id"],
            name=data["name"],
            created_at=data["created_at"],
            architecture_summary=data["architecture_summary"],
            training_config=data["training_config"],
            results=data["results"],
            tags=data.get("tags", {}),
        )