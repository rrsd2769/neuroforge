"""
ExperimentSnapshot — a flat, JSON-serializable record of one full training run.
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

    status values:
        "pending"   — submitted, training not yet started
        "running"   — training in progress
        "completed" — training and evaluation finished successfully
        "failed"    — training raised an exception
    """

    name: str
    architecture_summary: dict
    training_config: dict
    results: dict

    # New field — default "completed" so existing snapshots load correctly
    status: str = field(default="completed")

    experiment_id: str = field(default_factory=_new_id)
    created_at: str = field(default_factory=_now_iso)
    tags: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "created_at": self.created_at,
            "status": self.status,
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
            # Use .get() with "completed" default for backward compat
            status=data.get("status", "completed"),
            architecture_summary=data["architecture_summary"],
            training_config=data["training_config"],
            results=data["results"],
            tags=data.get("tags", {}),
        )