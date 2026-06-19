"""
Experiment domain entity.

An Experiment is the top-level record of one full NeuroForge run: which
architecture was generated, what happened to it, and how it ended. This is
what FileExperimentTracker (Day 7) persists to experiments/runs/<id>/.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class ExperimentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Experiment:
    """
    Top-level record of a single NeuroForge run.

    `experiment_id` is generated once at creation and used as the key for
    checkpoint paths and the experiment-log directory — deterministic per
    run (not time-based), so a run is always traceable from its id alone.
    """

    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: ExperimentStatus = ExperimentStatus.PENDING
    architecture_id: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: str | None = None
    error_message: str | None = None

    def mark_running(self) -> None:
        self.status = ExperimentStatus.RUNNING

    def mark_completed(self) -> None:
        self.status = ExperimentStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def mark_failed(self, error_message: str) -> None:
        self.status = ExperimentStatus.FAILED
        self.error_message = error_message
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "status": self.status.value,
            "architecture_id": self.architecture_id,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error_message": self.error_message,
        }