"""
ModelArtifact domain entity.

Links a trained model's persisted weights back to the architecture and
experiment that produced it, plus basic size statistics reused throughout
the platform (training logs, future benchmarking/compression pipelines).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelArtifact:
    """
    Reference to a trained model's checkpoint on disk.

    `checkpoint_path` is filled in by CheckpointStorage (Day 6).
    `total_params` is filled in by ModelFactory.count_parameters() (Day 5).
    """

    architecture_id: str
    experiment_id: str
    checkpoint_path: str | None = None
    total_params: int | None = None

    def to_dict(self) -> dict:
        return {
            "architecture_id": self.architecture_id,
            "experiment_id": self.experiment_id,
            "checkpoint_path": self.checkpoint_path,
            "total_params": self.total_params,
        }