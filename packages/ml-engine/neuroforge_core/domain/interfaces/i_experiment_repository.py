"""Port: contract for persisting and retrieving ExperimentSnapshots."""
from __future__ import annotations

from abc import ABC, abstractmethod

from neuroforge_core.domain.entities.experiment_snapshot import ExperimentSnapshot


class IExperimentRepository(ABC):

    @abstractmethod
    def save(self, snapshot: ExperimentSnapshot) -> None:
        """Persist a snapshot. Overwrites if experiment_id already exists."""

    @abstractmethod
    def load(self, experiment_id: str) -> ExperimentSnapshot:
        """
        Load a snapshot by experiment_id.
        Raises KeyError if not found.
        """

    @abstractmethod
    def list_all(self) -> list[ExperimentSnapshot]:
        """Return all persisted snapshots, sorted by created_at ascending."""

    @abstractmethod
    def delete(self, experiment_id: str) -> None:
        """Remove a snapshot. Silent no-op if not found."""