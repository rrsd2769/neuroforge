"""
ExperimentTrackerPort.

Abstract contract for persisting and retrieving Experiment records.
Implemented as FileExperimentTracker (Day 7); swapped later for a
PostgreSQL-backed adapter once apps/api exists — same port, zero changes
to any use case that depends on it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from neuroforge_core.domain.entities.experiment import Experiment


class ExperimentTrackerPort(ABC):
    @abstractmethod
    def log_experiment(self, experiment: Experiment, **artifacts: Any) -> None:
        """Persists an Experiment along with associated artifacts (metrics, evaluation, model)."""
        raise NotImplementedError

    @abstractmethod
    def get_experiment(self, experiment_id: str) -> Experiment:
        """Retrieves a previously logged Experiment by id."""
        raise NotImplementedError