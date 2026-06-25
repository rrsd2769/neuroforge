"""
FileExperimentTracker — JSON file adapter for IExperimentRepository.

Storage layout (matches the convention named in experiment.py docstring):

    experiments/
      runs/
        {experiment_id}/
          snapshot.json       ← ExperimentSnapshot.to_dict()

One sub-directory per experiment gives us room to add loss_curve.json,
checkpoints/, etc. in later days without restructuring.
"""
from __future__ import annotations

import json
from pathlib import Path

from neuroforge_core.domain.entities.experiment_snapshot import ExperimentSnapshot
from neuroforge_core.domain.interfaces.i_experiment_repository import IExperimentRepository

_SNAPSHOT_FILENAME = "snapshot.json"


class FileExperimentTracker(IExperimentRepository):
    """
    Persists ExperimentSnapshots as JSON files under a configurable root
    directory.

    Parameters
    ----------
    storage_root : str | Path
        Top-level directory.  Sub-path ``runs/`` is appended automatically.
        Default: ``"experiments"`` (relative to CWD).
    """

    def __init__(self, storage_root: str | Path = "experiments") -> None:
        self._runs_dir = Path(storage_root) / "runs"

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _run_dir(self, experiment_id: str) -> Path:
        return self._runs_dir / experiment_id

    def _snapshot_path(self, experiment_id: str) -> Path:
        return self._run_dir(experiment_id) / _SNAPSHOT_FILENAME

    def _ensure_run_dir(self, experiment_id: str) -> None:
        self._run_dir(experiment_id).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # IExperimentRepository
    # ------------------------------------------------------------------ #

    def save(self, snapshot: ExperimentSnapshot) -> None:
        self._ensure_run_dir(snapshot.experiment_id)
        path = self._snapshot_path(snapshot.experiment_id)
        path.write_text(
            json.dumps(snapshot.to_dict(), indent=2),
            encoding="utf-8",
        )

    def load(self, experiment_id: str) -> ExperimentSnapshot:
        path = self._snapshot_path(experiment_id)
        if not path.exists():
            raise KeyError(
                f"No experiment snapshot found for id: {experiment_id!r}\n"
                f"Expected at: {path}"
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        return ExperimentSnapshot.from_dict(data)

    def list_all(self) -> list[ExperimentSnapshot]:
        if not self._runs_dir.exists():
            return []

        snapshots: list[ExperimentSnapshot] = []
        for run_dir in self._runs_dir.iterdir():
            if not run_dir.is_dir():
                continue
            snapshot_path = run_dir / _SNAPSHOT_FILENAME
            if not snapshot_path.exists():
                continue
            try:
                data = json.loads(snapshot_path.read_text(encoding="utf-8"))
                snapshots.append(ExperimentSnapshot.from_dict(data))
            except (json.JSONDecodeError, KeyError):
                # Corrupted or incomplete file — skip gracefully
                continue

        snapshots.sort(key=lambda s: s.created_at)
        return snapshots

    def delete(self, experiment_id: str) -> None:
        path = self._snapshot_path(experiment_id)
        if path.exists():
            path.unlink()
        # Leave the run directory — it may hold other artifacts in later days