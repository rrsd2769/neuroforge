"""
FastAPI dependency factories.

Use these with Depends() in route handlers so the concrete storage
backend is injectable (and overridable in tests).
"""
from __future__ import annotations

import os
from functools import lru_cache

from neuroforge_core.application.use_cases.experiment_tracking_use_case import (
    ExperimentTrackingUseCase,
)
from neuroforge_core.infrastructure.adapters.file_experiment_tracker import (
    FileExperimentTracker,
)


@lru_cache(maxsize=1)
def _get_tracker() -> FileExperimentTracker:
    """Singleton tracker — one instance per process."""
    storage_root = os.getenv("NEUROFORGE_EXPERIMENTS_DIR", "experiments")
    return FileExperimentTracker(storage_root=storage_root)


def get_tracking_use_case() -> ExperimentTrackingUseCase:
    """
    FastAPI dependency: returns an ExperimentTrackingUseCase backed by the
    singleton FileExperimentTracker.

    Tests override this via app.dependency_overrides.
    """
    return ExperimentTrackingUseCase(repository=_get_tracker())