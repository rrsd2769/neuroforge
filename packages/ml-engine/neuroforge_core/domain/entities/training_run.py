import uuid
from dataclasses import dataclass, field
from datetime import datetime , timezone
from enum import Enum
from typing import Optional

from neuroforge_core.domain.entities.architecture import Architecture
from neuroforge_core.domain.value_objects.training_config import TrainingConfig
from neuroforge_core.domain.value_objects.training_metrics import TrainingHistory


class TrainingStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TrainingRun:
    """
    Aggregate root for one training session.

    Enforces valid status transitions:
      PENDING ──start()──▶ RUNNING ──complete()──▶ COMPLETED
                                   └──fail()──────▶ FAILED
    """

    architecture: Architecture
    config: TrainingConfig
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: TrainingStatus = field(default=TrainingStatus.PENDING)
    history: TrainingHistory = field(default_factory=TrainingHistory)
    started_at: Optional[datetime] = field(default=None)
    completed_at: Optional[datetime] = field(default=None)
    error_message: Optional[str] = field(default=None)

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self.status != TrainingStatus.PENDING:
            raise RuntimeError(
                f"Cannot start a run with status '{self.status.value}'"
            )
        self.status = TrainingStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)

    def complete(self) -> None:
        if self.status != TrainingStatus.RUNNING:
            raise RuntimeError(
                f"Cannot complete a run with status '{self.status.value}'"
            )
        self.status = TrainingStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)

    def fail(self, error: str) -> None:
        """Mark as failed from any state (used by error handlers)."""
        self.status = TrainingStatus.FAILED
        self.error_message = error
        self.completed_at = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def is_done(self) -> bool:
        return self.status in (TrainingStatus.COMPLETED, TrainingStatus.FAILED)

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at is not None and self.completed_at is not None:
            return (self.completed_at - self.started_at).total_seconds()
        return None