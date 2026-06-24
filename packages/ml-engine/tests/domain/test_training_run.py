import pytest
from unittest.mock import MagicMock

from neuroforge_core.domain.entities.architecture import Architecture
from neuroforge_core.domain.entities.training_run import TrainingRun, TrainingStatus
from neuroforge_core.domain.value_objects.training_config import TrainingConfig


@pytest.fixture
def mock_arch():
    return MagicMock(spec=Architecture)


@pytest.fixture
def simple_config():
    return TrainingConfig(learning_rate=0.01, epochs=2)


@pytest.fixture
def pending_run(mock_arch, simple_config):
    return TrainingRun(architecture=mock_arch, config=simple_config)


def test_initial_status_is_pending(pending_run):
    assert pending_run.status == TrainingStatus.PENDING
    assert pending_run.started_at is None
    assert pending_run.completed_at is None


def test_start_sets_running_and_started_at(pending_run):
    pending_run.start()
    assert pending_run.status == TrainingStatus.RUNNING
    assert pending_run.started_at is not None


def test_complete_sets_completed_status(pending_run):
    pending_run.start()
    pending_run.complete()
    assert pending_run.status == TrainingStatus.COMPLETED
    assert pending_run.completed_at is not None


def test_fail_sets_failed_status_and_message(pending_run):
    pending_run.fail("out of memory")
    assert pending_run.status == TrainingStatus.FAILED
    assert pending_run.error_message == "out of memory"
    assert pending_run.completed_at is not None


def test_cannot_start_twice(pending_run):
    pending_run.start()
    with pytest.raises(RuntimeError, match="running"):
        pending_run.start()


def test_cannot_complete_before_start(pending_run):
    with pytest.raises(RuntimeError, match="pending"):
        pending_run.complete()


def test_is_done_true_after_complete(pending_run):
    pending_run.start()
    pending_run.complete()
    assert pending_run.is_done is True


def test_is_done_true_after_fail(pending_run):
    pending_run.fail("boom")
    assert pending_run.is_done is True


def test_duration_seconds_after_complete(pending_run):
    pending_run.start()
    pending_run.complete()
    assert pending_run.duration_seconds is not None
    assert pending_run.duration_seconds >= 0.0