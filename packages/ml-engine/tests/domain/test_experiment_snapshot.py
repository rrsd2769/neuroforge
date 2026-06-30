"""Tests for ExperimentSnapshot status field."""
from neuroforge_core.domain.entities.experiment_snapshot import ExperimentSnapshot


def test_default_status_is_completed():
    snap = ExperimentSnapshot(
        name="Test",
        architecture_summary={},
        training_config={},
        results={},
    )
    assert snap.status == "completed"


def test_status_roundtrips_through_dict():
    snap = ExperimentSnapshot(
        name="Test",
        architecture_summary={},
        training_config={},
        results={},
        status="running",
    )
    restored = ExperimentSnapshot.from_dict(snap.to_dict())
    assert restored.status == "running"


def test_from_dict_backward_compat_no_status_key():
    """Existing snapshots on disk have no status key — must not raise."""
    data = {
        "experiment_id": "abc-123",
        "name": "OldRun",
        "created_at": "2026-01-01T00:00:00+00:00",
        "architecture_summary": {},
        "training_config": {},
        "results": {"top1_accuracy": 0.5},
        "tags": {},
        # NOTE: no "status" key — simulates a pre-Day-12 snapshot file
    }
    snap = ExperimentSnapshot.from_dict(data)
    assert snap.status == "completed"  # default applied


def test_all_status_values_roundtrip():
    for status in ("pending", "running", "completed", "failed"):
        snap = ExperimentSnapshot(
            name="Test", architecture_summary={},
            training_config={}, results={}, status=status,
        )
        assert ExperimentSnapshot.from_dict(snap.to_dict()).status == status