"""
Tests for Day 7: Experiment Tracking

Covers:
- ExperimentSnapshot serialization roundtrip
- Experiment.from_dict (the new classmethod)
- FileExperimentTracker CRUD + edge cases
- ExperimentTrackingUseCase orchestration
- compare() output structure
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from neuroforge_core.domain.entities.experiment_snapshot import ExperimentSnapshot
from neuroforge_core.infrastructure.adapters.file_experiment_tracker import (
    FileExperimentTracker,
)
from neuroforge_core.application.use_cases.experiment_tracking_use_case import (
    ExperimentTrackingUseCase,
)


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def sample_snapshot() -> ExperimentSnapshot:
    return ExperimentSnapshot(
        name="TestRun_v1",
        architecture_summary={
            "architecture_id": "arch-abc",
            "num_layers": 3,
            "num_classes": 10,
            "layers": [
                {"type": "conv", "out_channels": 32, "kernel_size": 3, "activation": "relu"},
                {"type": "flatten"},
                {"type": "dense", "out_features": 10, "activation": "none"},
            ],
        },
        training_config={
            "epochs": 5,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "batch_size": 64,
        },
        results={
            "final_train_loss": 1.2345,
            "top1_accuracy": 0.432,
            "top5_accuracy": 0.876,
            "mean_eval_loss": 1.456,
        },
        tags={"variant": "baseline"},
    )


@pytest.fixture
def tracker(tmp_path: Path) -> FileExperimentTracker:
    return FileExperimentTracker(storage_root=tmp_path / "experiments")


@pytest.fixture
def use_case(tracker: FileExperimentTracker) -> ExperimentTrackingUseCase:
    return ExperimentTrackingUseCase(repository=tracker)


def _make_snapshot(name: str, loss: float = 1.0) -> ExperimentSnapshot:
    return ExperimentSnapshot(
        name=name,
        architecture_summary={"num_layers": 2, "num_classes": 10, "layers": []},
        training_config={"epochs": 1, "learning_rate": 0.01, "optimizer": "sgd", "batch_size": 32},
        results={"final_train_loss": loss, "top1_accuracy": 0.25, "top5_accuracy": 0.70, "mean_eval_loss": 1.5},
    )


# ------------------------------------------------------------------ #
# ExperimentSnapshot serialization
# ------------------------------------------------------------------ #

class TestExperimentSnapshot:

    def test_to_dict_contains_all_keys(self, sample_snapshot):
        d = sample_snapshot.to_dict()
        assert set(d.keys()) == {
            "experiment_id", "name", "created_at", "status",
            "architecture_summary", "training_config", "results", "tags",
        }

    def test_from_dict_roundtrip(self, sample_snapshot):
        d = sample_snapshot.to_dict()
        restored = ExperimentSnapshot.from_dict(d)
        assert restored.experiment_id == sample_snapshot.experiment_id
        assert restored.name == sample_snapshot.name
        assert restored.results == sample_snapshot.results
        assert restored.tags == sample_snapshot.tags

    def test_default_id_is_unique(self):
        a = ExperimentSnapshot(name="A", architecture_summary={}, training_config={}, results={})
        b = ExperimentSnapshot(name="B", architecture_summary={}, training_config={}, results={})
        assert a.experiment_id != b.experiment_id

    def test_default_created_at_is_set(self):
        s = ExperimentSnapshot(name="X", architecture_summary={}, training_config={}, results={})
        assert s.created_at  # non-empty
        assert "T" in s.created_at  # ISO 8601

    def test_to_dict_is_json_serializable(self, sample_snapshot):
        # Must not raise
        json_str = json.dumps(sample_snapshot.to_dict())
        assert len(json_str) > 10

    def test_tags_default_to_empty_dict(self):
        s = ExperimentSnapshot(name="Y", architecture_summary={}, training_config={}, results={})
        assert s.tags == {}

    def test_from_dict_missing_tags_defaults_to_empty(self, sample_snapshot):
        d = sample_snapshot.to_dict()
        del d["tags"]
        restored = ExperimentSnapshot.from_dict(d)
        assert restored.tags == {}



# ------------------------------------------------------------------ #
# FileExperimentTracker
# ------------------------------------------------------------------ #

class TestFileExperimentTracker:

    def test_save_creates_snapshot_json(self, tracker, sample_snapshot, tmp_path):
        tracker.save(sample_snapshot)
        expected = (
            tmp_path / "experiments" / "runs" / sample_snapshot.experiment_id / "snapshot.json"
        )
        assert expected.exists()

    def test_load_returns_correct_snapshot(self, tracker, sample_snapshot):
        tracker.save(sample_snapshot)
        loaded = tracker.load(sample_snapshot.experiment_id)
        assert loaded.experiment_id == sample_snapshot.experiment_id
        assert loaded.name == sample_snapshot.name
        assert loaded.results == sample_snapshot.results

    def test_load_missing_id_raises_key_error(self, tracker):
        with pytest.raises(KeyError, match="no-such-id"):
            tracker.load("no-such-id")

    def test_list_all_empty_returns_empty_list(self, tracker):
        assert tracker.list_all() == []

    def test_list_all_returns_all_saved(self, tracker):
        s1 = _make_snapshot("Run1")
        s2 = _make_snapshot("Run2")
        tracker.save(s1)
        tracker.save(s2)
        all_snapshots = tracker.list_all()
        assert len(all_snapshots) == 2
        names = {s.name for s in all_snapshots}
        assert names == {"Run1", "Run2"}

    def test_list_all_sorted_by_created_at(self, tracker):
        s1 = _make_snapshot("Early")
        time.sleep(0.05)
        s2 = _make_snapshot("Late")
        tracker.save(s2)
        tracker.save(s1)
        all_snapshots = tracker.list_all()
        assert all_snapshots[0].name == "Early"
        assert all_snapshots[1].name == "Late"

    def test_save_overwrites_existing(self, tracker, sample_snapshot):
        tracker.save(sample_snapshot)
        # mutate and re-save
        sample_snapshot.results["final_train_loss"] = 9.999
        tracker.save(sample_snapshot)
        loaded = tracker.load(sample_snapshot.experiment_id)
        assert loaded.results["final_train_loss"] == 9.999

    def test_delete_removes_snapshot(self, tracker, sample_snapshot):
        tracker.save(sample_snapshot)
        tracker.delete(sample_snapshot.experiment_id)
        with pytest.raises(KeyError):
            tracker.load(sample_snapshot.experiment_id)

    def test_delete_nonexistent_is_silent(self, tracker):
        # Must not raise
        tracker.delete("ghost-id")

    def test_list_all_skips_corrupted_file(self, tracker, sample_snapshot, tmp_path):
        tracker.save(sample_snapshot)
        # Corrupt the file
        snap_path = (
            tmp_path / "experiments" / "runs" / sample_snapshot.experiment_id / "snapshot.json"
        )
        snap_path.write_text("{ invalid json !!!}", encoding="utf-8")
        # Should return empty list (corrupted file skipped)
        assert tracker.list_all() == []


# ------------------------------------------------------------------ #
# ExperimentTrackingUseCase
# ------------------------------------------------------------------ #

class TestExperimentTrackingUseCase:

    def test_save_and_load_roundtrip(self, use_case, sample_snapshot):
        use_case.save(sample_snapshot)
        loaded = use_case.load(sample_snapshot.experiment_id)
        assert loaded.name == sample_snapshot.name

    def test_save_returns_snapshot(self, use_case, sample_snapshot):
        returned = use_case.save(sample_snapshot)
        assert returned is sample_snapshot

    def test_list_all_reflects_saved(self, use_case):
        s1 = _make_snapshot("Alpha")
        s2 = _make_snapshot("Beta")
        use_case.save(s1)
        use_case.save(s2)
        all_s = use_case.list_all()
        assert len(all_s) == 2

    def test_compare_returns_list_of_dicts(self, use_case):
        s1 = _make_snapshot("ModelA")
        s2 = _make_snapshot("ModelB")
        use_case.save(s1)
        use_case.save(s2)
        rows = use_case.compare([s1.experiment_id, s2.experiment_id])
        assert isinstance(rows, list)
        assert len(rows) > 0
        assert all(isinstance(r, dict) for r in rows)

    def test_compare_has_metric_column(self, use_case):
        s1 = _make_snapshot("X")
        s2 = _make_snapshot("Y")
        use_case.save(s1)
        use_case.save(s2)
        rows = use_case.compare([s1.experiment_id, s2.experiment_id])
        assert all("Metric" in r for r in rows)

    def test_compare_has_experiment_name_columns(self, use_case):
        s1 = _make_snapshot("ModelX")
        s2 = _make_snapshot("ModelY")
        use_case.save(s1)
        use_case.save(s2)
        rows = use_case.compare([s1.experiment_id, s2.experiment_id])
        # Each row should have columns named after the experiment
        assert "ModelX" in rows[0]
        assert "ModelY" in rows[0]

    def test_compare_includes_accuracy_row(self, use_case):
        s1 = _make_snapshot("A")
        s2 = _make_snapshot("B")
        use_case.save(s1)
        use_case.save(s2)
        rows = use_case.compare([s1.experiment_id, s2.experiment_id])
        metrics = [r["Metric"] for r in rows]
        assert "Top-1 Accuracy" in metrics

    def test_compare_includes_loss_row(self, use_case):
        s1 = _make_snapshot("A")
        s2 = _make_snapshot("B")
        use_case.save(s1)
        use_case.save(s2)
        rows = use_case.compare([s1.experiment_id, s2.experiment_id])
        metrics = [r["Metric"] for r in rows]
        assert "Final Train Loss" in metrics

    def test_delete_via_use_case(self, use_case, sample_snapshot):
        use_case.save(sample_snapshot)
        use_case.delete(sample_snapshot.experiment_id)
        with pytest.raises(KeyError):
            use_case.load(sample_snapshot.experiment_id)