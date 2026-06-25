"""
Tests for Day 8: FastAPI REST layer.

Strategy:
- All tests use dependency_overrides to inject a tmp_path-isolated tracker.
- The /run endpoint test actually trains a tiny model (1 epoch, 50 samples).
  Mark it as slow so it can be skipped: pytest -m "not slow"
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from neuroforge_core.application.use_cases.experiment_tracking_use_case import (
    ExperimentTrackingUseCase,
)
from neuroforge_core.domain.entities.experiment_snapshot import ExperimentSnapshot
from neuroforge_core.infrastructure.adapters.file_experiment_tracker import (
    FileExperimentTracker,
)

from api.dependencies import get_tracking_use_case
from api.main import app


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def tracker(tmp_path: Path) -> FileExperimentTracker:
    return FileExperimentTracker(storage_root=tmp_path / "experiments")


@pytest.fixture
def client(tracker: FileExperimentTracker):
    """TestClient with the real tracker replaced by a tmp_path-isolated one."""
    use_case = ExperimentTrackingUseCase(repository=tracker)
    app.dependency_overrides[get_tracking_use_case] = lambda: use_case
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _make_snapshot(name: str, loss: float = 1.5) -> ExperimentSnapshot:
    return ExperimentSnapshot(
        name=name,
        architecture_summary={"num_layers": 2, "num_classes": 10, "layers": []},
        training_config={"epochs": 1, "learning_rate": 0.001, "optimizer": "adam"},
        results={
            "final_train_loss": loss,
            "top1_accuracy": 0.30,
            "top5_accuracy": 0.75,
            "mean_eval_loss": 1.8,
        },
    )


# ------------------------------------------------------------------ #
# Health
# ------------------------------------------------------------------ #

class TestHealth:

    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_body(self, client):
        r = client.get("/health")
        data = r.json()
        assert data["status"] == "ok"
        assert "version" in data


# ------------------------------------------------------------------ #
# GET /experiments  (list)
# ------------------------------------------------------------------ #

class TestListExperiments:

    def test_empty_list(self, client):
        r = client.get("/experiments")
        assert r.status_code == 200
        assert r.json() == []

    def test_returns_saved_snapshots(self, client, tracker):
        tracker.save(_make_snapshot("Alpha"))
        tracker.save(_make_snapshot("Beta"))
        r = client.get("/experiments")
        assert r.status_code == 200
        names = [s["name"] for s in r.json()]
        assert set(names) == {"Alpha", "Beta"}

    def test_response_schema_keys(self, client, tracker):
        tracker.save(_make_snapshot("Check"))
        r = client.get("/experiments")
        item = r.json()[0]
        assert "experiment_id" in item
        assert "name" in item
        assert "results" in item
        assert "created_at" in item


# ------------------------------------------------------------------ #
# GET /experiments/{id}
# ------------------------------------------------------------------ #

class TestGetExperiment:

    def test_get_existing(self, client, tracker):
        snap = _make_snapshot("FindMe")
        tracker.save(snap)
        r = client.get(f"/experiments/{snap.experiment_id}")
        assert r.status_code == 200
        assert r.json()["name"] == "FindMe"

    def test_get_missing_returns_404(self, client):
        r = client.get("/experiments/no-such-id")
        assert r.status_code == 404

    def test_404_detail_mentions_id(self, client):
        r = client.get("/experiments/ghost-id")
        assert "ghost-id" in r.json()["detail"]


# ------------------------------------------------------------------ #
# DELETE /experiments/{id}
# ------------------------------------------------------------------ #

class TestDeleteExperiment:

    def test_delete_existing_returns_204(self, client, tracker):
        snap = _make_snapshot("DeleteMe")
        tracker.save(snap)
        r = client.delete(f"/experiments/{snap.experiment_id}")
        assert r.status_code == 204

    def test_deleted_experiment_not_in_list(self, client, tracker):
        snap = _make_snapshot("Gone")
        tracker.save(snap)
        client.delete(f"/experiments/{snap.experiment_id}")
        r = client.get("/experiments")
        assert r.json() == []

    def test_delete_nonexistent_returns_204(self, client):
        # Silent no-op — idempotent delete
        r = client.delete("/experiments/ghost-id")
        assert r.status_code == 204


# ------------------------------------------------------------------ #
# POST /experiments/compare
# ------------------------------------------------------------------ #

class TestCompareExperiments:

    def test_compare_returns_200(self, client, tracker):
        s1 = _make_snapshot("M1", loss=1.2)
        s2 = _make_snapshot("M2", loss=1.8)
        tracker.save(s1)
        tracker.save(s2)
        r = client.post(
            "/experiments/compare",
            json={"ids": [s1.experiment_id, s2.experiment_id]},
        )
        assert r.status_code == 200

    def test_compare_response_has_rows(self, client, tracker):
        s1 = _make_snapshot("A")
        s2 = _make_snapshot("B")
        tracker.save(s1)
        tracker.save(s2)
        r = client.post(
            "/experiments/compare",
            json={"ids": [s1.experiment_id, s2.experiment_id]},
        )
        rows = r.json()["rows"]
        assert isinstance(rows, list)
        assert len(rows) > 0

    def test_compare_rows_have_metric_key(self, client, tracker):
        s1 = _make_snapshot("X")
        s2 = _make_snapshot("Y")
        tracker.save(s1)
        tracker.save(s2)
        r = client.post(
            "/experiments/compare",
            json={"ids": [s1.experiment_id, s2.experiment_id]},
        )
        rows = r.json()["rows"]
        assert all("Metric" in row for row in rows)

    def test_compare_missing_id_returns_404(self, client, tracker):
        snap = _make_snapshot("Real")
        tracker.save(snap)
        r = client.post(
            "/experiments/compare",
            json={"ids": [snap.experiment_id, "fake-id"]},
        )
        assert r.status_code == 404

    def test_compare_requires_at_least_two_ids(self, client):
        r = client.post("/experiments/compare", json={"ids": ["only-one"]})
        assert r.status_code == 422


# ------------------------------------------------------------------ #
# POST /experiments/run  (integration — actually trains a model)
# ------------------------------------------------------------------ #

MINIMAL_RUN_BODY = {
    "name": "APITest_SmallCNN",
    "architecture": {
        "num_classes": 10,
        "layers": [
            {"type": "conv", "out_channels": 16, "kernel_size": 3},
            {"type": "pool", "pool_size": 2},
            {"type": "flatten"},
            {"type": "dense", "units": 10, "activation": "none"},
        ],
    },
    "training_config": {
        "epochs": 1,
        "learning_rate": 0.001,
        "optimizer": "adam",
    },
    "dataset_config": {
        "train_samples": 64,
        "test_samples": 32,
    },
    "tags": {"source": "api_test"},
}


@pytest.mark.slow
class TestRunExperiment:

    def test_run_returns_201(self, client):
        r = client.post("/experiments/run", json=MINIMAL_RUN_BODY)
        assert r.status_code == 201

    def test_run_response_has_required_fields(self, client):
        r = client.post("/experiments/run", json=MINIMAL_RUN_BODY)
        data = r.json()
        assert "experiment_id" in data
        assert "name" in data
        assert "results" in data
        assert data["name"] == "APITest_SmallCNN"

    def test_run_results_contain_loss(self, client):
        r = client.post("/experiments/run", json=MINIMAL_RUN_BODY)
        results = r.json()["results"]
        assert "final_train_loss" in results
        assert isinstance(results["final_train_loss"], float)

    def test_run_results_contain_accuracy(self, client):
        r = client.post("/experiments/run", json=MINIMAL_RUN_BODY)
        results = r.json()["results"]
        assert "top1_accuracy" in results
        assert results["top1_accuracy"] is not None

    def test_run_persists_experiment(self, client):
        r = client.post("/experiments/run", json=MINIMAL_RUN_BODY)
        experiment_id = r.json()["experiment_id"]
        # Fetch it back
        r2 = client.get(f"/experiments/{experiment_id}")
        assert r2.status_code == 200
        assert r2.json()["experiment_id"] == experiment_id

    def test_run_appears_in_list(self, client):
        client.post("/experiments/run", json=MINIMAL_RUN_BODY)
        r = client.get("/experiments")
        assert len(r.json()) == 1

    def test_run_invalid_optimizer_returns_422(self, client):
        body = dict(MINIMAL_RUN_BODY)
        body["training_config"] = {"epochs": 1, "optimizer": "invalid_opt"}
        r = client.post("/experiments/run", json=body)
        assert r.status_code == 422

    def test_run_tags_preserved(self, client):
        r = client.post("/experiments/run", json=MINIMAL_RUN_BODY)
        assert r.json()["tags"] == {"source": "api_test"}