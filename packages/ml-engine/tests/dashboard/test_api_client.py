"""Unit tests for NeuroForgeClient — all HTTP calls are mocked."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unittest.mock import MagicMock, patch
import pytest
import requests

from dashboard.components.api_client import NeuroForgeClient, NeuroForgeAPIError


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def client() -> NeuroForgeClient:
    return NeuroForgeClient(base_url="http://test-api:8000", timeout=10)


def _mock_response(status_code: int = 200, json_data=None) -> MagicMock:
    mock = MagicMock()
    mock.status_code = status_code
    mock.ok = 200 <= status_code < 300
    mock.json.return_value = json_data if json_data is not None else {}
    mock.text = str(json_data)
    return mock


# ── Realistic sample data matching actual snapshot.json shape ─────────────────

SAMPLE_EXPERIMENT = {
    "experiment_id": "48b39a74-bfb0-42bb-b42d-d881a21348e9",
    "name": "ManualTest",
    "created_at": "2026-06-25T09:28:25.748898+00:00",
    "architecture_summary": {
        "architecture_id": "1a3f28c1-fc52-4fd6-9b92-36f6374f50de",
        "num_layers": 4,
        "num_classes": 10,
        "layers": [
            {"type": "conv", "out_channels": 16, "kernel_size": 3, "stride": 1,
             "padding": 1, "activation": "relu"},
            {"type": "pool", "pool_size": 2, "stride": 2, "pool_type": "max"},
            {"type": "flatten"},
            {"type": "dense", "units": 10, "activation": "none"},
        ],
    },
    "training_config": {
        "learning_rate": 0.001,
        "epochs": 1,
        "optimizer": "adam",
        "weight_decay": 0.0,
        "momentum": 0.9,
    },
    "results": {
        "final_train_loss": 2.371849,
        "top1_accuracy": 0.08,
        "top5_accuracy": 0.42,
        "mean_eval_loss": 2.474097,
    },
    "tags": {},
}

SAMPLE_RUN_PAYLOAD = {
    "name": "TestRun",
    "architecture": {
        "num_classes": 10,
        "layers": [
            {"type": "conv", "out_channels": 32, "kernel_size": 3,
             "stride": 1, "padding": 1, "activation": "relu"},
            {"type": "flatten"},
            {"type": "dense", "units": 10, "activation": "none"},
        ],
    },
    "training_config": {
        "epochs": 2,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "weight_decay": 0.0,
        "momentum": 0.9,
    },
    "dataset_config": {"train_samples": 500, "test_samples": 100},
    "tags": {},
}


# ── health_check ─────────────────────────────────────────────────────────────

class TestHealthCheck:
    def test_returns_true_on_200(self, client):
        with patch.object(client._session, "get", return_value=_mock_response(200)):
            assert client.health_check() is True

    def test_returns_false_on_500(self, client):
        with patch.object(client._session, "get", return_value=_mock_response(500)):
            assert client.health_check() is False

    def test_returns_false_on_connection_error(self, client):
        with patch.object(client._session, "get", side_effect=requests.ConnectionError()):
            assert client.health_check() is False


# ── run_experiment ────────────────────────────────────────────────────────────

class TestRunExperiment:
    def test_returns_experiment_with_results(self, client):
        with patch.object(client._session, "post",
                          return_value=_mock_response(200, SAMPLE_EXPERIMENT)):
            result = client.run_experiment(
                name="TestRun",
                architecture=SAMPLE_RUN_PAYLOAD["architecture"],
                training_config=SAMPLE_RUN_PAYLOAD["training_config"],
                dataset_config=SAMPLE_RUN_PAYLOAD["dataset_config"],
            )
        # Day 8 uses experiment_id, not id
        assert "experiment_id" in result
        assert "results" in result
        assert result["results"]["top1_accuracy"] == pytest.approx(0.08)
        assert result["results"]["top5_accuracy"] == pytest.approx(0.42)

    def test_sends_correct_payload_structure(self, client):
        with patch.object(client._session, "post",
                          return_value=_mock_response(200, SAMPLE_EXPERIMENT)) as mock_post:
            client.run_experiment(
                name="TestRun",
                architecture=SAMPLE_RUN_PAYLOAD["architecture"],
                training_config=SAMPLE_RUN_PAYLOAD["training_config"],
                dataset_config=SAMPLE_RUN_PAYLOAD["dataset_config"],
            )
        body = mock_post.call_args.kwargs["json"]
        assert "name" in body
        assert "architecture" in body
        assert "training_config" in body
        assert "dataset_config" in body
        assert "tags" in body

    def test_posts_to_experiments_run(self, client):
        with patch.object(client._session, "post",
                          return_value=_mock_response(200, SAMPLE_EXPERIMENT)) as mock_post:
            client.run_experiment(
                name="X",
                architecture={"num_classes": 10, "layers": []},
                training_config={"epochs": 1, "learning_rate": 0.001,
                                  "optimizer": "adam", "weight_decay": 0.0, "momentum": 0.9},
                dataset_config={"train_samples": 100, "test_samples": 50},
            )
        url = mock_post.call_args.args[0]
        assert url.endswith("/experiments/run")

    def test_raises_on_validation_error(self, client):
        err = {"detail": "Invalid architecture"}
        with patch.object(client._session, "post",
                          return_value=_mock_response(422, err)):
            with pytest.raises(NeuroForgeAPIError) as exc_info:
                client.run_experiment(
                    name="Bad",
                    architecture={},
                    training_config={},
                    dataset_config={},
                )
        assert exc_info.value.status_code == 422

    def test_default_tags_is_empty_dict(self, client):
        with patch.object(client._session, "post",
                          return_value=_mock_response(200, SAMPLE_EXPERIMENT)) as mock_post:
            client.run_experiment(
                name="X",
                architecture={"num_classes": 10, "layers": []},
                training_config={"epochs": 1, "learning_rate": 0.001,
                                  "optimizer": "adam", "weight_decay": 0.0, "momentum": 0.9},
                dataset_config={"train_samples": 100, "test_samples": 50},
            )
        body = mock_post.call_args.kwargs["json"]
        assert body["tags"] == {}


# ── list_experiments ──────────────────────────────────────────────────────────

class TestListExperiments:
    def test_returns_list(self, client):
        data = [SAMPLE_EXPERIMENT]
        with patch.object(client._session, "get", return_value=_mock_response(200, data)):
            result = client.list_experiments()
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["experiment_id"] == SAMPLE_EXPERIMENT["experiment_id"]

    def test_hits_correct_path(self, client):
        with patch.object(client._session, "get",
                          return_value=_mock_response(200, [])) as mock_get:
            client.list_experiments()
        url = mock_get.call_args.args[0]
        assert url.endswith("/experiments")
        # Must NOT have /api/v1 prefix — Day 8 routes directly
        assert "/api/v1/" not in url

    def test_raises_on_server_error(self, client):
        with patch.object(client._session, "get",
                          return_value=_mock_response(500, {"detail": "Internal error"})):
            with pytest.raises(NeuroForgeAPIError):
                client.list_experiments()

    def test_returns_empty_list(self, client):
        with patch.object(client._session, "get", return_value=_mock_response(200, [])):
            result = client.list_experiments()
        assert result == []


# ── get_experiment ────────────────────────────────────────────────────────────

class TestGetExperiment:
    def test_returns_experiment_dict(self, client):
        with patch.object(client._session, "get",
                          return_value=_mock_response(200, SAMPLE_EXPERIMENT)):
            result = client.get_experiment("48b39a74-bfb0-42bb-b42d-d881a21348e9")
        assert result["experiment_id"] == SAMPLE_EXPERIMENT["experiment_id"]
        assert result["name"] == "ManualTest"

    def test_results_field_present(self, client):
        with patch.object(client._session, "get",
                          return_value=_mock_response(200, SAMPLE_EXPERIMENT)):
            result = client.get_experiment("48b39a74")
        assert "results" in result
        assert "top1_accuracy" in result["results"]
        assert "top5_accuracy" in result["results"]
        assert "final_train_loss" in result["results"]
        assert "mean_eval_loss" in result["results"]

    def test_hits_correct_path(self, client):
        exp_id = "48b39a74-bfb0-42bb-b42d-d881a21348e9"
        with patch.object(client._session, "get",
                          return_value=_mock_response(200, SAMPLE_EXPERIMENT)) as mock_get:
            client.get_experiment(exp_id)
        url = mock_get.call_args.args[0]
        assert url.endswith(f"/experiments/{exp_id}")

    def test_raises_on_not_found(self, client):
        with patch.object(client._session, "get",
                          return_value=_mock_response(404, {"detail": "Not found"})):
            with pytest.raises(NeuroForgeAPIError) as exc_info:
                client.get_experiment("nonexistent")
        assert exc_info.value.status_code == 404

    def test_architecture_summary_field_present(self, client):
        with patch.object(client._session, "get",
                          return_value=_mock_response(200, SAMPLE_EXPERIMENT)):
            result = client.get_experiment("48b39a74")
        # Day 8 uses architecture_summary, not architecture
        assert "architecture_summary" in result
        assert "num_layers" in result["architecture_summary"]