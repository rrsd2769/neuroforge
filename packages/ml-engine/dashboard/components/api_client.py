"""HTTP client wrapping the NeuroForge FastAPI backend."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from typing import Any
import requests

from dashboard.config import CONFIG


class NeuroForgeAPIError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"API error {status_code}: {detail}")


class NeuroForgeClient:
    """HTTP wrapper around the NeuroForge FastAPI backend."""

    def __init__(
        self,
        base_url: str = CONFIG.api_base_url,
        timeout: int = CONFIG.api_timeout,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        try:
            resp = self._session.get(
                f"{self.base_url}/health", timeout=CONFIG.health_timeout
            )
            return resp.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Experiments — the single "run everything" endpoint
    # ------------------------------------------------------------------

    def run_experiment(
        self,
        name: str,
        architecture: dict[str, Any],
        training_config: dict[str, Any],
        dataset_config: dict[str, Any],
        tags: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Submit a full experiment: architecture definition + training config.

        architecture shape:
            { "num_classes": 10, "layers": [ { "type": "conv", ... }, ... ] }

        training_config shape:
            { "epochs": 5, "learning_rate": 0.001, "optimizer": "adam",
              "weight_decay": 0.0, "momentum": 0.9 }

        dataset_config shape:
            { "train_samples": 500, "test_samples": 100 }
        """
        payload = {
            "name": name,
            "architecture": architecture,
            "training_config": training_config,
            "dataset_config": dataset_config,
            "tags": tags or {},
        }
        return self._post("/experiments/run", payload)

    def list_experiments(self) -> list[dict[str, Any]]:
        resp = self._session.get(
            f"{self.base_url}/experiments", timeout=self.timeout
        )
        self._raise_for_status(resp)
        return resp.json()

    def get_experiment(self, experiment_id: str) -> dict[str, Any]:
        resp = self._session.get(
            f"{self.base_url}/experiments/{experiment_id}",
            timeout=10,
        )
        self._raise_for_status(resp)
        return resp.json()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        resp = self._session.post(
            f"{self.base_url}{path}", json=payload, timeout=self.timeout
        )
        self._raise_for_status(resp)
        return resp.json()

    @staticmethod
    def _raise_for_status(resp: requests.Response) -> None:
        if not resp.ok:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise NeuroForgeAPIError(resp.status_code, detail)
        

    def run_architecture_search(
        self,
        n_trials: int = 15,
        epochs_per_trial: int = 2,
        train_samples_per_trial: int = 1000,
        test_samples_per_trial: int = 300,
        min_depth: int = 2,
        max_depth: int = 6,
        channel_choices: list[int] | None = None,
    ) -> dict:
        payload = {
            "n_trials": n_trials,
            "epochs_per_trial": epochs_per_trial,
            "train_samples_per_trial": train_samples_per_trial,
            "test_samples_per_trial": test_samples_per_trial,
            "min_depth": min_depth,
            "max_depth": max_depth,
            "channel_choices": channel_choices or [16, 32, 64, 128],
        }
        return self._post("/architecture-search/run", payload)


    def get_architecture_search(self, search_id: str) -> dict:  
        resp = self._session.get(
            f"{self.base_url}/architecture-search/{search_id}", timeout=10,
        )
        self._raise_for_status(resp)
        return resp.json()