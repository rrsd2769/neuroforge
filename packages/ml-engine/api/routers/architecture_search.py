"""
Architecture search routes — Optuna-driven NAS.

POST /architecture-search/run         → queue a multi-trial search, return search_id
GET  /architecture-search/{search_id} → poll status + trial results

Each trial: build architecture (Optuna-suggested) + hyperparameters,
compile, train briefly, evaluate, report accuracy back to Optuna.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import optuna
from fastapi import APIRouter, BackgroundTasks, HTTPException

from neuroforge_core.application.train_model import TrainModelRequest, TrainModelUseCase
from neuroforge_core.domain.value_objects.evaluation_config import EvaluationConfig
from neuroforge_core.domain.value_objects.search_space import SearchSpace
from neuroforge_core.infrastructure.adapters.pytorch_evaluator import PyTorchEvaluator
from neuroforge_core.domain.value_objects.training_config import TrainingConfig, OptimizerType
from neuroforge_core.infrastructure.adapters.pytorch_model_compiler import (
    PyTorchModelCompiler,
)
from neuroforge_core.infrastructure.generators.optuna_architecture_builder import (
    OptunaArchitectureBuilder,
)
from neuroforge_core.infrastructure.training.pytorch_trainer import PyTorchTrainer

from api.routers.experiments import _get_cifar10_loaders, _DEVICE
from api.schemas import ArchitectureSearchRequest, SearchStatusResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/architecture-search", tags=["architecture-search"])

_SEARCH_DIR = Path(os.getenv("NEUROFORGE_SEARCH_DIR", "experiments/architecture_search"))
_SEARCH_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ #
# Lightweight file-backed persistence
#
# Pragmatic scope cut: this is direct file I/O in the router, not a
# proper port/adapter like IExperimentRepository. Promoting this to a
# SearchResultRepository port is the natural next refactor.
# ------------------------------------------------------------------ #

def _search_path(search_id: str) -> Path:
    return _SEARCH_DIR / f"{search_id}.json"


def _save_search_state(state: dict) -> None:
    _search_path(state["search_id"]).write_text(json.dumps(state, indent=2))


def _load_search_state(search_id: str) -> dict:
    path = _search_path(search_id)
    if not path.exists():
        raise KeyError(search_id)
    return json.loads(path.read_text())


# ------------------------------------------------------------------ #
# Background search loop
# ------------------------------------------------------------------ #

def _run_search(search_id: str, body: ArchitectureSearchRequest) -> None:
    state = _load_search_state(search_id)
    state["status"] = "running"
    _save_search_state(state)

    search_space = SearchSpace(
        min_depth=body.min_depth,
        max_depth=body.max_depth,
        channel_choices=tuple(body.channel_choices),
    )
    builder = OptunaArchitectureBuilder(search_space)
    compiler = PyTorchModelCompiler()
    evaluator = PyTorchEvaluator()
    input_shape = (3, 32, 32)

    train_loader, test_loader = _get_cifar10_loaders(
        train_samples=body.train_samples_per_trial,
        test_samples=body.test_samples_per_trial,
    )

    def objective(trial: optuna.Trial) -> float:
        arch = builder.build(trial, num_classes=10, input_shape=input_shape)

        model = compiler.compile(arch, input_channels=3, num_classes=10)
        param_count = sum(p.numel() for p in model.parameters())

        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd"])
        config = TrainingConfig(
            epochs =  body.epochs_per_trial,
            learning_rate =  learning_rate,
            optimizer = OptimizerType(optimizer_name.lower()),
            weight_decay = 0.0001,
            momentum = 0.9,
        )

        trainer = PyTorchTrainer(model=model, device=_DEVICE)
        use_case = TrainModelUseCase(trainer=trainer)
        response = use_case.execute(TrainModelRequest(
            architecture=arch, config=config, train_loader=train_loader,
        ))

        if not response.succeeded:
            raise optuna.TrialPruned(f"Trial {trial.number} training failed.")

        eval_metrics = evaluator.evaluate(
            model=model, data_loader=test_loader, config=EvaluationConfig(top_k=5),
        )

        trial_result = {
            "trial_number": trial.number,
            "top1_accuracy": round(float(eval_metrics.accuracy), 6),
            "top5_accuracy": round(float(eval_metrics.top_k_accuracy), 6),
            "parameter_count": param_count,
            "learning_rate": learning_rate,
            "optimizer": optimizer_name,
            "num_layers": arch.layer_count(),
            "architecture_id": arch.id,
        }
        state["trials"].append(trial_result)
        state["completed_trials"] = len(state["trials"])
        _save_search_state(state)

        logger.info(
            "Search %s trial %d: top1=%.3f params=%d",
            search_id, trial.number, eval_metrics.accuracy, param_count,
        )
        return float(eval_metrics.accuracy)

    try:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=body.n_trials)

        best = max(state["trials"], key=lambda t: t["top1_accuracy"]) if state["trials"] else None
        state["status"] = "completed"
        state["best_trial"] = best
        _save_search_state(state)
        logger.info("Search %s completed. Best top1: %s", search_id,
                    best["top1_accuracy"] if best else "N/A")

    except Exception as exc:
        logger.exception("Search %s failed: %s", search_id, exc)
        state["status"] = "failed"
        state["error"] = str(exc)
        _save_search_state(state)


# ------------------------------------------------------------------ #
# POST /architecture-search/run
# ------------------------------------------------------------------ #

@router.post("/run", response_model=SearchStatusResponse, status_code=201)
def run_search(
    body: ArchitectureSearchRequest,
    background_tasks: BackgroundTasks,
) -> SearchStatusResponse:
    """Queue an Optuna-driven architecture search. Returns immediately."""
    search_id = str(uuid.uuid4())
    state = {
        "search_id": search_id,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_trials": body.n_trials,
        "completed_trials": 0,
        "trials": [],
        "best_trial": None,
        "error": None,
    }
    _save_search_state(state)

    background_tasks.add_task(_run_search, search_id, body)

    logger.info("Search %s queued: %d trials", search_id, body.n_trials)
    return SearchStatusResponse(**state)


# ------------------------------------------------------------------ #
# GET /architecture-search/{search_id}
# ------------------------------------------------------------------ #

@router.get("/{search_id}", response_model=SearchStatusResponse)
def get_search(search_id: str) -> SearchStatusResponse:
    try:
        state = _load_search_state(search_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Search not found: {search_id!r}")
    return SearchStatusResponse(**state)