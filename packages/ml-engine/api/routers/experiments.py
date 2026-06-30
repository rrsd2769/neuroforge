"""
Experiment routes.

POST  /experiments/run        → validate → save pending → background train → 201
GET   /experiments            → list all saved snapshots
GET   /experiments/{id}       → single snapshot (poll this for status)
DELETE /experiments/{id}      → remove snapshot (204)
POST  /experiments/compare    → side-by-side comparison rows
"""
from __future__ import annotations

import logging

import torch
import torchvision
import torchvision.transforms as transforms
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from torch.utils.data import DataLoader, Subset

from neuroforge_core.application.experiment_utils import (
    build_snapshot,
    serialize_architecture,
    serialize_training_config,
)
from neuroforge_core.application.train_model import TrainModelRequest, TrainModelUseCase
from neuroforge_core.application.use_cases.experiment_tracking_use_case import (
    ExperimentTrackingUseCase,
)
from neuroforge_core.domain.entities.experiment_snapshot import ExperimentSnapshot
from neuroforge_core.domain.value_objects.evaluation_config import EvaluationConfig
from neuroforge_core.infrastructure.adapters.pytorch_evaluator import PyTorchEvaluator
from neuroforge_core.infrastructure.adapters.pytorch_model_compiler import (
    PyTorchModelCompiler,
)
from neuroforge_core.infrastructure.training.pytorch_trainer import PyTorchTrainer

from api.dependencies import get_tracking_use_case, _get_tracker
from api.layer_parser import parse_architecture, parse_training_config
from api.schemas import (
    CompareRequest,
    CompareResponse,
    RunExperimentRequest,
    SnapshotResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/experiments", tags=["experiments"])

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_CIFAR10_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def _get_cifar10_loaders(
    train_samples: int,
    test_samples: int,
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader]:
    train_ds = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=_CIFAR10_TRANSFORM
    )
    test_ds = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=_CIFAR10_TRANSFORM
    )
    train_loader = DataLoader(
        Subset(train_ds, list(range(min(train_samples, len(train_ds))))),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        Subset(test_ds, list(range(min(test_samples, len(test_ds))))),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, test_loader


# ------------------------------------------------------------------ #
# Background training task
# ------------------------------------------------------------------ #

def _train_and_update(
    experiment_id: str,
    created_at: str,
    body: RunExperimentRequest,
    arch,
    config,
) -> None:
    """
    Runs in a background thread after the 201 response is already sent.

    Uses _get_tracker() directly to get the singleton FileExperimentTracker —
    do not pass the request-scoped ExperimentTrackingUseCase into this
    function, FastAPI may tear it down once the response is sent.
    """
    tracker = _get_tracker()
    tracking = ExperimentTrackingUseCase(repository=tracker)

    # Mark as running
    try:
        snapshot = tracking.load(experiment_id)
        snapshot.status = "running"
        tracking.save(snapshot)
    except Exception as exc:
        logger.error("Failed to mark experiment %s as running: %s", experiment_id, exc)
        return

    try:
        logger.info("Starting training for experiment %s", experiment_id)

        train_loader, test_loader = _get_cifar10_loaders(
            train_samples=body.dataset_config.train_samples,
            test_samples=body.dataset_config.test_samples,
        )

        compiler = PyTorchModelCompiler()
        model = compiler.compile(arch)

        trainer = PyTorchTrainer(model=model, device=_DEVICE)
        train_use_case = TrainModelUseCase(trainer=trainer)
        request = TrainModelRequest(
            architecture=arch,
            config=config,
            train_loader=train_loader,
        )
        response = train_use_case.execute(request)

        if not response.succeeded:
            raise RuntimeError("Training failed: use case returned succeeded=False")

        evaluator = PyTorchEvaluator()
        eval_metrics = evaluator.evaluate(
            model=model,
            data_loader=test_loader,
            config=EvaluationConfig(top_k=5),
        )

        # build_snapshot accepts experiment_id directly — reuse the same id
        # so this becomes an update, not a new record.
        completed = build_snapshot(
            name=body.name,
            architecture=arch,
            training_config=config,
            final_train_loss=response.final_train_loss,
            eval_metrics=eval_metrics,
            tags=body.tags,
            experiment_id=experiment_id,
        )
        # created_at isn't a build_snapshot param — preserve the original
        # submission time rather than the completion time.
        completed.created_at = created_at
        completed.status = "completed"
        tracking.save(completed)

        logger.info(
            "Experiment %s completed. Top-1: %.3f",
            experiment_id,
            eval_metrics.accuracy,
        )

    except Exception as exc:
        logger.exception("Training failed for experiment %s: %s", experiment_id, exc)
        try:
            snapshot = tracking.load(experiment_id)
            snapshot.status = "failed"
            snapshot.tags["error"] = str(exc)
            tracking.save(snapshot)
        except Exception as save_exc:
            logger.error("Could not save failed status: %s", save_exc)


# ------------------------------------------------------------------ #
# POST /experiments/run
# ------------------------------------------------------------------ #

@router.post("/run", response_model=SnapshotResponse, status_code=201)
def run_experiment(
    body: RunExperimentRequest,
    background_tasks: BackgroundTasks,
    tracking: ExperimentTrackingUseCase = Depends(get_tracking_use_case),
) -> SnapshotResponse:
    """
    Validate architecture and config, save a pending snapshot, then
    kick off training in a background thread. Returns 201 immediately.

    Poll GET /experiments/{experiment_id} for status updates:
        "pending"   → queued, not yet started
        "running"   → training in progress
        "completed" → results available
        "failed"    → see tags["error"] for reason
    """
    try:
        arch = parse_architecture(body.architecture)
        config = parse_training_config(body.training_config)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Fail fast on structurally invalid architectures before queuing
    compiler = PyTorchModelCompiler()
    try:
        compiler.compile(arch)
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"Architecture compile error: {exc}"
        )

    # Use the same serializers build_snapshot uses, so the pending
    # snapshot has the identical shape to the eventual completed one.
    arch_summary = serialize_architecture(arch)
    config_dict = serialize_training_config(config)

    pending = ExperimentSnapshot(
        name=body.name,
        architecture_summary=arch_summary,
        training_config=config_dict,
        results={},
        status="pending",
        tags=body.tags or {},
    )
    tracking.save(pending)

    background_tasks.add_task(
        _train_and_update,
        experiment_id=pending.experiment_id,
        created_at=pending.created_at,
        body=body,
        arch=arch,
        config=config,
    )

    logger.info(
        "Experiment %s queued. Training will run in background.",
        pending.experiment_id,
    )
    return SnapshotResponse.from_snapshot(pending)


# ------------------------------------------------------------------ #
# GET /experiments
# ------------------------------------------------------------------ #

@router.get("", response_model=list[SnapshotResponse])
def list_experiments(
    tracking: ExperimentTrackingUseCase = Depends(get_tracking_use_case),
) -> list[SnapshotResponse]:
    snapshots = tracking.list_all()
    return [SnapshotResponse.from_snapshot(s) for s in snapshots]


# ------------------------------------------------------------------ #
# GET /experiments/{experiment_id}
# ------------------------------------------------------------------ #

@router.get("/{experiment_id}", response_model=SnapshotResponse)
def get_experiment(
    experiment_id: str,
    tracking: ExperimentTrackingUseCase = Depends(get_tracking_use_case),
) -> SnapshotResponse:
    try:
        snapshot = tracking.load(experiment_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment not found: {experiment_id!r}",
        )
    return SnapshotResponse.from_snapshot(snapshot)


# ------------------------------------------------------------------ #
# DELETE /experiments/{experiment_id}
# ------------------------------------------------------------------ #

@router.delete("/{experiment_id}", status_code=204)
def delete_experiment(
    experiment_id: str,
    tracking: ExperimentTrackingUseCase = Depends(get_tracking_use_case),
) -> None:
    tracking.delete(experiment_id)


# ------------------------------------------------------------------ #
# POST /experiments/compare
# ------------------------------------------------------------------ #

@router.post("/compare", response_model=CompareResponse)
def compare_experiments(
    body: CompareRequest,
    tracking: ExperimentTrackingUseCase = Depends(get_tracking_use_case),
) -> CompareResponse:
    missing = []
    for eid in body.ids:
        try:
            tracking.load(eid)
        except KeyError:
            missing.append(eid)

    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment(s) not found: {missing}",
        )

    rows = tracking.compare(body.ids)
    return CompareResponse(rows=rows)