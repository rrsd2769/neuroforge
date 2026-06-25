"""
Experiment routes.

POST  /experiments/run        → compile + train + evaluate + save → snapshot
GET   /experiments            → list all saved snapshots
GET   /experiments/{id}       → single snapshot
DELETE /experiments/{id}      → remove snapshot (204)
POST  /experiments/compare    → side-by-side comparison rows
"""
from __future__ import annotations

import torch
import torchvision
import torchvision.transforms as transforms
from fastapi import APIRouter, Depends, HTTPException
from torch.utils.data import DataLoader, Subset

from neuroforge_core.application.experiment_utils import build_snapshot
from neuroforge_core.application.train_model import TrainModelRequest, TrainModelUseCase
from neuroforge_core.application.use_cases.experiment_tracking_use_case import (
    ExperimentTrackingUseCase,
)
from neuroforge_core.domain.value_objects.evaluation_config import EvaluationConfig
from neuroforge_core.infrastructure.adapters.file_experiment_tracker import (
    FileExperimentTracker,
)
from neuroforge_core.infrastructure.adapters.pytorch_evaluator import PyTorchEvaluator
from neuroforge_core.infrastructure.adapters.pytorch_model_compiler import (
    PyTorchModelCompiler,
)
from neuroforge_core.infrastructure.training.pytorch_trainer import PyTorchTrainer

from api.dependencies import get_tracking_use_case
from api.layer_parser import parse_architecture, parse_training_config
from api.schemas import (
    CompareRequest,
    CompareResponse,
    RunExperimentRequest,
    SnapshotResponse,
)

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
# POST /experiments/run
# ------------------------------------------------------------------ #

@router.post("/run", response_model=SnapshotResponse, status_code=201)
def run_experiment(
    body: RunExperimentRequest,
    tracking: ExperimentTrackingUseCase = Depends(get_tracking_use_case),
) -> SnapshotResponse:
    """
    Compile, train, and evaluate an architecture on CIFAR-10.
    Saves the result as an ExperimentSnapshot and returns it.
    """
    # Parse domain objects
    try:
        arch = parse_architecture(body.architecture)
        config = parse_training_config(body.training_config)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Load data
    train_loader, test_loader = _get_cifar10_loaders(
        train_samples=body.dataset_config.train_samples,
        test_samples=body.dataset_config.test_samples,
    )

    # Compile
    compiler = PyTorchModelCompiler()
    try:
        model = compiler.compile(arch)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Architecture compile error: {e}")

    # Train
    trainer = PyTorchTrainer(model=model, device=_DEVICE)
    train_use_case = TrainModelUseCase(trainer=trainer)
    request = TrainModelRequest(
        architecture=arch,
        config=config,
        train_loader=train_loader,
    )
    response = train_use_case.execute(request)

    if not response.succeeded:
        raise HTTPException(status_code=500, detail="Training failed")

    # Evaluate
    evaluator = PyTorchEvaluator()
    eval_metrics = evaluator.evaluate(
        model=model,
        data_loader=test_loader,
        config=EvaluationConfig(top_k=5),
    )

    # Build + save snapshot
    snapshot = build_snapshot(
        name=body.name,
        architecture=arch,
        training_config=config,
        final_train_loss=response.final_train_loss,
        eval_metrics=eval_metrics,
        tags=body.tags,
    )
    tracking.save(snapshot)

    return SnapshotResponse.from_snapshot(snapshot)


# ------------------------------------------------------------------ #
# GET /experiments
# ------------------------------------------------------------------ #

@router.get("", response_model=list[SnapshotResponse])
def list_experiments(
    tracking: ExperimentTrackingUseCase = Depends(get_tracking_use_case),
) -> list[SnapshotResponse]:
    """List all saved experiments, sorted by creation time ascending."""
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
    """Load a single experiment by ID."""
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
    """Delete a saved experiment. Returns 204 whether or not it existed."""
    tracking.delete(experiment_id)


# ------------------------------------------------------------------ #
# POST /experiments/compare
# ------------------------------------------------------------------ #

@router.post("/compare", response_model=CompareResponse)
def compare_experiments(
    body: CompareRequest,
    tracking: ExperimentTrackingUseCase = Depends(get_tracking_use_case),
) -> CompareResponse:
    """
    Return a side-by-side comparison table for a list of experiment IDs.
    Missing IDs return a 404.
    """
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