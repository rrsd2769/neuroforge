"""
Pydantic request and response schemas for the NeuroForge REST API.

Deliberately separate from domain value objects — the API contract can
evolve independently of the domain model.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Union, Optional

from pydantic import BaseModel, Field


# ------------------------------------------------------------------ #
# Layer input schemas (request body → domain Layer objects)
# ------------------------------------------------------------------ #

class ConvLayerIn(BaseModel):
    type: Literal["conv"]
    out_channels: int = 32
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    activation: str = "relu"


class PoolLayerIn(BaseModel):
    type: Literal["pool"]
    pool_size: int = 2
    stride: int = 2
    pool_type: str = "max"


class FlattenLayerIn(BaseModel):
    type: Literal["flatten"]


class DenseLayerIn(BaseModel):
    type: Literal["dense"]
    units: int = 128
    activation: str = "relu"


class DropoutLayerIn(BaseModel):
    type: Literal["dropout"]
    rate: float = 0.5


# Pydantic v2 discriminated union — validated automatically from {"type": "..."}
LayerIn = Annotated[
    Union[ConvLayerIn, PoolLayerIn, FlattenLayerIn, DenseLayerIn, DropoutLayerIn],
    Field(discriminator="type"),
]


# ------------------------------------------------------------------ #
# Architecture input schema
# ------------------------------------------------------------------ #

class ArchitectureIn(BaseModel):
    num_classes: int = 10
    layers: list[LayerIn]


# ------------------------------------------------------------------ #
# Training config input schema
# ------------------------------------------------------------------ #

class TrainingConfigIn(BaseModel):
    epochs: int = 5
    learning_rate: float = 0.001
    optimizer: str = "adam"        # "adam" | "sgd" | "adamw"
    weight_decay: float = 0.0
    momentum: float = 0.9


# ------------------------------------------------------------------ #
# Dataset config input schema
# ------------------------------------------------------------------ #

class DatasetConfigIn(BaseModel):
    train_samples: int = Field(default=500, ge=1, le=50000)
    test_samples: int = Field(default=100, ge=1, le=10000)


# ------------------------------------------------------------------ #
# Top-level run request
# ------------------------------------------------------------------ #

class RunExperimentRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    architecture: ArchitectureIn
    training_config: TrainingConfigIn = TrainingConfigIn()
    dataset_config: DatasetConfigIn = DatasetConfigIn()
    tags: dict[str, str] = {}


# ------------------------------------------------------------------ #
# Compare request
# ------------------------------------------------------------------ #

class CompareRequest(BaseModel):
    ids: list[str] = Field(..., min_length=2)


# ------------------------------------------------------------------ #
# Response schemas
# ------------------------------------------------------------------ #

class SnapshotResponse(BaseModel):
    """JSON-safe representation of a saved ExperimentSnapshot."""
    experiment_id: str
    name: str
    created_at: str
    status: str
    architecture_summary: dict[str, Any]
    training_config: dict[str, Any]
    results: dict[str, Any]
    tags: dict[str, str]

    @classmethod
    def from_snapshot(cls, snapshot) -> "SnapshotResponse":
        return cls(**snapshot.to_dict())


class CompareResponse(BaseModel):
    rows: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    version: str

class ArchitectureSearchRequest(BaseModel):
    n_trials: int = 15
    epochs_per_trial: int = 2
    train_samples_per_trial: int = 1000
    test_samples_per_trial: int = 300
    min_depth: int = 2
    max_depth: int = 6
    channel_choices: list[int] = [16, 32, 64, 128]


class TrialResultSchema(BaseModel):
    trial_number: int
    top1_accuracy: float
    top5_accuracy: float
    parameter_count: int
    learning_rate: float
    optimizer: str
    num_layers: int
    architecture_id: str


class SearchStatusResponse(BaseModel):
    search_id: str
    status: str
    created_at: str
    n_trials: int
    completed_trials: int
    trials: list[dict]
    best_trial: Optional[dict] = None
    error: Optional[str] = None