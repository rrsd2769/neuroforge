# NeuroForge — Architecture Deep Dive

## Overview

NeuroForge applies **hexagonal architecture** (also called ports and adapters) to a machine learning system. The core principle: the domain layer — entities, ports, use cases — is pure Python with zero framework dependencies. PyTorch, FastAPI, Streamlit, and the file system are all infrastructure that plugs into the domain through typed abstract contracts called **ports**.

---

## Layer Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                         DRIVERS                                  │
│                                                                  │
│   ┌─────────────────────┐     ┌──────────────────────────────┐  │
│   │   FastAPI (api/)    │     │   Streamlit (dashboard/)     │  │
│   │                     │     │                              │  │
│   │  POST /experiments  │     │  Run Experiment page         │  │
│   │  GET  /experiments  │     │  Experiments browser         │  │
│   │  GET  /health       │     │  Results deep-dive           │  │
│   └──────────┬──────────┘     └──────────────┬───────────────┘  │
│              │ calls use cases               │ HTTP only        │
└──────────────┼───────────────────────────────┼──────────────────┘
               │                               │
┌──────────────▼───────────────────────────────┼──────────────────┐
│                      APPLICATION              │                  │
│                                              (api_client.py)    │
│   ArchitectureSearchUseCase                                      │
│   EvaluateModelUseCase                                           │
│   ExperimentTrackingUseCase                                      │
│   DatasetManager                                                 │
│   train_model()                                                  │
│                                                                  │
│   [knows ports, never knows adapters]                            │
└──────────────┬───────────────────────────────────────────────────┘
               │ depends on (inward only)
┌──────────────▼───────────────────────────────────────────────────┐
│                        DOMAIN                                     │
│                                                                   │
│   ENTITIES                    PORTS (interfaces/)                 │
│   ───────────────             ─────────────────────────────────  │
│   Architecture                ArchitectureGeneratorPort          │
│   TrainingRun                 DatasetSourcePort                   │
│   Experiment                  IModelCompiler                      │
│   ExperimentSnapshot          IModelTrainer / TrainerPort         │
│   EvaluationResult            IModelEvaluator / EvaluatorPort     │
│   TrainingMetrics             IExperimentRepository               │
│   Dataset                     ExperimentTrackerPort               │
│   ModelArtifact                                                   │
│                                                                   │
│   VALUE OBJECTS                                                   │
│   ───────────────                                                 │
│   TrainingConfig              SearchSpace                         │
│   EvaluationConfig            Layer (typed layer definitions)     │
│   EvaluationMetrics           TrainingMetrics                     │
│                                                                   │
│   [zero imports of torch, fastapi, streamlit, or any I/O]        │
└──────────────▲───────────────────────────────────────────────────┘
               │ implements ports (outward)
┌──────────────┴───────────────────────────────────────────────────┐
│                     INFRASTRUCTURE                                │
│                                                                   │
│   ADAPTERS (infrastructure/adapters/)                            │
│   ─────────────────────────────────                              │
│   PyTorchModelCompiler    → IModelCompiler                       │
│   PyTorchTrainer          → IModelTrainer + TrainerPort           │
│   PyTorchEvaluator        → EvaluatorPort + IModelEvaluator       │
│   FileExperimentTracker   → IExperimentRepository                │
│                             + ExperimentTrackerPort              │
│                                                                   │
│   DATASETS (infrastructure/datasets/)                            │
│   ─────────────────────────────────                              │
│   CIFAR10Loader           → DatasetSourcePort                    │
│   DataloaderFactory, DatasetValidator, Preprocessing             │
│                                                                   │
│   GENERATORS (infrastructure/generators/)                        │
│   ──────────────────────────────────────                         │
│   RandomArchitectureGenerator → ArchitectureGeneratorPort        │
│                                                                   │
│   TRAINING (infrastructure/training/)                            │
│   ───────────────────────────────────                            │
│   PyTorchTrainer (full training loop implementation)             │
└──────────────────────────────────────────────────────────────────┘
```

---

## Port Definitions

All ports live in `neuroforge_core/domain/interfaces/`. Each is an abstract base class (ABC) — the domain layer knows only the contract, never the implementation.

### `ArchitectureGeneratorPort`
```python
class ArchitectureGeneratorPort(ABC):
    @abstractmethod
    def generate(self, search_space: SearchSpace) -> Architecture:
        """Proposes one Architecture satisfying the given SearchSpace constraints."""
```
**Adapter:** `RandomArchitectureGenerator` — samples layer counts, channel widths, and activation functions randomly within the SearchSpace bounds.  
**Swap candidate:** Bayesian optimizer, DARTS, evolutionary search.

---

### `DatasetSourcePort`
```python
class DatasetSourcePort(ABC):
    @abstractmethod
    def load(self) -> Dataset:
        """Loads (downloading/caching if needed) and returns a Dataset entity."""
```
**Adapter:** `CIFAR10Loader` — downloads CIFAR-10 via TorchVision, wraps in domain `Dataset` entity.  
**Swap candidate:** FashionMNIST, Tiny ImageNet, any custom dataset.

---

### `IModelCompiler`
```python
class IModelCompiler(ABC):
    @abstractmethod
    def compile(self, architecture: Architecture,
                input_channels: int = 3, num_classes: int = 10) -> Any:
        """Compile an Architecture into a trainable model object."""

    @abstractmethod
    def is_valid_architecture(self, architecture: Architecture, ...) -> bool:
        """Return True if the architecture compiles and produces valid output shape."""
```
**Adapter:** `PyTorchModelCompiler` — translates domain layer definitions into `torch.nn.Sequential` modules. Returns `Any` (not `torch.nn.Module`) so torch never appears in the domain layer.  
**Swap candidate:** JAX model builder, ONNX exporter.

---

### `IModelTrainer` / `TrainerPort`
```python
class IModelTrainer(ABC):
    @abstractmethod
    def train(self, run: TrainingRun,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None) -> TrainingRun:
        """Execute the training loop; return the mutated run."""

class TrainerPort(ABC):
    @abstractmethod
    def train(self, model: Any, train_loader: Any,
              val_loader: Any, config: TrainingConfig) -> list[TrainingMetrics]:
        """Runs the full training loop and returns per-epoch metrics."""
```
**Adapter:** `PyTorchTrainer` — implements both contracts. Manages optimizer selection (Adam/SGD), loss computation, epoch loops, and metric aggregation. The `TrainingRun` state machine enforces `start() → history.append() → complete()/fail()`.  
**Swap candidate:** JAX trainer, HuggingFace Trainer wrapper, sklearn classifier.

---

### `EvaluatorPort` / `IModelEvaluator`
```python
class EvaluatorPort(ABC):
    @abstractmethod
    def evaluate(self, model: Any, test_loader: Any) -> EvaluationResult:
        """Runs inference over the test set and returns aggregated metrics."""

class IModelEvaluator(ABC):
    @abstractmethod
    def evaluate(self, model: Any, data_loader: Any,
                 config: EvaluationConfig,
                 class_names: Optional[List[str]] = None) -> EvaluationMetrics:
        """Run full evaluation pass and return structured metrics."""
```
**Adapter:** `PyTorchEvaluator` — implements both. Runs `torch.no_grad()` inference, computes top-1/top-5 accuracy, and aggregates loss. Stateless — `model` is passed to `evaluate()`, not `__init__`.  
**Swap candidate:** TorchMetrics evaluator, custom per-class precision/recall evaluator.

---

### `IExperimentRepository` / `ExperimentTrackerPort`
```python
class IExperimentRepository(ABC):
    @abstractmethod
    def save(self, snapshot: ExperimentSnapshot) -> None: ...
    @abstractmethod
    def load(self, experiment_id: str) -> ExperimentSnapshot: ...
    @abstractmethod
    def list_all(self) -> list[ExperimentSnapshot]: ...
    @abstractmethod
    def delete(self, experiment_id: str) -> None: ...

class ExperimentTrackerPort(ABC):
    @abstractmethod
    def log_experiment(self, experiment: Experiment, **artifacts: Any) -> None: ...
    @abstractmethod
    def get_experiment(self, experiment_id: str) -> Experiment: ...
```
**Adapter:** `FileExperimentTracker` — implements both. Persists `ExperimentSnapshot` as JSON files to disk. `list_all()` returns sorted by `created_at` ascending.  
**Swap candidate:** SQLite adapter, PostgreSQL adapter, MLflow tracker, Weights & Biases.

---

## Data Flow: `POST /experiments/run`

```
HTTP POST /experiments/run  (JSON payload)
         │
         ▼
api/routers/experiments.py
  └─ validates with RunExperimentRequest (Pydantic v2 schema)
  └─ calls api/dependencies.py to resolve concrete adapters
         │
         ▼
application/train_model.py  (or use_cases/architecture_search.py)
  └─ calls IModelCompiler.compile(architecture)     → torch.nn.Module
  └─ calls DatasetSourcePort.load()                 → Dataset
  └─ calls IModelTrainer.train(run, loaders)        → TrainingRun
  └─ calls EvaluatorPort.evaluate(model, loader)   → EvaluationResult
  └─ calls IExperimentRepository.save(snapshot)    → JSON file on disk
         │
         ▼
api/schemas.py
  └─ ExperimentResponse (Pydantic serialization)
         │
         ▼
HTTP 201 response  (experiment_id, results, architecture_summary)
```

---

## Dependency Injection (api/dependencies.py)

The API layer is where concrete adapters are wired to ports. Use cases never import adapters directly — they receive them through constructor injection:

```python
# api/dependencies.py (simplified)
def get_model_compiler() -> IModelCompiler:
    return PyTorchModelCompiler()

def get_trainer() -> IModelTrainer:
    return PyTorchTrainer()

def get_repository() -> IExperimentRepository:
    return FileExperimentTracker(storage_dir="./experiments")
```

FastAPI's `Depends()` system wires these into route handlers. In tests, the `TestClient` overrides these with mock adapters — the domain and application layers are never touched by test fixtures.

---

## Test Architecture

| Test location | What it tests | Framework deps needed |
|---|---|---|
| `tests/domain/` | Entities, value objects | None |
| `tests/unit/domain/` | Fine-grained entity behaviour | None |
| `tests/application/` | Use cases with mocked ports | None |
| `tests/unit/application/` | DatasetManager, experiment utils | None |
| `tests/infrastructure/` | PyTorchTrainer, RandomArchitectureGenerator | PyTorch |
| `tests/unit/infrastructure/` | CIFAR10Loader, DataloaderFactory, preprocessing | PyTorch, TorchVision |
| `tests/test_day6/` | ModelCompiler, PyTorchEvaluator, EvaluationMetrics | PyTorch |
| `tests/test_day7/` | ExperimentTracking, FileExperimentTracker | File I/O |
| `tests/test_day8_api.py` | FastAPI endpoints via TestClient | FastAPI TestClient |
| `tests/dashboard/` | API client, charts, UI helpers | requests (mocked) |

The domain and application tests have **no framework dependencies** — they run in milliseconds with no GPU, no network, and no disk access. This is the testability payoff of the hexagonal design.

---

## What Could Be Extended

Because every major concern is behind a port, these extensions are **one adapter each** — no domain changes required:

| Extension | Port to implement | Estimated effort |
|---|---|---|
| SQLite experiment storage | `IExperimentRepository` | ~100 lines |
| PostgreSQL experiment storage | `IExperimentRepository` | ~150 lines |
| MLflow tracking | `ExperimentTrackerPort` | ~80 lines |
| JAX training backend | `IModelTrainer` | ~200 lines |
| FashionMNIST dataset | `DatasetSourcePort` | ~30 lines |
| Bayesian architecture search | `ArchitectureGeneratorPort` | ~300 lines |
| ONNX model export | `IModelCompiler` | ~100 lines |