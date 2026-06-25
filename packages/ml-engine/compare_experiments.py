"""
compare_experiments.py
======================
End-to-end demo: trains two architectures on a tiny CIFAR-10 subset,
saves both as ExperimentSnapshots, loads them back from disk, and prints
a side-by-side comparison table.

Run from packages/ml-engine/:
    python compare_experiments.py

Expects CIFAR-10 to already be downloaded (cached from earlier days).
If not, set download=True in the torchvision calls — first run will fetch ~170 MB.
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# ------------------------------------------------------------------ #
# Path setup — makes neuroforge_core importable
# ------------------------------------------------------------------ #
sys.path.insert(0, str(Path(__file__).parent))

from neuroforge_core.application.experiment_utils import build_snapshot
from neuroforge_core.application.use_cases.experiment_tracking_use_case import (
    ExperimentTrackingUseCase,
)
from neuroforge_core.domain.entities.architecture import Architecture
from neuroforge_core.domain.value_objects.layer import (
    ConvLayer,
    DenseLayer,
    DropoutLayer,
    FlattenLayer,
    PoolLayer,
)
from neuroforge_core.domain.value_objects.training_config import (
    OptimizerType,
    TrainingConfig,
)
from neuroforge_core.infrastructure.adapters.file_experiment_tracker import (
    FileExperimentTracker,
)
from neuroforge_core.infrastructure.adapters.pytorch_evaluator import PyTorchEvaluator
from neuroforge_core.infrastructure.adapters.pytorch_model_compiler import (
    PyTorchModelCompiler,
)
from neuroforge_core.infrastructure.training.pytorch_trainer import PyTorchTrainer

from neuroforge_core.application.train_model import TrainModelUseCase, TrainModelRequest

# ------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------ #

STORAGE_ROOT = Path("experiments_demo")   # isolated from real runs
TRAIN_SAMPLES = 300
TEST_SAMPLES = 100
EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DEVICE = torch.device("cpu")


# ------------------------------------------------------------------ #
# Data
# ------------------------------------------------------------------ #

def _get_loaders() -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_full = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_full = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    train_loader = DataLoader(
        Subset(train_full, list(range(TRAIN_SAMPLES))),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        Subset(test_full, list(range(TEST_SAMPLES))),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    return train_loader, test_loader


# ------------------------------------------------------------------ #
# Architecture definitions
# ------------------------------------------------------------------ #

def _small_cnn() -> Architecture:
    """3-layer: Conv → Pool → Flatten → Dense → Dense"""
    return Architecture(
        layers=[
            ConvLayer(out_channels=32, kernel_size=3, activation="relu"),
            PoolLayer(pool_size=2, pool_type="max"),
            FlattenLayer(),
            DenseLayer(units=128, activation="relu"),
            DenseLayer(units=10, activation="none"),
        ],
        num_classes=10,
    )

def _deeper_cnn() -> Architecture:
    """5-layer: Conv → Conv → Pool → Flatten → Dense → Dropout → Dense"""
    return Architecture(
        layers=[
            ConvLayer(out_channels=32, kernel_size=3, activation="relu"),
            ConvLayer(out_channels=64, kernel_size=3, activation="relu"),
            PoolLayer(pool_size=2, pool_type="max"),
            FlattenLayer(),
            DenseLayer(units=256, activation="relu"),
            DropoutLayer(rate=0.3),
            DenseLayer(units=10, activation="none"),
        ],
        num_classes=10,
    )


# ------------------------------------------------------------------ #
# Training helper
# ------------------------------------------------------------------ #


def _train_and_evaluate(
    name: str,
    arch: Architecture,
    train_loader: DataLoader,
    test_loader: DataLoader,
) -> dict:
    """Compile → Train → Evaluate. Returns a dict of results."""
    print(f"\n{'='*50}")
    print(f"  Training: {name}")
    print(f"{'='*50}")

    config = TrainingConfig(
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        optimizer=OptimizerType.ADAM,
    )

    # Compile
    compiler = PyTorchModelCompiler()
    model = compiler.compile(arch)

    # Train via use case (handles TrainingRun construction internally)
    t0 = time.time()
    trainer = PyTorchTrainer(model=model, device=DEVICE)
    use_case = TrainModelUseCase(trainer=trainer)
    request = TrainModelRequest(
        architecture=arch,
        config=config,
        train_loader=train_loader,
    )
    response = use_case.execute(request)
    elapsed = time.time() - t0
    print(f"  Training complete in {elapsed:.1f}s | "
          f"Final loss: {response.final_train_loss:.4f}")

    # Evaluate
# Evaluate
    from neuroforge_core.domain.value_objects.evaluation_config import EvaluationConfig
    evaluator = PyTorchEvaluator()
    eval_config = EvaluationConfig(top_k=5)
    eval_metrics = evaluator.evaluate(
        model=model,
        data_loader=test_loader,
        config=eval_config,
    )
    print(f"  Top-1: {eval_metrics.accuracy * 100:.2f}%  "
          f"Top-5: {eval_metrics.top_k_accuracy * 100:.2f}%  "
          f"Eval loss: {eval_metrics.average_loss:.4f}")

    return {
        "config": config,
        "final_train_loss": response.final_train_loss,
        "eval_metrics": eval_metrics,
    }


# ------------------------------------------------------------------ #
# Table printing
# ------------------------------------------------------------------ #

def _print_comparison(rows: list[dict]) -> None:
    if not rows:
        print("No data to compare.")
        return

    headers = list(rows[0].keys())
    # Column widths
    col_widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            col_widths[h] = max(col_widths[h], len(str(row.get(h, ""))))

    sep = "+-" + "-+-".join("-" * col_widths[h] for h in headers) + "-+"
    header_line = "| " + " | ".join(h.ljust(col_widths[h]) for h in headers) + " |"

    print()
    print("  EXPERIMENT COMPARISON")
    print(sep)
    print(header_line)
    print(sep)
    for row in rows:
        line = "| " + " | ".join(str(row.get(h, "")).ljust(col_widths[h]) for h in headers) + " |"
        print(line)
    print(sep)
    print()


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main() -> None:
    # Clean up any previous demo run for reproducibility
    if STORAGE_ROOT.exists():
        shutil.rmtree(STORAGE_ROOT)

    print("\nNeuroForge — Experiment Comparison Demo")
    print(f"Architectures: SmallCNN vs DeeperCNN")
    print(f"Training samples: {TRAIN_SAMPLES} | Test samples: {TEST_SAMPLES} | Epochs: {EPOCHS}")

    train_loader, test_loader = _get_loaders()

    # --- Run 1: SmallCNN ---
    small_arch = _small_cnn()
    small_result = _train_and_evaluate("SmallCNN", small_arch, train_loader, test_loader)

    # --- Run 2: DeeperCNN ---
    deep_arch = _deeper_cnn()
    deep_result = _train_and_evaluate("DeeperCNN", deep_arch, train_loader, test_loader)

    # --- Build snapshots ---
    tracker = FileExperimentTracker(storage_root=STORAGE_ROOT)
    use_case = ExperimentTrackingUseCase(repository=tracker)

    small_snap = build_snapshot(
        name="SmallCNN",
        architecture=small_arch,
        training_config=small_result["config"],
        final_train_loss=small_result["final_train_loss"],
        eval_metrics=small_result["eval_metrics"],
        tags={"variant": "baseline"},
    )
    deep_snap = build_snapshot(
        name="DeeperCNN",
        architecture=deep_arch,
        training_config=deep_result["config"],
        final_train_loss=deep_result["final_train_loss"],
        eval_metrics=deep_result["eval_metrics"],
        tags={"variant": "deeper"},
    )

    # --- Save ---
    use_case.save(small_snap)
    use_case.save(deep_snap)
    print(f"\nSaved 2 experiments to {STORAGE_ROOT}/runs/")

    # --- Load back (proving persistence works) ---
    loaded_small = use_case.load(small_snap.experiment_id)
    loaded_deep = use_case.load(deep_snap.experiment_id)
    print(f"Loaded back: '{loaded_small.name}' and '{loaded_deep.name}' ✓")

    # --- List all ---
    all_snapshots = use_case.list_all()
    print(f"Repository contains {len(all_snapshots)} experiment(s) ✓")

    # --- Compare ---
    rows = use_case.compare([small_snap.experiment_id, deep_snap.experiment_id])
    _print_comparison(rows)


if __name__ == "__main__":
    main()