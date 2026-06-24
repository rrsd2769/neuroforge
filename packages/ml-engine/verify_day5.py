#!/usr/bin/env python3
"""Day 5 exit-gate: smoke-tests every new component then runs pytest."""
import subprocess
import sys
from unittest.mock import MagicMock


def section(n: int, title: str) -> None:
    print(f"\n[{n}/5] {title} ...")


def main() -> None:
    print("=" * 55)
    print(" NeuroForge Day 5 — Exit-Gate Verification")
    print("=" * 55)

    # 1 — Imports
    section(1, "Importing all new modules")
    from neuroforge_core.domain.value_objects.training_config import (
        TrainingConfig, OptimizerType,
    )
    from neuroforge_core.domain.value_objects.training_metrics import (
        EpochMetrics, TrainingHistory,
    )
    from neuroforge_core.domain.entities.training_run import (
        TrainingRun, TrainingStatus,
    )
    from neuroforge_core.domain.interfaces.model_trainer import IModelTrainer
    from neuroforge_core.infrastructure.training.pytorch_trainer import PyTorchTrainer
    from neuroforge_core.application.train_model import (
        TrainModelUseCase, TrainModelRequest,
    )
    print("    ✓ all imports resolved")

    # 2 — Value objects
    section(2, "Validating value objects")
    cfg = TrainingConfig(learning_rate=1e-3, epochs=2, optimizer=OptimizerType.ADAM)
    assert cfg.learning_rate == 1e-3
    h = TrainingHistory()
    h.record(EpochMetrics(epoch=1, train_loss=0.5, train_accuracy=0.8))
    assert len(h) == 1 and h.final_train_loss == 0.5
    print("    ✓ TrainingConfig and TrainingHistory work correctly")

    # 3 — TrainingRun state machine
    section(3, "Exercising TrainingRun state machine")
    from neuroforge_core.domain.entities.architecture import Architecture
    arch = MagicMock(spec=Architecture)
    run = TrainingRun(architecture=arch, config=cfg)
    assert run.status == TrainingStatus.PENDING
    run.start()
    assert run.status == TrainingStatus.RUNNING
    run.complete()
    assert run.status == TrainingStatus.COMPLETED and run.is_done
    assert run.duration_seconds is not None
    print("    ✓ PENDING → RUNNING → COMPLETED, guards intact")

    # 4 — PyTorchTrainer end-to-end (CPU, 2 epochs)
    section(4, "Running PyTorchTrainer (CPU, 2 epochs, synthetic data)")
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    inputs = torch.randn(16, 4)
    targets = torch.randint(0, 3, (16,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=8)

    arch2 = MagicMock(spec=Architecture)
    run2 = TrainingRun(architecture=arch2, config=cfg)
    trainer = PyTorchTrainer(model)
    trainer.train(run2, loader, val_loader=loader)

    assert run2.status == TrainingStatus.COMPLETED
    assert len(run2.history) == 2
    assert run2.history.best_val_accuracy is not None
    print("    ✓ 2 epochs, val metrics populated, status COMPLETED")

    # 5 — pytest
    section(5, "Running full test suite")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
    )
    if result.returncode != 0:
        print("\n✗  Test suite FAILED — review output above.")
        sys.exit(1)

    print("\n" + "=" * 55)
    print(" ✓  All Day 5 checks passed. Ready to tag v0.5.0.")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()