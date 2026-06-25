"""
verify_day6.py — End-to-end verification of ModelCompiler + Evaluation pipeline.
Run from packages/ml-engine/:  python verify_day6.py
"""

import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

# ── Domain ────────────────────────────────────────────────────────────────────
from neuroforge_core.domain.entities.architecture import Architecture
from neuroforge_core.domain.value_objects.layer import (
    ConvLayer,
    DenseLayer,
    DropoutLayer,
    FlattenLayer,
    PoolLayer,
)
from neuroforge_core.domain.value_objects.evaluation_config import EvaluationConfig
from neuroforge_core.domain.value_objects.training_config import (
    TrainingConfig,
    OptimizerType,
)

# ── Infrastructure ────────────────────────────────────────────────────────────
from neuroforge_core.infrastructure.adapters.pytorch_model_compiler import (
    PyTorchModelCompiler,
    ModelCompilationError,
)
from neuroforge_core.infrastructure.adapters.pytorch_evaluator import PyTorchEvaluator
from neuroforge_core.infrastructure.training.pytorch_trainer import PyTorchTrainer

# ── Application ───────────────────────────────────────────────────────────────
from neuroforge_core.application.use_cases.evaluate_model_use_case import (
    EvaluateModelUseCase,
)
from neuroforge_core.application.train_model import (
    TrainModelUseCase,
    TrainModelRequest,
)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def check(label: str, passed: bool) -> None:
    status = "✓" if passed else "✗"
    print(f"  [{status}] {label}")
    if not passed:
        sys.exit(f"\nFailed at: {label}")


def main() -> None:
    print("\n" + "═" * 60)
    print("  NeuroForge — Day 6 Verification")
    print("  ModelCompiler + Evaluation Pipeline")
    print("═" * 60)

    # ── 1. Compile an Architecture ────────────────────────────────────────────
    section("1 · Architecture → ModelCompiler")

    architecture = Architecture(
        layers=[
            ConvLayer(out_channels=32, kernel_size=3, stride=1, padding=1, activation="relu"),
            PoolLayer(pool_size=2, stride=2),
            ConvLayer(out_channels=64, kernel_size=3, stride=1, padding=1, activation="relu"),
            PoolLayer(pool_size=2, stride=2),
            FlattenLayer(),
            DenseLayer(units=256, activation="relu"),
            DropoutLayer(rate=0.3),
        ],
        num_classes=10,
    )

    compiler = PyTorchModelCompiler()

    check(
        "is_valid_architecture returns True",
        compiler.is_valid_architecture(architecture),
    )

    model: nn.Module = compiler.compile(architecture)
    check("compile() returns nn.Module", isinstance(model, nn.Module))

    with torch.no_grad():
        out = model(torch.zeros(4, 3, 32, 32))
    check("Output shape is (4, 10)", out.shape == (4, 10))
    print(f"\n  Layers: {len(architecture.layers)} → compiled to nn.Sequential")
    print(f"  Output shape: {tuple(out.shape)}")

    # ── 2. Invalid architecture detection ─────────────────────────────────────
    section("2 · Invalid Architecture Detection")

    bad_arch = Architecture(
        layers=[
            PoolLayer(pool_size=2, stride=2),  # 32 → 16
            PoolLayer(pool_size=2, stride=2),  # 16 → 8
            PoolLayer(pool_size=2, stride=2),  # 8  → 4
            PoolLayer(pool_size=2, stride=2),  # 4  → 2
            PoolLayer(pool_size=2, stride=2),  # 2  → 1
            PoolLayer(pool_size=2, stride=2),  # 1  → 0  ← invalid
            FlattenLayer(),
        ],
        num_classes=10,
    )
    check(
        "is_valid_architecture returns False for broken arch",
        compiler.is_valid_architecture(bad_arch) is False,
    )

    # ── 3. Load CIFAR-10 subset ───────────────────────────────────────────────
    section("3 · Data Loading (CIFAR-10 subset — 500 train / 200 test)")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ])

    print("  Downloading CIFAR-10 if not cached ...")
    train_full = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_full = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        Subset(train_full, range(500)), batch_size=50, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        Subset(test_full, range(200)), batch_size=50, shuffle=False, num_workers=0
    )

    check("Train loader has 10 batches", len(train_loader) == 10)
    check("Test loader has 4 batches",   len(test_loader)  == 4)

    # ── 4. Train for 2 epochs ─────────────────────────────────────────────────
    section("4 · Training (2 epochs on 500 samples)")

    training_config = TrainingConfig(
        epochs=2,
        learning_rate=1e-3,
        optimizer=OptimizerType.ADAM,
    )

    # PyTorchTrainer takes the compiled model in its constructor.
    # Weights are updated in-place — `model` is trained after execute().
    trainer        = PyTorchTrainer(model=model, device=torch.device("cpu"))
    train_use_case = TrainModelUseCase(trainer=trainer)

    request = TrainModelRequest(
        architecture=architecture,
        config=training_config,
        train_loader=train_loader,
        val_loader=None,
    )

    t0 = time.time()
    response = train_use_case.execute(request)
    elapsed = time.time() - t0

    check("Training succeeded", response.succeeded)
    print(f"\n  Time:             {elapsed:.1f}s")
    if response.final_train_loss is not None:
        print(f"  Final train loss: {response.final_train_loss:.4f}")

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    section("5 · Evaluation")

    eval_config       = EvaluationConfig(device="cpu", top_k=5, batch_size=50)
    evaluate_use_case = EvaluateModelUseCase(evaluator=PyTorchEvaluator())

    metrics = evaluate_use_case.execute(
        model=model,
        data_loader=test_loader,
        config=eval_config,
    )

    check("Metrics returned",       metrics is not None)
    check("Accuracy in [0, 1]",     0.0 <= metrics.accuracy <= 1.0)
    check("Top-5 >= Top-1",         metrics.top_k_accuracy >= metrics.accuracy)
    check("200 samples evaluated",  metrics.total_samples == 200)
    check("10 per-class entries",   len(metrics.per_class_accuracy) == 10)

    print(f"\n{'─' * 60}")
    print(metrics.summary())
    print(f"{'─' * 60}")

    # ── 6. Per-class breakdown ─────────────────────────────────────────────────
    section("6 · Per-Class Results")
    for cls_id, acc in sorted(metrics.per_class_accuracy.items()):
        bar = "█" * int(acc * 30)
        print(f"  {CIFAR10_CLASSES[cls_id]:12s}  {acc * 100:5.1f}%  {bar}")

    print("\n" + "═" * 60)
    print("  Day 6 verification PASSED")
    print("  Architecture → Compile → Train → Evaluate: COMPLETE")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()