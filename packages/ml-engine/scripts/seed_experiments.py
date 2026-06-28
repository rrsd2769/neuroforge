"""
Seed NeuroForge with 5 real training experiments.

Run from packages/ml-engine/ with the API already running:
    python scripts/seed_experiments.py

Expected runtime: 15-25 minutes on CPU.
Each experiment differs in depth, width, optimizer, or regularization
so the comparison charts tell a meaningful story.
"""
from __future__ import annotations

import sys
import time
import requests

API_URL = "http://localhost:8000"
TIMEOUT = 600  # 10 minutes per experiment — do not lower


EXPERIMENTS = [
    {
        "name": "Baseline-Tiny",
        "architecture": {
            "num_classes": 10,
            "layers": [
                {"type": "conv", "out_channels": 16, "kernel_size": 3,
                 "stride": 1, "padding": 1, "activation": "relu"},
                {"type": "pool", "pool_size": 2, "stride": 2, "pool_type": "max"},
                {"type": "flatten"},
                {"type": "dense", "units": 10, "activation": "none"},
            ],
        },
        "training_config": {
            "epochs": 3, "learning_rate": 0.001, "optimizer": "adam",
            "weight_decay": 0.0, "momentum": 0.9,
        },
        "dataset_config": {"train_samples": 1000, "test_samples": 200},
        "tags": {"day": "10", "tier": "baseline"},
    },
    {
        "name": "Conv-Medium",
        "architecture": {
            "num_classes": 10,
            "layers": [
                {"type": "conv", "out_channels": 32, "kernel_size": 3,
                 "stride": 1, "padding": 1, "activation": "relu"},
                {"type": "pool", "pool_size": 2, "stride": 2, "pool_type": "max"},
                {"type": "conv", "out_channels": 64, "kernel_size": 3,
                 "stride": 1, "padding": 1, "activation": "relu"},
                {"type": "pool", "pool_size": 2, "stride": 2, "pool_type": "max"},
                {"type": "flatten"},
                {"type": "dense", "units": 128, "activation": "relu"},
                {"type": "dense", "units": 10, "activation": "none"},
            ],
        },
        "training_config": {
            "epochs": 5, "learning_rate": 0.001, "optimizer": "adam",
            "weight_decay": 0.0, "momentum": 0.9,
        },
        "dataset_config": {"train_samples": 2000, "test_samples": 400},
        "tags": {"day": "10", "tier": "medium"},
    },
    {
        "name": "Conv-Dropout",
        "architecture": {
            "num_classes": 10,
            "layers": [
                {"type": "conv", "out_channels": 32, "kernel_size": 3,
                 "stride": 1, "padding": 1, "activation": "relu"},
                {"type": "pool", "pool_size": 2, "stride": 2, "pool_type": "max"},
                {"type": "conv", "out_channels": 64, "kernel_size": 3,
                 "stride": 1, "padding": 1, "activation": "relu"},
                {"type": "pool", "pool_size": 2, "stride": 2, "pool_type": "max"},
                {"type": "flatten"},
                {"type": "dense", "units": 256, "activation": "relu"},
                {"type": "dropout", "rate": 0.5},
                {"type": "dense", "units": 10, "activation": "none"},
            ],
        },
        "training_config": {
            "epochs": 8, "learning_rate": 0.001, "optimizer": "adam",
            "weight_decay": 0.0001, "momentum": 0.9,
        },
        "dataset_config": {"train_samples": 2000, "test_samples": 400},
        "tags": {"day": "10", "tier": "regularized"},
    },
    {
        "name": "SGD-Momentum",
        "architecture": {
            "num_classes": 10,
            "layers": [
                {"type": "conv", "out_channels": 32, "kernel_size": 3,
                 "stride": 1, "padding": 1, "activation": "relu"},
                {"type": "pool", "pool_size": 2, "stride": 2, "pool_type": "max"},
                {"type": "conv", "out_channels": 64, "kernel_size": 3,
                 "stride": 1, "padding": 1, "activation": "relu"},
                {"type": "pool", "pool_size": 2, "stride": 2, "pool_type": "max"},
                {"type": "flatten"},
                {"type": "dense", "units": 128, "activation": "relu"},
                {"type": "dense", "units": 10, "activation": "none"},
            ],
        },
        "training_config": {
            "epochs": 5, "learning_rate": 0.01, "optimizer": "sgd",
            "weight_decay": 0.0001, "momentum": 0.9,
        },
        "dataset_config": {"train_samples": 2000, "test_samples": 400},
        "tags": {"day": "10", "tier": "sgd"},
    },
    {
        "name": "Wide-Best",
        "architecture": {
            "num_classes": 10,
            "layers": [
                {"type": "conv", "out_channels": 64, "kernel_size": 3,
                 "stride": 1, "padding": 1, "activation": "relu"},
                {"type": "pool", "pool_size": 2, "stride": 2, "pool_type": "max"},
                {"type": "conv", "out_channels": 128, "kernel_size": 3,
                 "stride": 1, "padding": 1, "activation": "relu"},
                {"type": "pool", "pool_size": 2, "stride": 2, "pool_type": "max"},
                {"type": "flatten"},
                {"type": "dense", "units": 256, "activation": "relu"},
                {"type": "dropout", "rate": 0.3},
                {"type": "dense", "units": 10, "activation": "none"},
            ],
        },
        "training_config": {
            "epochs": 10, "learning_rate": 0.001, "optimizer": "adam",
            "weight_decay": 0.0001, "momentum": 0.9,
        },
        "dataset_config": {"train_samples": 3000, "test_samples": 500},
        "tags": {"day": "10", "tier": "best"},
    },
]


def check_api() -> bool:
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def run_experiment(spec: dict, index: int, total: int) -> dict | None:
    name = spec["name"]
    epochs = spec["training_config"]["epochs"]
    samples = spec["dataset_config"]["train_samples"]
    estimated_min = round(epochs * samples / 1000 * 0.8)

    print(f"\n[{index}/{total}] Running: {name}")
    print(f"  Epochs: {epochs} | Samples: {samples} | Est. time: ~{estimated_min} min")

    start = time.time()
    try:
        resp = requests.post(
            f"{API_URL}/experiments/run",
            json=spec,
            timeout=TIMEOUT,
        )
        elapsed = round(time.time() - start, 1)

        if resp.status_code == 200:
            result = resp.json()
            results = result.get("results") or {}
            top1 = results.get("top1_accuracy", 0) * 100
            top5 = results.get("top5_accuracy", 0) * 100
            loss = results.get("final_train_loss", 0)
            print(f"  ✅ Done in {elapsed}s")
            print(f"  Top-1: {top1:.1f}% | Top-5: {top5:.1f}% | Train Loss: {loss:.4f}")
            return result
        else:
            print(f"  ❌ Failed: HTTP {resp.status_code} — {resp.text[:200]}")
            return None
    except requests.Timeout:
        print(f"  ❌ Timed out after {TIMEOUT}s")
        return None
    except Exception as exc:
        print(f"  ❌ Error: {exc}")
        return None


def main() -> None:
    print("=" * 55)
    print(" NeuroForge — Experiment Seeder")
    print("=" * 55)

    if not check_api():
        print("\n❌ API not reachable at", API_URL)
        print("   Start it with: uvicorn api.main:app --reload --port 8000")
        sys.exit(1)

    print(f"\n✅ API is up. Running {len(EXPERIMENTS)} experiments...")
    print("   This will take 15-25 minutes on CPU. Leave it running.\n")

    results = []
    for i, spec in enumerate(EXPERIMENTS, 1):
        result = run_experiment(spec, i, len(EXPERIMENTS))
        if result:
            results.append(result)

    print("\n" + "=" * 55)
    print(f" Seeding complete: {len(results)}/{len(EXPERIMENTS)} succeeded")
    print("=" * 55)

    if results:
        print("\nResults summary:")
        print(f"  {'Name':<20} {'Top-1':>8} {'Top-5':>8} {'Loss':>10}")
        print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*10}")
        for r in results:
            res = r.get("results") or {}
            top1 = res.get("top1_accuracy", 0) * 100
            top5 = res.get("top5_accuracy", 0) * 100
            loss = res.get("final_train_loss", 0)
            print(f"  {r.get('name', '?'):<20} {top1:>7.1f}% {top5:>7.1f}% {loss:>10.4f}")

    print("\nRefresh the Experiments page in your dashboard to see the results.")


if __name__ == "__main__":
    main()