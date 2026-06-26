"""Reusable Streamlit UI components."""
from __future__ import annotations

import streamlit as st


def api_status_badge(is_healthy: bool) -> None:
    if is_healthy:
        st.markdown("🟢 **API Status:** Connected",
                    help="NeuroForge FastAPI server is reachable")
    else:
        st.markdown("🔴 **API Status:** Unreachable",
                    help="Start: uvicorn api.main:app --reload --port 8000")


def format_accuracy(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.2f}%"


def format_loss(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.4f}"


def get_top1(experiment: dict) -> float | None:
    """Extract top-1 accuracy from actual Day 8 response shape."""
    results = experiment.get("results") or {}
    return results.get("top1_accuracy")


def get_top5(experiment: dict) -> float | None:
    results = experiment.get("results") or {}
    return results.get("top5_accuracy")


def get_train_loss(experiment: dict) -> float | None:
    results = experiment.get("results") or {}
    return results.get("final_train_loss")


def get_eval_loss(experiment: dict) -> float | None:
    results = experiment.get("results") or {}
    return results.get("mean_eval_loss")


def get_exp_id(experiment: dict) -> str:
    """Day 8 uses 'experiment_id', not 'id'."""
    return experiment.get("experiment_id", "")


def has_results(experiment: dict) -> bool:
    return bool(experiment.get("results"))