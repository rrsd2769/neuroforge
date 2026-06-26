"""Smoke tests for chart builders using actual Day 8 response shapes."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import plotly.graph_objects as go
import pytest

from dashboard.components.charts import (
    accuracy_bar_chart,
    loss_scatter,
    top1_vs_top5_chart,
    layer_composition_pie,
)

# ── Sample data using actual Day 8 field names ────────────────────────────────

EXPERIMENT_A = {
    "experiment_id": "aaaa-1111",
    "name": "RunA",
    "results": {
        "top1_accuracy": 0.85,
        "top5_accuracy": 0.96,
        "final_train_loss": 0.42,
        "mean_eval_loss": 0.48,
    },
}

EXPERIMENT_B = {
    "experiment_id": "bbbb-2222",
    "name": "RunB",
    "results": {
        "top1_accuracy": 0.78,
        "top5_accuracy": 0.91,
        "final_train_loss": 0.55,
        "mean_eval_loss": 0.60,
    },
}

EXPERIMENT_NO_RESULTS = {
    "experiment_id": "cccc-3333",
    "name": "RunC",
    "results": None,
}

SAMPLE_LAYERS = [
    {"type": "conv"},
    {"type": "conv"},
    {"type": "pool"},
    {"type": "flatten"},
    {"type": "dense"},
    {"type": "dropout"},
    {"type": "dense"},
]


# ── accuracy_bar_chart ────────────────────────────────────────────────────────

class TestAccuracyBarChart:
    def test_returns_figure(self):
        fig = accuracy_bar_chart([EXPERIMENT_A, EXPERIMENT_B])
        assert isinstance(fig, go.Figure)

    def test_uses_experiment_name_not_id(self):
        fig = accuracy_bar_chart([EXPERIMENT_A])
        # x axis should show name, not raw ID
        bar = fig.data[0]
        assert "RunA" in list(bar.x)

    def test_empty_returns_figure(self):
        fig = accuracy_bar_chart([])
        assert isinstance(fig, go.Figure)

    def test_single_experiment(self):
        fig = accuracy_bar_chart([EXPERIMENT_A])
        assert isinstance(fig, go.Figure)


# ── loss_scatter ──────────────────────────────────────────────────────────────

class TestLossScatter:
    def test_returns_figure(self):
        fig = loss_scatter([EXPERIMENT_A, EXPERIMENT_B])
        assert isinstance(fig, go.Figure)

    def test_empty_returns_figure(self):
        fig = loss_scatter([])
        assert isinstance(fig, go.Figure)

    def test_single_experiment(self):
        fig = loss_scatter([EXPERIMENT_A])
        assert isinstance(fig, go.Figure)


# ── top1_vs_top5_chart ────────────────────────────────────────────────────────

class TestTop1VsTop5Chart:
    def test_returns_figure(self):
        fig = top1_vs_top5_chart([EXPERIMENT_A, EXPERIMENT_B])
        assert isinstance(fig, go.Figure)

    def test_has_two_bar_traces(self):
        fig = top1_vs_top5_chart([EXPERIMENT_A, EXPERIMENT_B])
        assert len(fig.data) == 2  # one for top-1, one for top-5

    def test_empty_returns_figure(self):
        fig = top1_vs_top5_chart([])
        assert isinstance(fig, go.Figure)


# ── layer_composition_pie ─────────────────────────────────────────────────────

class TestLayerCompositionPie:
    def test_returns_figure(self):
        fig = layer_composition_pie(SAMPLE_LAYERS)
        assert isinstance(fig, go.Figure)

    def test_empty_returns_figure(self):
        fig = layer_composition_pie([])
        assert isinstance(fig, go.Figure)

    def test_counts_layer_types(self):
        layers = [
            {"type": "conv"},
            {"type": "conv"},
            {"type": "dense"},
        ]
        fig = layer_composition_pie(layers)
        pie = fig.data[0]
        # conv appears twice, dense once
        label_value = dict(zip(pie.labels, pie.values))
        assert label_value["conv"] == 2
        assert label_value["dense"] == 1