"""Plotly chart builders for the NeuroForge dashboard."""
from __future__ import annotations

from typing import Any
import plotly.graph_objects as go

BRAND_COLORS = {
    "primary": "#6366f1",
    "success": "#22c55e",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "muted": "#94a3b8",
}

_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e2e8f0"),
)


def accuracy_bar_chart(experiments: list[dict[str, Any]]) -> go.Figure:
    """Bar chart of top-1 accuracy across experiments."""
    if not experiments:
        return _empty_figure("No experiments to compare")

    # Day 8 field: results.top1_accuracy, id field: experiment_id
    ids = [exp.get("experiment_id", "")[:8] for exp in experiments]
    names = [exp.get("name", ids[i]) for i, exp in enumerate(experiments)]
    accuracies = [
        (exp.get("results") or {}).get("top1_accuracy", 0) * 100
        for exp in experiments
    ]

    fig = go.Figure(go.Bar(
        x=names,
        y=accuracies,
        marker_color=BRAND_COLORS["primary"],
        text=[f"{a:.1f}%" for a in accuracies],
        textposition="outside",
    ))
    fig.update_layout(
        title="Top-1 Accuracy by Experiment",
        xaxis_title="Experiment Name",
        yaxis_title="Accuracy (%)",
        yaxis_range=[0, 110],
        height=380,
        **_LAYOUT,
    )
    return fig


def loss_scatter(experiments: list[dict[str, Any]]) -> go.Figure:
    """Scatter: train loss vs eval loss per experiment."""
    if not experiments:
        return _empty_figure("No experiments to plot")

    names = [exp.get("name", exp.get("experiment_id", "")[:8]) for exp in experiments]
    train_losses = [
        (exp.get("results") or {}).get("final_train_loss", 0) for exp in experiments
    ]
    eval_losses = [
        (exp.get("results") or {}).get("mean_eval_loss", 0) for exp in experiments
    ]

    fig = go.Figure(go.Scatter(
        x=train_losses,
        y=eval_losses,
        mode="markers+text",
        text=names,
        textposition="top center",
        marker=dict(
            size=14,
            color=eval_losses,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Eval Loss"),
        ),
    ))
    fig.update_layout(
        title="Train Loss vs Eval Loss",
        xaxis_title="Final Train Loss",
        yaxis_title="Mean Eval Loss",
        height=400,
        **_LAYOUT,
    )
    return fig


def top1_vs_top5_chart(experiments: list[dict[str, Any]]) -> go.Figure:
    """Grouped bar: top-1 vs top-5 accuracy per experiment."""
    if not experiments:
        return _empty_figure("No experiments")

    names = [exp.get("name", exp.get("experiment_id", "")[:8]) for exp in experiments]
    top1 = [(exp.get("results") or {}).get("top1_accuracy", 0) * 100 for exp in experiments]
    top5 = [(exp.get("results") or {}).get("top5_accuracy", 0) * 100 for exp in experiments]

    fig = go.Figure([
        go.Bar(name="Top-1", x=names, y=top1, marker_color=BRAND_COLORS["primary"]),
        go.Bar(name="Top-5", x=names, y=top5, marker_color=BRAND_COLORS["success"]),
    ])
    fig.update_layout(
        barmode="group",
        title="Top-1 vs Top-5 Accuracy",
        yaxis_title="Accuracy (%)",
        yaxis_range=[0, 110],
        height=380,
        **_LAYOUT,
    )
    return fig


def layer_composition_pie(layers: list[dict]) -> go.Figure:
    """Pie chart of layer types in a given architecture."""
    if not layers:
        return _empty_figure("No layers")

    from collections import Counter
    counts = Counter(layer.get("type", "unknown") for layer in layers)

    fig = go.Figure(go.Pie(
        labels=list(counts.keys()),
        values=list(counts.values()),
        hole=0.4,
        marker_colors=[
            BRAND_COLORS["primary"],
            BRAND_COLORS["success"],
            BRAND_COLORS["warning"],
            BRAND_COLORS["danger"],
            BRAND_COLORS["muted"],
        ],
    ))
    fig.update_layout(
        title="Layer Type Distribution",
        height=320,
        **_LAYOUT,
    )
    return fig

def accuracy_trend_chart(experiments: list[dict[str, Any]]) -> go.Figure:
    """
    Line chart of top-1 accuracy ordered by creation time.
    Shows whether accuracy is improving across successive experiments.
    """
    if not experiments:
        return _empty_figure("No experiments yet")

    # Sort by created_at ascending
    sorted_exps = sorted(
        [e for e in experiments if (e.get("results") or {}).get("top1_accuracy") is not None],
        key=lambda e: e.get("created_at", ""),
    )

    if not sorted_exps:
        return _empty_figure("No completed experiments")

    names = [e.get("name", e.get("experiment_id", "")[:8]) for e in sorted_exps]
    top1 = [(e.get("results") or {}).get("top1_accuracy", 0) * 100 for e in sorted_exps]
    top5 = [(e.get("results") or {}).get("top5_accuracy", 0) * 100 for e in sorted_exps]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=names, y=top1, mode="lines+markers+text",
        name="Top-1", line=dict(color=BRAND_COLORS["primary"], width=2),
        marker=dict(size=10),
        text=[f"{v:.1f}%" for v in top1],
        textposition="top center",
    ))
    fig.add_trace(go.Scatter(
        x=names, y=top5, mode="lines+markers",
        name="Top-5", line=dict(color=BRAND_COLORS["success"], width=2, dash="dot"),
        marker=dict(size=8),
    ))
    # Random-chance baseline for CIFAR-10
    fig.add_hline(
        y=10, line_dash="dash", line_color=BRAND_COLORS["muted"],
        annotation_text="Random chance (10%)", annotation_position="right",
    )
    fig.update_layout(
        title="Accuracy Trend Across Experiments",
        xaxis_title="Experiment (chronological)",
        yaxis_title="Accuracy (%)",
        yaxis_range=[0, 105],
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        **_LAYOUT,
    )
    return fig


def comparison_bar_chart(experiments: list[dict[str, Any]]) -> go.Figure:
    """
    Grouped bar chart for a selected subset of experiments.
    Shows top-1, top-5, and inverted-loss side by side for direct comparison.
    """
    if not experiments:
        return _empty_figure("Select experiments to compare")

    names = [e.get("name", e.get("experiment_id", "")[:8]) for e in experiments]
    top1 = [(e.get("results") or {}).get("top1_accuracy", 0) * 100 for e in experiments]
    top5 = [(e.get("results") or {}).get("top5_accuracy", 0) * 100 for e in experiments]

    fig = go.Figure([
        go.Bar(name="Top-1 Acc (%)", x=names, y=top1,
               marker_color=BRAND_COLORS["primary"],
               text=[f"{v:.1f}%" for v in top1], textposition="outside"),
        go.Bar(name="Top-5 Acc (%)", x=names, y=top5,
               marker_color=BRAND_COLORS["success"],
               text=[f"{v:.1f}%" for v in top5], textposition="outside"),
    ])
    fig.update_layout(
        barmode="group",
        title="Head-to-Head Comparison",
        yaxis_title="Accuracy (%)",
        yaxis_range=[0, 115],
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        **_LAYOUT,
    )
    return fig


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color=BRAND_COLORS["muted"]),
    )
    fig.update_layout(height=300, **_LAYOUT)
    return fig