"""Results page — deep-dive into a single experiment.

NOTE: Day 8 has no separate /evaluate endpoint.
      Results (top1_accuracy, top5_accuracy, losses) are returned
      directly from POST /experiments/run and GET /experiments/{id}.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st

from dashboard.config import CONFIG
from dashboard.components.api_client import NeuroForgeClient, NeuroForgeAPIError
from dashboard.components.charts import layer_composition_pie
from dashboard.components.ui_helpers import (
    api_status_badge,
    format_accuracy,
    format_loss,
    get_top1,
    get_top5,
    get_train_loss,
    get_eval_loss,
    get_exp_id,
    has_results,
)

st.set_page_config(
    page_title=f"Results | {CONFIG.page_title}",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource
def get_client() -> NeuroForgeClient:
    return NeuroForgeClient()


def main() -> None:
    client = get_client()

    st.title("📊 Results")
    st.caption("Inspect the full results of a training run.")

    if not client.health_check():
        api_status_badge(False)
        st.stop()

    # ── Experiment selector ───────────────────────────────────────────────────
    default_id = st.session_state.get("results_experiment_id", "")
    experiment_id = st.text_input(
        "Experiment ID",
        value=default_id,
        placeholder="Paste UUID here, or use 'View Results' from the Experiments page",
    )

    if not experiment_id:
        st.info("Enter an experiment ID above, or select one from the **Experiments** page.")
        return

    if st.button("🔍 Load Results", type="primary"):
        try:
            exp = client.get_experiment(experiment_id)
            st.session_state[f"exp_detail_{experiment_id}"] = exp
        except NeuroForgeAPIError as exc:
            st.error(f"Could not load experiment: {exc}")
            return

    exp = st.session_state.get(f"exp_detail_{experiment_id}")
    if exp is None:
        st.info("Click **Load Results** to fetch this experiment.")
        return

    # ── Header ────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader(f"**{exp.get('name', 'Experiment')}**")
    st.caption(
        f"ID: `{get_exp_id(exp)}` · "
        f"Created: {exp.get('created_at', '')[:19].replace('T', ' ')}"
    )

    if not has_results(exp):
        st.warning("This experiment has no results yet.")
        return

    # ── Accuracy & loss metrics ───────────────────────────────────────────────
    st.markdown("#### Performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Top-1 Accuracy", format_accuracy(get_top1(exp)))
    m2.metric("Top-5 Accuracy", format_accuracy(get_top5(exp)))
    m3.metric("Train Loss", format_loss(get_train_loss(exp)))
    m4.metric("Eval Loss", format_loss(get_eval_loss(exp)))

    # ── Training config ───────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### Training Config")
    tc = exp.get("training_config") or {}
    tc1, tc2, tc3, tc4 = st.columns(4)
    tc1.metric("Epochs", tc.get("epochs", "—"))
    tc2.metric("Learning Rate", tc.get("learning_rate", "—"))
    tc3.metric("Optimizer", tc.get("optimizer", "—"))
    tc4.metric("Weight Decay", tc.get("weight_decay", "—"))

    # ── Architecture summary ──────────────────────────────────────────────────
    st.divider()
    st.markdown("#### Architecture")
    arch = exp.get("architecture_summary") or {}
    a1, a2, a3 = st.columns(3)
    a1.metric("Num Layers", arch.get("num_layers", "—"))
    a2.metric("Num Classes", arch.get("num_classes", "—"))
    a3.metric("Architecture ID", (arch.get("architecture_id") or "—")[:12] + "…")

    layers = arch.get("layers", [])
    if layers:
        left, right = st.columns([2, 3])
        with left:
            st.plotly_chart(layer_composition_pie(layers), use_container_width=True)
        with right:
            st.markdown("**Layer stack:**")
            for i, layer in enumerate(layers):
                ltype = layer.get("type", "unknown")
                if ltype == "conv":
                    detail = (f"out_channels={layer.get('out_channels')}, "
                              f"kernel={layer.get('kernel_size')}, "
                              f"act={layer.get('activation')}")
                elif ltype == "pool":
                    detail = f"{layer.get('pool_type')}, size={layer.get('pool_size')}"
                elif ltype == "dense":
                    detail = f"units={layer.get('units')}, act={layer.get('activation')}"
                elif ltype == "dropout":
                    detail = f"rate={layer.get('rate')}"
                else:
                    detail = ""
                st.markdown(f"`[{i+1}] {ltype.upper()}` — {detail}")

    # ── Raw JSON ─────────────────────────────────────────────────────────────
    with st.expander("Raw experiment JSON"):
        st.json(exp)


if __name__ == "__main__":
    main()