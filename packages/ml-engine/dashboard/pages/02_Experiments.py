"""Experiments browser page."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import pandas as pd

from dashboard.config import CONFIG
from dashboard.components.api_client import NeuroForgeClient, NeuroForgeAPIError
from dashboard.components.charts import accuracy_bar_chart, top1_vs_top5_chart, loss_scatter , comparison_bar_chart
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
    page_title=f"Experiments | {CONFIG.page_title}",
    page_icon="🧪",
    layout="wide",
)


@st.cache_resource
def get_client() -> NeuroForgeClient:
    return NeuroForgeClient()


def _build_dataframe(experiments: list[dict]) -> pd.DataFrame:
    rows = []
    for exp in experiments:
        tc = exp.get("training_config") or {}
        arch = exp.get("architecture_summary") or {}
        rows.append({
            "Name": exp.get("name", "—"),
            "ID": get_exp_id(exp)[:12] + "…",
            "Top-1 Acc": format_accuracy(get_top1(exp)),
            "Top-5 Acc": format_accuracy(get_top5(exp)),
            "Train Loss": format_loss(get_train_loss(exp)),
            "Eval Loss": format_loss(get_eval_loss(exp)),
            "Epochs": tc.get("epochs", "—"),
            "LR": tc.get("learning_rate", "—"),
            "Optimizer": tc.get("optimizer", "—"),
            "Layers": arch.get("num_layers", "—"),
            "Created": exp.get("created_at", "")[:19].replace("T", " "),
            "_full_id": get_exp_id(exp),
        })
    return pd.DataFrame(rows)


def main() -> None:
    client = get_client()

    st.title("🧪 Experiments")
    st.caption("Browse and compare all tracked training runs.")

    if not client.health_check():
        api_status_badge(False)
        st.stop()

    if st.button("🔄 Refresh"):
        st.cache_data.clear()

    try:
        experiments = client.list_experiments()
    except NeuroForgeAPIError as exc:
        st.error(f"Could not load experiments: {exc}")
        return

    if not experiments:
        st.info("No experiments yet. Go to **Run Experiment** to create one.")
        return

    with_results = [e for e in experiments if has_results(e)]

    # ── Summary ───────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Total", len(experiments))
    col2.metric("With Results", len(with_results))
    best = max((get_top1(e) for e in with_results if get_top1(e) is not None), default=None)
    col3.metric("Best Top-1", format_accuracy(best))

    # ── Charts ────────────────────────────────────────────────────────────────
    if len(with_results) >= 2:
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(accuracy_bar_chart(with_results), use_container_width=True)
        with c2:
            st.plotly_chart(top1_vs_top5_chart(with_results), use_container_width=True)
        st.plotly_chart(loss_scatter(with_results), use_container_width=True)
    elif len(with_results) == 1:
        st.divider()
        st.plotly_chart(accuracy_bar_chart(with_results), use_container_width=True)

    # ── Table ─────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("All Runs")
    df = _build_dataframe(experiments)
    display_df = df.drop(columns=["_full_id"])
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Send to Results page ──────────────────────────────────────────────────
    st.divider()
    st.subheader("Inspect Experiment")

    full_ids = [get_exp_id(e) for e in experiments]
    names = [e.get("name", get_exp_id(e)[:8]) for e in experiments]
    selected = st.selectbox(
        "Select experiment",
        options=full_ids,
        format_func=lambda x: next(
            (e.get("name", x[:8]) for e in experiments if get_exp_id(e) == x), x[:8]
        ),
    )

    # ── Multi-experiment comparison ───────────────────────────────────────────────
    st.divider()
    st.subheader("Compare Experiments")
    st.caption("Select 2 or more experiments to compare side-by-side.")

    if "comparison_selection" not in st.session_state:
        st.session_state["comparison_selection"] = []

    exp_options = {
        e.get("name", get_exp_id(e)[:8]): e
        for e in with_results
}

    selected_names = st.multiselect(
        "Select experiments",
        options=list(exp_options.keys()),
        default=st.session_state["comparison_selection"],
        key="comparison_multiselect",
    )
    st.session_state["comparison_selection"] = selected_names

    if len(selected_names) >= 2:
        selected_exps = [exp_options[n] for n in selected_names]
        st.plotly_chart(
            comparison_bar_chart(selected_exps), use_container_width=True
        )

    # Side-by-side metric table
        comp_rows = []
        for exp in selected_exps:
            tc = exp.get("training_config") or {}
            arch = exp.get("architecture_summary") or {}
            comp_rows.append({
                "Name": exp.get("name", "—"),
                "Top-1": format_accuracy(get_top1(exp)),
                "Top-5": format_accuracy(get_top5(exp)),
                "Train Loss": format_loss(get_train_loss(exp)),
                "Eval Loss": format_loss(get_eval_loss(exp)),
                "Epochs": tc.get("epochs", "—"),
                "LR": tc.get("learning_rate", "—"),
                "Optimizer": tc.get("optimizer", "—"),
                "Layers": arch.get("num_layers", "—"),
            })
        import pandas as pd
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    elif len(selected_names) == 1:
        st.info("Select at least one more experiment to compare.")

    if selected and st.button("📊 View Results"):
        st.session_state["results_experiment_id"] = selected
        st.success("Experiment selected. Open the **Results** page.")


if __name__ == "__main__":
    main()