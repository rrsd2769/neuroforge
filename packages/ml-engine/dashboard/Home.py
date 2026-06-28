"""NeuroForge Dashboard — Home Page"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from dashboard.config import CONFIG
from dashboard.components.api_client import NeuroForgeClient
from dashboard.components.charts import accuracy_trend_chart
from dashboard.components.ui_helpers import (
    api_status_badge,
    format_accuracy,
    format_loss,
    get_top1,
    get_top5,
    get_eval_loss,
    has_results,
)

st.set_page_config(
    page_title=CONFIG.page_title,
    page_icon=CONFIG.page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_client() -> NeuroForgeClient:
    return NeuroForgeClient()


def main() -> None:
    client = get_client()

    st.title("🧠 NeuroForge")
    st.caption("Neural Architecture Search · Training · Evaluation")
    st.divider()

    with st.spinner("Checking API…"):
        healthy = client.health_check()
    api_status_badge(healthy)

    if not healthy:
        st.warning(
            "API not reachable. Start it:\n\n"
            "```bash\nuvicorn api.main:app --reload --port 8000\n```"
        )
        st.stop()

    st.divider()

    try:
        experiments = client.list_experiments()
    except Exception as exc:
        st.error(f"Could not load experiments: {exc}")
        experiments = []

    completed = [e for e in experiments if has_results(e)]
    accuracies = [get_top1(e) for e in completed if get_top1(e) is not None]
    best_acc = max(accuracies) if accuracies else None

    # Best model callout
    if best_acc is not None:
        best_exp = max(completed, key=lambda e: get_top1(e) or 0)
        st.success(
            f"🏆 **Best model:** {best_exp.get('name', 'Unknown')} — "
            f"Top-1: **{format_accuracy(get_top1(best_exp))}** | "
            f"Top-5: **{format_accuracy(get_top5(best_exp))}** | "
            f"Eval Loss: **{format_loss(get_eval_loss(best_exp))}**"
        )

    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Experiments", len(experiments))
    col2.metric("Completed", len(completed))
    col3.metric("Best Top-1", format_accuracy(best_acc))

    # Improvement: first vs last completed
    if len(completed) >= 2:
        sorted_completed = sorted(completed, key=lambda e: e.get("created_at", ""))
        first_acc = get_top1(sorted_completed[0]) or 0
        last_acc = get_top1(sorted_completed[-1]) or 0
        delta = (last_acc - first_acc) * 100
        col4.metric(
            "Accuracy Gain",
            format_accuracy(last_acc),
            delta=f"{delta:+.1f}pp vs first run",
        )
    else:
        col4.metric("Running", sum(1 for e in experiments if not has_results(e)))

    # Trend chart
    if len(completed) >= 2:
        st.divider()
        st.plotly_chart(accuracy_trend_chart(completed), use_container_width=True)
    elif len(completed) == 0:
        st.info("No completed experiments yet. Run the seed script or submit a run.")

    st.divider()

    st.subheader("Navigate")
    nav1, nav2, nav3 = st.columns(3)
    with nav1:
        st.markdown("### ⚗️ Run Experiment")
        st.markdown("Build architecture layer-by-layer and launch a training run.")
    with nav2:
        st.markdown("### 🧪 Experiments")
        st.markdown("Compare runs side-by-side with accuracy and loss charts.")
    with nav3:
        st.markdown("### 📊 Results")
        st.markdown("Deep-dive into any run — metrics, config, and architecture.")

    st.caption(f"API: {CONFIG.api_base_url} · Navigate using the sidebar →")


if __name__ == "__main__":
    main()