"""NeuroForge Dashboard — Home Page

Run with:
    streamlit run dashboard/Home.py
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from dashboard.config import CONFIG
from dashboard.components.api_client import NeuroForgeClient
from dashboard.components.ui_helpers import (
    api_status_badge,
    format_accuracy,
    get_top1,
    get_exp_id,
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
            "The FastAPI backend is not reachable. Start it with:\n\n"
            "```bash\ncd packages/ml-engine\n"
            "uvicorn api.main:app --reload --port 8000\n```"
        )
        st.stop()

    st.divider()

    try:
        experiments = client.list_experiments()
    except Exception as exc:
        st.error(f"Could not load experiments: {exc}")
        experiments = []

    # Only count experiments that have results
    completed = [e for e in experiments if has_results(e)]
    accuracies = [get_top1(e) for e in completed if get_top1(e) is not None]
    best_acc = max(accuracies) if accuracies else None

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Experiments", len(experiments))
    col2.metric("With Results", len(completed))
    col3.metric("Best Top-1 Accuracy", format_accuracy(best_acc))

    st.divider()

    st.subheader("Where to go")
    nav1, nav2, nav3 = st.columns(3)

    with nav1:
        st.markdown("### ⚗️ Run Experiment")
        st.markdown(
            "Build an architecture layer-by-layer, set training config, "
            "and submit a full training run."
        )
    with nav2:
        st.markdown("### 🧪 Experiments")
        st.markdown(
            "Browse all tracked runs. Compare top-1/top-5 accuracy and "
            "loss across experiments."
        )
    with nav3:
        st.markdown("### 📊 Results")
        st.markdown(
            "Deep-dive into a single experiment — view accuracy, loss, "
            "and full architecture details."
        )

    st.caption(f"API: {CONFIG.api_base_url} · Navigate using the sidebar →")


if __name__ == "__main__":
    main()