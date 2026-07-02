"""Architecture Search page — Optuna-driven NAS."""
from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st

from dashboard.config import CONFIG
from dashboard.components.api_client import NeuroForgeClient
from dashboard.components.charts import pareto_front_chart
from dashboard.components.ui_helpers import api_status_badge

st.set_page_config(
    page_title=f"Architecture Search | {CONFIG.page_title}",
    page_icon="🔬",
    layout="wide",
)


@st.cache_resource
def get_client() -> NeuroForgeClient:
    return NeuroForgeClient()


def main() -> None:
    client = get_client()

    st.title("🔬 Architecture Search (Optuna NAS)")
    st.caption(
        "Bayesian-optimization-driven search over architecture and "
        "hyperparameters — each trial informs the next, unlike random sampling."
    )

    if not client.health_check():
        api_status_badge(False)
        st.stop()

    polling_id = st.session_state.get("polling_search_id")
    if polling_id:
        try:
            search = client.get_architecture_search(polling_id)
            status = search.get("status", "unknown")
            trials = search.get("trials", [])
            n_trials = search.get("n_trials", 0)
            completed = search.get("completed_trials", 0)

            if status in ("pending", "running"):
                st.info(f"🔄 Running trial {completed}/{n_trials}…")
                if trials:
                    st.plotly_chart(pareto_front_chart(trials), use_container_width=True)
                time.sleep(6)
                st.rerun()

            elif status == "completed":
                st.success(f"✅ Search complete — {n_trials} trials.")
                best = search.get("best_trial")
                if best:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Best Top-1", f"{best['top1_accuracy']*100:.2f}%")
                    m2.metric("Parameters", f"{best['parameter_count']:,}")
                    m3.metric("Optimizer", best['optimizer'])
                st.plotly_chart(pareto_front_chart(trials), use_container_width=True)

                with st.expander("All trial results"):
                    st.json(trials)

                if st.button("🔁 Run Another Search"):
                    del st.session_state["polling_search_id"]
                    st.rerun()
                return

            elif status == "failed":
                st.error(f"❌ Search failed: {search.get('error', 'Unknown error')}")
                if st.button("🔁 Try Again"):
                    del st.session_state["polling_search_id"]
                    st.rerun()
                return

        except Exception as exc:
            st.error(f"Could not check search status: {exc}")
            if st.button("Clear"):
                del st.session_state["polling_search_id"]
                st.rerun()
            return

    st.divider()
    st.subheader("Search Configuration")

    col1, col2 = st.columns(2)
    with col1:
        n_trials = st.slider("Number of trials", 5, 30, 15)
        epochs_per_trial = st.slider("Epochs per trial", 1, 5, 2)
        min_depth = st.slider("Min conv layers", 1, 4, 2)
    with col2:
        train_samples = st.number_input("Train samples per trial", 200, 5000, 1000, step=100)
        test_samples = st.number_input("Test samples per trial", 50, 2000, 300, step=50)
        max_depth = st.slider("Max conv layers", min_depth, 8, 6)

    est_min = round(n_trials * epochs_per_trial * (train_samples / 1000) * 1.2)
    st.caption(f"⏱ Estimated total time: ~{est_min} min. Runs in background.")

    if st.button("🚀 Run Search", type="primary", use_container_width=True):
        try:
            result = client.run_architecture_search(
                n_trials=n_trials,
                epochs_per_trial=epochs_per_trial,
                train_samples_per_trial=train_samples,
                test_samples_per_trial=test_samples,
                min_depth=min_depth,
                max_depth=max_depth,
            )
            st.session_state["polling_search_id"] = result["search_id"]
            st.rerun()
        except Exception as exc:
            st.error(f"Failed to start search: {exc}")


if __name__ == "__main__":
    main()