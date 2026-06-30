"""Run Experiment page — build architecture + submit training run."""
from __future__ import annotations

import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st

from dashboard.config import CONFIG
from dashboard.components.api_client import NeuroForgeClient, NeuroForgeAPIError
from dashboard.components.charts import layer_composition_pie
from dashboard.components.ui_helpers import api_status_badge

st.set_page_config(
    page_title=f"Run Experiment | {CONFIG.page_title}",
    page_icon="⚗️",
    layout="wide",
)

# ── Default architecture (matches the snapshot.json example) ─────────────────
DEFAULT_LAYERS = [
    {"type": "conv", "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1, "activation": "relu"},
    {"type": "pool", "pool_size": 2, "stride": 2, "pool_type": "max"},
    {"type": "flatten"},
    {"type": "dense", "units": 128, "activation": "relu"},
    {"type": "dropout", "rate": 0.5},
    {"type": "dense", "units": 10, "activation": "none"},
]


@st.cache_resource
def get_client() -> NeuroForgeClient:
    return NeuroForgeClient()


def _init_state() -> None:
    if "layers" not in st.session_state:
        st.session_state["layers"] = list(DEFAULT_LAYERS)
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None


def _render_layer_stack() -> None:
    """Show current layers with delete buttons."""
    layers = st.session_state["layers"]
    if not layers:
        st.info("No layers yet. Add layers below.")
        return

    st.markdown("**Current architecture:**")
    for i, layer in enumerate(layers):
        col_desc, col_del = st.columns([6, 1])
        layer_type = layer.get("type", "unknown")

        # Human-readable summary per layer type
        if layer_type == "conv":
            desc = (f"[{i+1}] Conv — {layer.get('out_channels')} channels, "
                    f"kernel {layer.get('kernel_size')}, "
                    f"activation: {layer.get('activation')}")
        elif layer_type == "pool":
            desc = (f"[{i+1}] Pool — {layer.get('pool_type')}, "
                    f"size {layer.get('pool_size')}")
        elif layer_type == "flatten":
            desc = f"[{i+1}] Flatten"
        elif layer_type == "dense":
            desc = (f"[{i+1}] Dense — {layer.get('units')} units, "
                    f"activation: {layer.get('activation')}")
        elif layer_type == "dropout":
            desc = f"[{i+1}] Dropout — rate {layer.get('rate')}"
        else:
            desc = f"[{i+1}] {layer_type}"

        col_desc.markdown(f"`{desc}`")
        if col_del.button("🗑", key=f"del_{i}", help="Remove this layer"):
            st.session_state["layers"].pop(i)
            st.rerun()


def _add_layer_form() -> None:
    """Form to append a new layer."""
    st.markdown("**Add a layer:**")
    layer_type = st.selectbox(
        "Layer type",
        ["conv", "pool", "flatten", "dense", "dropout"],
        key="new_layer_type",
    )

    new_layer: dict = {"type": layer_type}

    if layer_type == "conv":
        c1, c2, c3, c4 = st.columns(4)
        new_layer["out_channels"] = c1.number_input("Out channels", 1, 512, 32, key="conv_ch")
        new_layer["kernel_size"] = c2.number_input("Kernel size", 1, 7, 3, key="conv_ks")
        new_layer["stride"] = c3.number_input("Stride", 1, 4, 1, key="conv_s")
        new_layer["padding"] = c4.number_input("Padding", 0, 4, 1, key="conv_p")
        new_layer["activation"] = st.selectbox("Activation", ["relu", "tanh", "sigmoid", "none"], key="conv_act")

    elif layer_type == "pool":
        c1, c2, c3 = st.columns(3)
        new_layer["pool_type"] = c1.selectbox("Pool type", ["max", "avg"], key="pool_t")
        new_layer["pool_size"] = c2.number_input("Pool size", 1, 8, 2, key="pool_sz")
        new_layer["stride"] = c3.number_input("Stride", 1, 4, 2, key="pool_s")

    elif layer_type == "flatten":
        st.caption("No parameters needed.")

    elif layer_type == "dense":
        c1, c2 = st.columns(2)
        new_layer["units"] = c1.number_input("Units", 1, 4096, 128, key="dense_u")
        new_layer["activation"] = c2.selectbox("Activation", ["relu", "tanh", "sigmoid", "none"], key="dense_act")

    elif layer_type == "dropout":
        new_layer["rate"] = st.slider("Dropout rate", 0.0, 0.9, 0.5, 0.05, key="drop_r")

    if st.button("➕ Add Layer", type="primary"):
        st.session_state["layers"].append(new_layer)
        st.rerun()

def main() -> None:
    client = get_client()
    _init_state()

    st.title("⚗️ Run Experiment")
    st.caption("Build an architecture layer-by-layer and launch a training run.")

    if not client.health_check():
        api_status_badge(False)
        st.stop()

    # ── Active training poll ──────────────────────────────────────────────────
    # Check at the TOP before rendering anything else
    polling_id = st.session_state.get("polling_experiment_id")
    if polling_id:
        try:
            exp = client.get_experiment(polling_id)
            status = exp.get("status", "unknown")

            if status == "pending":
                st.info("⏳ Experiment queued — training will start shortly…")
                st.caption(f"Experiment ID: `{polling_id}`")
                time.sleep(5)
                st.rerun()

            elif status == "running":
                st.info("🔄 Training in progress…")
                st.caption(f"Experiment ID: `{polling_id}`")
                tc = exp.get("training_config", {})
                st.caption(
                    f"Epochs: {tc.get('epochs', '?')} | "
                    f"LR: {tc.get('learning_rate', '?')} | "
                    f"Optimizer: {tc.get('optimizer', '?')}"
                )
                time.sleep(8)
                st.rerun()

            elif status == "completed":
                results = exp.get("results") or {}
                top1 = results.get("top1_accuracy", 0)
                top5 = results.get("top5_accuracy", 0)
                train_loss = results.get("final_train_loss", 0)
                eval_loss = results.get("mean_eval_loss", 0)

                st.success("✅ Training complete!")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Top-1 Accuracy", f"{top1 * 100:.2f}%")
                m2.metric("Top-5 Accuracy", f"{top5 * 100:.2f}%")
                m3.metric("Train Loss", f"{train_loss:.4f}")
                m4.metric("Eval Loss", f"{eval_loss:.4f}")

                if top1 < 0.10:
                    st.warning(
                        "Top-1 below random chance (10%). "
                        "Try more epochs, deeper network, or more training samples."
                    )

                st.session_state["results_experiment_id"] = polling_id
                st.info("Open the **Results** page for full breakdown.")

                if st.button("🔁 Run Another Experiment"):
                    del st.session_state["polling_experiment_id"]
                    st.rerun()
                return  # Don't show the form while results are displayed

            elif status == "failed":
                error = exp.get("tags", {}).get("error", "Unknown error")
                st.error(f"❌ Training failed: {error}")
                if st.button("🔁 Try Again"):
                    del st.session_state["polling_experiment_id"]
                    st.rerun()
                return

        except Exception as exc:
            st.error(f"Could not check training status: {exc}")
            if st.button("Clear"):
                del st.session_state["polling_experiment_id"]
                st.rerun()
            return

    st.divider()

    # ── Layout: left = builder, right = config ────────────────────────────────
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Architecture Builder")

        col_load, col_clear = st.columns(2)
        if col_load.button("📋 Load Default"):
            st.session_state["layers"] = list(DEFAULT_LAYERS)
            st.rerun()
        if col_clear.button("🗑 Clear All"):
            st.session_state["layers"] = []
            st.rerun()

        st.markdown("---")
        _render_layer_stack()
        st.markdown("---")
        _add_layer_form()

        if st.session_state["layers"]:
            st.plotly_chart(
                layer_composition_pie(st.session_state["layers"]),
                use_container_width=True,
            )

    with right:
        st.subheader("Training Config")

        exp_name = st.text_input("Experiment name", value="MyExperiment")
        num_classes = st.number_input("Num classes", 2, 1000, 10)
        epochs = st.number_input("Epochs", 1, 100, 5)
        learning_rate = st.select_slider(
            "Learning rate",
            options=[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
            value=0.001,
        )
        optimizer = st.selectbox("Optimizer", ["adam", "sgd"])
        weight_decay = st.number_input("Weight decay", 0.0, 0.1, 0.0, format="%.4f")
        momentum = st.number_input("Momentum (SGD only)", 0.0, 1.0, 0.9, format="%.2f")

        st.markdown("---")
        st.subheader("Dataset Config")
        train_samples = st.number_input("Train samples", 100, 50000, 500, step=100)
        test_samples = st.number_input("Test samples", 50, 10000, 100, step=50)

        st.markdown("---")

        layers = st.session_state["layers"]
        has_flatten = any(l.get("type") == "flatten" for l in layers)
        last_is_dense = layers and layers[-1].get("type") == "dense"

        if not layers:
            st.warning("Add at least one layer.")
        elif not has_flatten:
            st.warning("Architecture needs a Flatten layer before Dense layers.")
        elif not last_is_dense:
            st.warning("Last layer should be Dense (output layer).")
        else:
            estimated_seconds = epochs * (train_samples / 100) * 1.5
            estimated_min = max(1, round(estimated_seconds / 60))
            st.caption(
                f"⏱ Estimated time: ~{estimated_min} min. "
                "Page stays responsive — training runs in background."
            )

            if st.button("🚀 Run Experiment", type="primary", use_container_width=True):
                architecture = {"num_classes": num_classes, "layers": layers}
                training_config = {
                    "epochs": epochs,
                    "learning_rate": float(learning_rate),
                    "optimizer": optimizer,
                    "weight_decay": float(weight_decay),
                    "momentum": float(momentum),
                }
                dataset_config = {
                    "train_samples": train_samples,
                    "test_samples": test_samples,
                }

                try:
                    result = client.run_experiment(
                        name=exp_name,
                        architecture=architecture,
                        training_config=training_config,
                        dataset_config=dataset_config,
                    )
                    exp_id = result.get("experiment_id", "")
                    st.session_state["polling_experiment_id"] = exp_id
                    st.rerun()  # immediately enter the polling loop above
                except Exception as exc:
                    st.error(f"Failed to submit experiment: {exc}")


if __name__ == "__main__":
    main()

