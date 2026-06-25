"""ExperimentTrackingUseCase — save, load, list, and compare experiments."""
from __future__ import annotations

from neuroforge_core.domain.entities.experiment_snapshot import ExperimentSnapshot
from neuroforge_core.domain.interfaces.i_experiment_repository import IExperimentRepository


class ExperimentTrackingUseCase:
    """
    Orchestrates experiment persistence and comparison.

    Depends only on IExperimentRepository — the concrete storage backend
    (FileExperimentTracker, future DB adapter, etc.) is injected.
    """

    def __init__(self, repository: IExperimentRepository) -> None:
        self._repo = repository

    # ------------------------------------------------------------------ #
    # CRUD
    # ------------------------------------------------------------------ #

    def save(self, snapshot: ExperimentSnapshot) -> ExperimentSnapshot:
        """Persist a snapshot and return it (convenient for chaining)."""
        self._repo.save(snapshot)
        return snapshot

    def load(self, experiment_id: str) -> ExperimentSnapshot:
        """Load a snapshot by id. Raises KeyError if not found."""
        return self._repo.load(experiment_id)

    def list_all(self) -> list[ExperimentSnapshot]:
        """Return all snapshots sorted by created_at ascending."""
        return self._repo.list_all()

    def delete(self, experiment_id: str) -> None:
        """Remove a snapshot. Silent no-op if not found."""
        self._repo.delete(experiment_id)

    # ------------------------------------------------------------------ #
    # Comparison
    # ------------------------------------------------------------------ #

    def compare(self, experiment_ids: list[str]) -> list[dict]:
        """
        Build a comparison table from a list of experiment ids.

        Returns
        -------
        list[dict]
            One dict per metric row.  Keys: ``"Metric"`` plus one key per
            experiment name.  Example::

                [
                    {"Metric": "Layers", "SmallCNN": "3", "DeepCNN": "5"},
                    {"Metric": "Top-1 Accuracy", "SmallCNN": "32.4%", "DeepCNN": "38.7%"},
                    ...
                ]

            Format is deliberately pandas-friendly: pass directly to
            ``pd.DataFrame(rows)`` or ``st.dataframe(rows)`` in Day 9.
        """
        snapshots = [self._repo.load(eid) for eid in experiment_ids]
        return self._build_comparison_rows(snapshots)

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_comparison_rows(snapshots: list[ExperimentSnapshot]) -> list[dict]:
        names = [s.name for s in snapshots]

        def row(metric: str, values: list) -> dict:
            r: dict = {"Metric": metric}
            for name, val in zip(names, values):
                r[name] = val
            return r

        def fmt_pct(v) -> str:
            return f"{v * 100:.2f}%" if v is not None else "N/A"

        def fmt_loss(v) -> str:
            return f"{v:.4f}" if v is not None else "N/A"

        rows: list[dict] = []

        # --- Architecture ---
        rows.append(row("Num Layers", [s.architecture_summary.get("num_layers", "?") for s in snapshots]))
        rows.append(row("Num Classes", [s.architecture_summary.get("num_classes", "?") for s in snapshots]))

        # --- Training config ---
        rows.append(row("Optimizer", [s.training_config.get("optimizer", "?") for s in snapshots]))
        rows.append(row("Learning Rate", [s.training_config.get("learning_rate", "?") for s in snapshots]))
        rows.append(row("Epochs", [s.training_config.get("epochs", "?") for s in snapshots]))
        rows.append(row("Weight Decay", [s.training_config.get("weight_decay", "?") for s in snapshots]))
        rows.append(row("Momentum", [s.training_config.get("momentum", "?") for s in snapshots]))

        # --- Results ---
        rows.append(row("Final Train Loss", [fmt_loss(s.results.get("final_train_loss")) for s in snapshots]))
        rows.append(row("Top-1 Accuracy", [fmt_pct(s.results.get("top1_accuracy")) for s in snapshots]))
        rows.append(row("Top-5 Accuracy", [fmt_pct(s.results.get("top5_accuracy")) for s in snapshots]))
        rows.append(row("Mean Eval Loss", [fmt_loss(s.results.get("mean_eval_loss")) for s in snapshots]))

        # --- Metadata ---
        rows.append(row("Experiment ID", [s.experiment_id[:8] + "..." for s in snapshots]))
        rows.append(row("Created At", [s.created_at[:19].replace("T", " ") for s in snapshots]))

        return rows