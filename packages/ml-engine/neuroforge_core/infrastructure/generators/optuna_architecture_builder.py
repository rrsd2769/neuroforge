"""
OptunaArchitectureBuilder — infrastructure adapter.

Mirrors RandomArchitectureGenerator's construction logic exactly (same
layer types, same retry/validity guard) but substitutes trial.suggest_*()
calls for random.choice()/randint() so Optuna's TPE sampler can learn
from prior trial outcomes instead of sampling blind.

Not a literal ArchitectureGeneratorPort implementation — that port's
generate(search_space) signature has no slot for an Optuna trial object.
This class lives alongside the port rather than forcing an awkward fit.
"""
from __future__ import annotations

from typing import List, Tuple

import optuna

from neuroforge_core.domain.entities.architecture import Architecture
from neuroforge_core.domain.value_objects.layer import (
    ConvLayer,
    DenseLayer,
    DropoutLayer,
    FlattenLayer,
    Layer,
    PoolLayer,
)
from neuroforge_core.domain.value_objects.search_space import SearchSpace


class OptunaArchitectureBuilder:
    """Builds one candidate Architecture per Optuna trial."""

    def __init__(self, search_space: SearchSpace) -> None:
        self._search_space = search_space

    def build(
        self,
        trial: optuna.Trial,
        num_classes: int,
        input_shape: Tuple[int, ...],
    ) -> Architecture:
        """
        Construct a candidate Architecture using trial.suggest_*() at every
        decision point. Raises optuna.TrialPruned() if the result is
        structurally invalid for input_shape — this is a construction
        failure, not a meaningful "bad architecture" signal, so it must
        not be scored as 0% accuracy.
        """
        layers = self._build_conv_block(trial, input_shape)
        layers.append(FlattenLayer())
        layers += self._build_dense_block(trial)
        layers.append(DenseLayer(units=num_classes, activation="softmax"))

        arch = Architecture(layers=layers, num_classes=num_classes)
        if not arch.is_valid_for_input(input_shape):
            raise optuna.TrialPruned(
                f"Trial {trial.number} produced an invalid architecture for "
                f"input_shape={input_shape}."
            )
        return arch

    # ------------------------------------------------------------------
    # Private helpers — mirror RandomArchitectureGenerator's structure
    # ------------------------------------------------------------------

    def _build_conv_block(
        self, trial: optuna.Trial, input_shape: Tuple[int, ...]
    ) -> List[Layer]:
        layers: List[Layer] = []
        shape = input_shape

        n_conv = trial.suggest_int(
            "n_conv_layers", self._search_space.min_depth, self._search_space.max_depth
        )

        for i in range(n_conv):
            out_channels = trial.suggest_categorical(
                f"conv_{i}_channels", list(self._search_space.channel_choices)
            )
            kernel_size = trial.suggest_categorical(f"conv_{i}_kernel", [3, 5])
            padding = kernel_size // 2  # "same" padding — no shrinkage

            conv = ConvLayer(
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            )
            candidate_shape = conv.output_shape(shape)
            if any(d <= 0 for d in candidate_shape):
                break  # safety: abandon this conv, same as RandomArchitectureGenerator

            layers.append(conv)
            shape = candidate_shape

            use_pool = trial.suggest_categorical(f"conv_{i}_pool", [True, False])
            if use_pool and min(shape[1], shape[2]) >= 4:
                pool = PoolLayer(pool_size=2, stride=2)
                candidate_shape = pool.output_shape(shape)
                if all(d > 0 for d in candidate_shape):
                    layers.append(pool)
                    shape = candidate_shape

        return layers

    def _build_dense_block(self, trial: optuna.Trial) -> List[Layer]:
        layers: List[Layer] = []
        n_dense = trial.suggest_int("n_dense_layers", 1, 2)

        for i in range(n_dense):
            units = trial.suggest_categorical(f"dense_{i}_units", [64, 128, 256, 512])
            layers.append(DenseLayer(units=units))

            use_dropout = trial.suggest_categorical(f"dense_{i}_dropout", [True, False])
            if use_dropout:
                rate = trial.suggest_categorical(
                    f"dense_{i}_dropout_rate", [0.25, 0.3, 0.4, 0.5]
                )
                layers.append(DropoutLayer(rate=rate))

        return layers