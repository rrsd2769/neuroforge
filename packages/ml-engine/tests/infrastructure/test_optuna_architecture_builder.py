"""Tests for OptunaArchitectureBuilder — pure construction logic, no real training."""
from __future__ import annotations

import optuna
import pytest

from neuroforge_core.domain.value_objects.search_space import SearchSpace
from neuroforge_core.infrastructure.generators.optuna_architecture_builder import (
    OptunaArchitectureBuilder,
)


@pytest.fixture
def search_space() -> SearchSpace:
    return SearchSpace(min_depth=1, max_depth=3, channel_choices=(16, 32, 64))


@pytest.fixture
def builder(search_space) -> OptunaArchitectureBuilder:
    return OptunaArchitectureBuilder(search_space)


def _get_trial() -> optuna.Trial:
    """Get a real Optuna Trial object without running a full study.optimize() loop."""
    study = optuna.create_study()
    return study.ask()


class TestOptunaArchitectureBuilder:
    def test_builds_valid_architecture(self, builder):
        trial = _get_trial()
        arch = builder.build(trial, num_classes=10, input_shape=(3, 32, 32))
        assert arch.is_valid_for_input((3, 32, 32))

    def test_ends_with_dense_output_layer(self, builder):
        trial = _get_trial()
        arch = builder.build(trial, num_classes=10, input_shape=(3, 32, 32))
        assert arch.layers[-1].units == 10

    def test_respects_num_classes(self, builder):
        trial = _get_trial()
        arch = builder.build(trial, num_classes=7, input_shape=(3, 32, 32))
        assert arch.num_classes == 7
        assert arch.layers[-1].units == 7

    def test_multiple_trials_produce_different_architectures(self, builder):
        study = optuna.create_study()
        archs = []
        for _ in range(5):
            trial = study.ask()
            arch = builder.build(trial, num_classes=10, input_shape=(3, 32, 32))
            archs.append(arch.layer_count())
        # With randomized suggest_int/categorical, not all 5 should be identical depth
        assert len(set(archs)) > 1
