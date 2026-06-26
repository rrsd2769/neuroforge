"""Tests for ui_helpers field extraction — verifies Day 8 field name mapping."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from dashboard.components.ui_helpers import (
    format_accuracy,
    format_loss,
    get_top1,
    get_top5,
    get_train_loss,
    get_eval_loss,
    get_exp_id,
    has_results,
)

FULL_EXPERIMENT = {
    "experiment_id": "48b39a74-bfb0-42bb-b42d-d881a21348e9",
    "name": "ManualTest",
    "results": {
        "top1_accuracy": 0.08,
        "top5_accuracy": 0.42,
        "final_train_loss": 2.371849,
        "mean_eval_loss": 2.474097,
    },
}

EMPTY_EXPERIMENT = {
    "experiment_id": "cccc-3333",
    "name": "NoResults",
    "results": None,
}


class TestFieldExtraction:
    def test_get_exp_id_uses_experiment_id_key(self):
        # Critical: Day 8 uses 'experiment_id' not 'id'
        assert get_exp_id(FULL_EXPERIMENT) == "48b39a74-bfb0-42bb-b42d-d881a21348e9"

    def test_get_exp_id_missing_returns_empty(self):
        assert get_exp_id({}) == ""

    def test_get_top1_reads_results_top1_accuracy(self):
        assert get_top1(FULL_EXPERIMENT) == pytest.approx(0.08)

    def test_get_top5_reads_results_top5_accuracy(self):
        assert get_top5(FULL_EXPERIMENT) == pytest.approx(0.42)

    def test_get_train_loss_reads_final_train_loss(self):
        assert get_train_loss(FULL_EXPERIMENT) == pytest.approx(2.371849)

    def test_get_eval_loss_reads_mean_eval_loss(self):
        assert get_eval_loss(FULL_EXPERIMENT) == pytest.approx(2.474097)

    def test_all_getters_return_none_when_results_is_none(self):
        assert get_top1(EMPTY_EXPERIMENT) is None
        assert get_top5(EMPTY_EXPERIMENT) is None
        assert get_train_loss(EMPTY_EXPERIMENT) is None
        assert get_eval_loss(EMPTY_EXPERIMENT) is None

    def test_has_results_true_when_results_present(self):
        assert has_results(FULL_EXPERIMENT) is True

    def test_has_results_false_when_none(self):
        assert has_results(EMPTY_EXPERIMENT) is False

    def test_has_results_false_when_missing(self):
        assert has_results({}) is False


class TestFormatters:
    def test_format_accuracy_multiplies_by_100(self):
        assert format_accuracy(0.85) == "85.00%"

    def test_format_accuracy_none_returns_dash(self):
        assert format_accuracy(None) == "—"

    def test_format_loss_four_decimal_places(self):
        assert format_loss(2.371849) == "2.3718"

    def test_format_loss_none_returns_dash(self):
        assert format_loss(None) == "—"