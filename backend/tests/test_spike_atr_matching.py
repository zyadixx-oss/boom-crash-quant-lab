import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from diagnose_spike_atr_matching import assign_atr_bins, fit_atr_edges, matched_estimate


def test_matching_uses_signal_mixture_instead_of_baseline_population_mixture():
    # Population:90% low-rate group. Signals:90% high-rate group, so raw lift exaggerates.
    result = matched_estimate([900, 100], [90, 60], [10, 90], [3, 60])
    assert np.isclose(result["precision"], 0.63)
    assert np.isclose(result["matched_baseline"], 0.55)
    assert np.isclose(result["matched_lift"], 0.63 / 0.55)
    assert np.isclose(result["difference"], 0.08)
    assert result["available"]


def test_one_bin_reduces_exactly_to_ordinary_precision_base_and_lift():
    result = matched_estimate([100], [20], [10], [4])
    assert np.isclose(result["precision"], 0.4)
    assert np.isclose(result["matched_baseline"], 0.2)
    assert np.isclose(result["matched_lift"], 2.0)
    assert np.isclose(result["difference"], 0.2)


def test_signal_bin_without_baseline_exposure_is_unavailable():
    result = matched_estimate([100, 0], [20, 0], [0, 5], [0, 2])
    assert result["missing_baseline_exposure"]
    assert not result["available"]
    assert np.isnan(result["matched_baseline"])
    assert np.isnan(result["matched_lift"])


def test_unused_empty_bin_does_not_poison_matching_or_bootstrap_rows():
    result = matched_estimate([[0, 100], [0, 200]], [[0, 20], [0, 40]],
                              [[0, 10], [0, 20]], [[0, 4], [0, 8]])
    np.testing.assert_array_equal(result["available"], [True, True])
    np.testing.assert_allclose(result["matched_baseline"], [0.2, 0.2])
    np.testing.assert_allclose(result["matched_lift"], [2, 2])


def test_frozen_bin_edges_ignore_any_changed_final_atr_values():
    normalized = np.linspace(0.001, 0.01, 100)
    development = np.arange(100) < 70
    original = fit_atr_edges(normalized, development)
    altered = normalized.copy()
    altered[~development] *= 1000
    np.testing.assert_array_equal(original, fit_atr_edges(altered, development))
    assignments = assign_atr_bins(np.array([0.00001, 100, np.nan]), original)
    np.testing.assert_array_equal(assignments, [0, 9, -1])
