"""Independent mathematical fixtures for a descriptive-only block diagnostic."""
import numpy as np
import pytest

from scripts.diagnose_multiframe_development_uncertainty import intervals


def test_single_possible_three_day_block_has_analytic_distribution():
    result = intervals([1, -1, 2], [1, 0, 2], [0, 1, 0], seed=123, draws=100)
    assert result['CI95_daily_mean_R_descriptive'] == [2/3, 2/3]
    assert result['CI95_PF_descriptive'] == [3, 3]
    assert result['profit_qualified'] is False


def test_no_loss_bootstrap_is_kept_undefined():
    result = intervals([1]*90, [1]*90, [0]*90, seed=1, draws=100)
    assert result['CI95_daily_mean_R_descriptive'] == [1, 1]
    assert result['CI95_PF_descriptive'] is None and result['undefined_PF_draws'] == 100


def test_idle_calendar_is_zero_without_fabricated_trades_or_pf():
    result = intervals([0]*90, [0]*90, [0]*90, seed=0, draws=100)
    assert result['CI95_daily_mean_R_descriptive'] == [0, 0]
    assert result['CI95_PF_descriptive'] is None


def test_sparse_events_have_uncertain_mean_and_seed_is_reproducible():
    r = np.zeros(90);r[10] = 2;r[70] = -3
    first = intervals(r, np.maximum(r, 0), np.maximum(-r, 0), seed=456, draws=2000)
    assert first == intervals(r, np.maximum(r, 0), np.maximum(-r, 0), seed=456, draws=2000)
    assert first['CI95_daily_mean_R_descriptive'][0] < 0 < first['CI95_daily_mean_R_descriptive'][1]


@pytest.mark.parametrize('r,gain,loss', [([1, 2], [1, 2], [0, 0]),
    ([float('nan')]*3, [1]*3, [0]*3), ([1]*3, [2]*3, [0]*3),
    ([1]*3, [-1]*3, [-2]*3), ([1]*3, [1]*2, [0]*3)])
def test_incomplete_or_inconsistent_daily_vectors_rejected(r, gain, loss):
    with pytest.raises(ValueError):
        intervals(r, gain, loss, seed=1, draws=10)
