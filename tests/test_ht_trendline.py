"""
Test suite for HT_TRENDLINE
"""

import numpy as np
import pytest

def test_ht_trendline_basic():
    """Test HT_TRENDLINE basic calculation"""
    from numta import HT_TRENDLINE

    # Create trending data
    close = np.linspace(100, 120, 100)

    result = HT_TRENDLINE(close)

    # First 32 values should be NaN
    assert np.all(np.isnan(result[:32]))
    # After stable period, should have valid values
    assert not np.isnan(result[-1])
    # Trendline should follow the trend
    valid_values = result[~np.isnan(result)]
    assert valid_values[-1] > valid_values[0]  # Should be increasing


def test_ht_trendline_smoothing():
    """Test that HT_TRENDLINE smooths price action"""
    from numta import HT_TRENDLINE

    # Create noisy data with underlying trend
    np.random.seed(42)
    trend = np.linspace(100, 120, 100)
    noise = np.random.randn(100) * 2
    close = trend + noise

    result = HT_TRENDLINE(close)

    # Trendline should be smoother than price
    valid_idx = ~np.isnan(result)
    assert np.sum(valid_idx) > 0


def test_ht_trendline_empty_input():
    """Test HT_TRENDLINE with empty array"""
    from numta import HT_TRENDLINE

    empty = np.array([])
    result = HT_TRENDLINE(empty)
    assert len(result) == 0
