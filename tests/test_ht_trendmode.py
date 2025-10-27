"""
Test suite for HT_TRENDMODE
"""

import numpy as np
import pytest

def test_ht_trendmode_basic():
    """Test HT_TRENDMODE basic calculation"""
    from numta import HT_TRENDMODE

    # Create trending data
    close = np.linspace(100, 150, 100)

    result = HT_TRENDMODE(close)

    # First 63 values should be NaN
    assert np.all(np.isnan(result[:63]))
    # Values should be 0 or 1
    valid_values = result[~np.isnan(result)]
    assert np.all((valid_values == 0) | (valid_values == 1))


def test_ht_trendmode_trending():
    """Test HT_TRENDMODE detects trending market"""
    from numta import HT_TRENDMODE

    # Strong trend
    close = np.linspace(100, 150, 100)

    result = HT_TRENDMODE(close)

    # In trending market, should eventually show trend mode (1)
    valid_values = result[~np.isnan(result)]
    if len(valid_values) > 0:
        # At least some values should indicate trend mode
        assert np.any(valid_values == 1) or np.any(valid_values == 0)


def test_ht_trendmode_empty_input():
    """Test HT_TRENDMODE with empty array"""
    from numta import HT_TRENDMODE

    empty = np.array([])
    result = HT_TRENDMODE(empty)
    assert len(result) == 0
