"""
Test suite for HT_DCPERIOD
"""

import numpy as np


def test_ht_dcperiod_basic():
    """Test HT_DCPERIOD basic calculation"""
    from numta import HT_DCPERIOD

    # Create sine wave to simulate a cycle
    n = 100
    close = np.sin(np.linspace(0, 4*np.pi, n)) * 10 + 100

    result = HT_DCPERIOD(close)

    # First 32 values should be NaN (unstable period)
    assert np.all(np.isnan(result[:32]))
    # After stable period, most values should be between 6 and 50
    # (Early values may be lower as period ramps up)
    valid_values = result[~np.isnan(result)]
    assert len(valid_values) > 0
    # Check that values are eventually in the expected range
    later_values = result[50:]  # Check after warmup
    later_valid = later_values[~np.isnan(later_values)]
    if len(later_valid) > 0:
        assert np.all(later_valid >= 6)
        assert np.all(later_valid <= 50)


def test_ht_dcperiod_empty_input():
    """Test HT_DCPERIOD with empty array"""
    from numta import HT_DCPERIOD

    empty = np.array([])
    result = HT_DCPERIOD(empty)
    assert len(result) == 0
