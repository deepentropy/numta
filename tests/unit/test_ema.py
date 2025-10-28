"""
Test suite for EMA
"""

import numpy as np


def test_ema_basic():
    """Test EMA basic calculation"""
    from numta import EMA

    close = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
    result = EMA(close, timeperiod=5)

    # First 4 values (timeperiod-1) should be NaN
    assert np.all(np.isnan(result[:4]))
    # After lookback, should have valid values
    assert not np.isnan(result[-1])


def test_ema_trending():
    """Test EMA follows trend"""
    from numta import EMA

    # Uptrend
    close = np.linspace(100, 120, 20)
    result = EMA(close, timeperiod=5)

    # EMA should be increasing
    valid_values = result[~np.isnan(result)]
    assert valid_values[-1] > valid_values[0]
