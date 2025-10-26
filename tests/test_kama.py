"""
Test suite for KAMA
"""

import numpy as np
import pytest

def test_kama_basic():
    """Test KAMA basic calculation"""
    from talib_pure import KAMA

    close = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
                     110, 112, 111, 113, 115, 114, 116, 118, 117, 119,
                     120, 122, 121, 123, 125, 124, 126, 128, 127, 129, 130])

    result = KAMA(close, timeperiod=10)

    # First timeperiod values should be NaN
    assert np.all(np.isnan(result[:10]))
    # After lookback, should have valid values
    assert not np.isnan(result[-1])


def test_kama_trending():
    """Test KAMA follows trend"""
    from talib_pure import KAMA

    # Strong uptrend
    close = np.linspace(100, 150, 50)

    result = KAMA(close, timeperiod=10)

    # KAMA should be increasing in uptrend
    valid_values = result[~np.isnan(result)]
    assert valid_values[-1] > valid_values[0]


def test_kama_efficiency():
    """Test that KAMA responds to market efficiency"""
    from talib_pure import KAMA

    # Create efficient trending market vs choppy market
    np.random.seed(42)

    # Efficient trend
    trend = np.linspace(100, 150, 50)
    kama_trend = KAMA(trend, timeperiod=10)

    # Choppy market
    choppy = np.ones(50) * 100 + np.random.randn(50) * 5
    kama_choppy = KAMA(choppy, timeperiod=10)

    # Both should have valid values
    assert not np.isnan(kama_trend[-1])
    assert not np.isnan(kama_choppy[-1])


def test_kama_empty_input():
    """Test KAMA with empty array"""
    from talib_pure import KAMA

    empty = np.array([])
    result = KAMA(empty)
    assert len(result) == 0


def test_kama_input_validation():
    """Test KAMA validates input"""
    from talib_pure import KAMA

    close = np.array([100, 105])

    with pytest.raises(ValueError):
        KAMA(close, timeperiod=1)  # timeperiod must be >= 2
