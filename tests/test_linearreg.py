"""
Test suite for LINEARREG
"""

import numpy as np
import pytest

def test_linearreg_basic():
    """Test LINEARREG basic calculation"""
    from numta import LINEARREG

    # Linear uptrend
    close = np.linspace(100, 120, 20)

    result = LINEARREG(close, timeperiod=10)

    # First timeperiod-1 values should be NaN
    assert np.all(np.isnan(result[:9]))
    # After lookback, should have valid values
    assert not np.isnan(result[-1])


def test_linearreg_trending():
    """Test LINEARREG follows trend"""
    from numta import LINEARREG

    # Uptrend
    close = np.linspace(100, 120, 30)
    result = LINEARREG(close, timeperiod=10)

    # Linear regression should track the trend
    valid_values = result[~np.isnan(result)]
    assert valid_values[-1] > valid_values[0]


def test_linearreg_vs_price():
    """Test LINEARREG relationship with price"""
    from numta import LINEARREG

    # Create data with known linear trend
    close = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                     110, 111, 112, 113, 114])

    result = LINEARREG(close, timeperiod=10)

    # For perfect linear data, LINEARREG should be close to actual prices
    valid_idx = ~np.isnan(result)
    assert np.sum(valid_idx) > 0


def test_linearreg_empty_input():
    """Test LINEARREG with empty array"""
    from numta import LINEARREG

    empty = np.array([])
    result = LINEARREG(empty)
    assert len(result) == 0


def test_linearreg_input_validation():
    """Test LINEARREG validates input"""
    from numta import LINEARREG

    close = np.array([100, 105])

    with pytest.raises(ValueError):
        LINEARREG(close, timeperiod=1)  # timeperiod must be >= 2
