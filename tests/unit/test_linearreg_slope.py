"""
Test suite for LINEARREG_SLOPE
"""

import numpy as np
import pytest

def test_linearreg_slope_basic():
    """Test LINEARREG_SLOPE basic calculation"""
    from numta import LINEARREG_SLOPE

    # Linear uptrend
    close = np.linspace(100, 120, 30)

    result = LINEARREG_SLOPE(close, timeperiod=10)

    # First timeperiod-1 values should be NaN
    assert np.all(np.isnan(result[:9]))
    # After lookback, should have valid values
    assert not np.isnan(result[-1])


def test_linearreg_slope_uptrend():
    """Test LINEARREG_SLOPE shows positive slope in uptrend"""
    from numta import LINEARREG_SLOPE

    # Strong uptrend
    close = np.linspace(100, 150, 30)

    result = LINEARREG_SLOPE(close, timeperiod=10)

    # In uptrend, slope should be positive
    valid_values = result[~np.isnan(result)]
    assert np.all(valid_values > 0)


def test_linearreg_slope_downtrend():
    """Test LINEARREG_SLOPE shows negative slope in downtrend"""
    from numta import LINEARREG_SLOPE

    # Downtrend
    close = np.linspace(150, 100, 30)

    result = LINEARREG_SLOPE(close, timeperiod=10)

    # In downtrend, slope should be negative
    valid_values = result[~np.isnan(result)]
    assert np.all(valid_values < 0)


def test_linearreg_slope_flat():
    """Test LINEARREG_SLOPE shows near-zero slope for flat data"""
    from numta import LINEARREG_SLOPE

    # Flat market
    close = np.ones(30) * 100

    result = LINEARREG_SLOPE(close, timeperiod=10)

    # For flat data, slope should be near zero
    valid_values = result[~np.isnan(result)]
    assert np.all(np.abs(valid_values) < 0.0001)


def test_linearreg_slope_empty_input():
    """Test LINEARREG_SLOPE with empty array"""
    from numta import LINEARREG_SLOPE

    empty = np.array([])
    result = LINEARREG_SLOPE(empty)
    assert len(result) == 0


def test_linearreg_slope_input_validation():
    """Test LINEARREG_SLOPE validates input"""
    from numta import LINEARREG_SLOPE

    close = np.array([100, 105])

    with pytest.raises(ValueError):
        LINEARREG_SLOPE(close, timeperiod=1)
