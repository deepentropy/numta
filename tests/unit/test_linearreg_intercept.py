"""
Test suite for LINEARREG_INTERCEPT
"""

import numpy as np
import pytest

def test_linearreg_intercept_basic():
    """Test LINEARREG_INTERCEPT basic calculation"""
    from numta import LINEARREG_INTERCEPT

    # Linear uptrend
    close = np.linspace(100, 120, 20)

    result = LINEARREG_INTERCEPT(close, timeperiod=10)

    # First timeperiod-1 values should be NaN
    assert np.all(np.isnan(result[:9]))
    # After lookback, should have valid values
    assert not np.isnan(result[-1])


def test_linearreg_intercept_relationship():
    """Test relationship between LINEARREG, INTERCEPT, and SLOPE"""
    from numta import LINEARREG, LINEARREG_INTERCEPT, LINEARREG_SLOPE

    close = np.linspace(100, 120, 30)
    timeperiod = 10

    linreg = LINEARREG(close, timeperiod=timeperiod)
    intercept = LINEARREG_INTERCEPT(close, timeperiod=timeperiod)
    slope = LINEARREG_SLOPE(close, timeperiod=timeperiod)

    # LINEARREG should equal intercept + slope * (timeperiod-1)
    valid_idx = ~np.isnan(linreg)
    if np.sum(valid_idx) > 0:
        reconstructed = intercept[valid_idx] + slope[valid_idx] * (timeperiod - 1)
        np.testing.assert_array_almost_equal(linreg[valid_idx], reconstructed, decimal=10)


def test_linearreg_intercept_empty_input():
    """Test LINEARREG_INTERCEPT with empty array"""
    from numta import LINEARREG_INTERCEPT

    empty = np.array([])
    result = LINEARREG_INTERCEPT(empty)
    assert len(result) == 0


def test_linearreg_intercept_input_validation():
    """Test LINEARREG_INTERCEPT validates input"""
    from numta import LINEARREG_INTERCEPT

    close = np.array([100, 105])

    with pytest.raises(ValueError):
        LINEARREG_INTERCEPT(close, timeperiod=1)
