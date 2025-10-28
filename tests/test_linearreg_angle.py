"""
Test suite for LINEARREG_ANGLE
"""

import numpy as np
import pytest

def test_linearreg_angle_basic():
    """Test LINEARREG_ANGLE basic calculation"""
    from talib_pure import LINEARREG_ANGLE

    # Linear uptrend
    close = np.linspace(100, 120, 30)

    result = LINEARREG_ANGLE(close, timeperiod=10)

    # First timeperiod-1 values should be NaN
    assert np.all(np.isnan(result[:9]))
    # After lookback, should have valid values
    assert not np.isnan(result[-1])


def test_linearreg_angle_uptrend():
    """Test LINEARREG_ANGLE shows positive angle in uptrend"""
    from talib_pure import LINEARREG_ANGLE

    # Strong uptrend
    close = np.linspace(100, 150, 30)

    result = LINEARREG_ANGLE(close, timeperiod=10)

    # In uptrend, angle should be positive
    valid_values = result[~np.isnan(result)]
    assert np.all(valid_values > 0)  # Positive angles


def test_linearreg_angle_downtrend():
    """Test LINEARREG_ANGLE shows negative angle in downtrend"""
    from talib_pure import LINEARREG_ANGLE

    # Downtrend
    close = np.linspace(150, 100, 30)

    result = LINEARREG_ANGLE(close, timeperiod=10)

    # In downtrend, angle should be negative
    valid_values = result[~np.isnan(result)]
    assert np.all(valid_values < 0)  # Negative angles


def test_linearreg_angle_flat():
    """Test LINEARREG_ANGLE shows near-zero angle for flat data"""
    from talib_pure import LINEARREG_ANGLE

    # Flat market
    close = np.ones(30) * 100

    result = LINEARREG_ANGLE(close, timeperiod=10)

    # For flat data, angle should be near zero
    valid_values = result[~np.isnan(result)]
    assert np.all(np.abs(valid_values) < 1.0)  # Near zero


def test_linearreg_angle_range():
    """Test LINEARREG_ANGLE values are in valid range"""
    from talib_pure import LINEARREG_ANGLE

    np.random.seed(42)
    close = np.random.randn(50) * 10 + 100

    result = LINEARREG_ANGLE(close, timeperiod=10)

    # Angle should be between -90 and 90 degrees
    valid_values = result[~np.isnan(result)]
    assert np.all(valid_values >= -90)
    assert np.all(valid_values <= 90)


def test_linearreg_angle_empty_input():
    """Test LINEARREG_ANGLE with empty array"""
    from talib_pure import LINEARREG_ANGLE

    empty = np.array([])
    result = LINEARREG_ANGLE(empty)
    assert len(result) == 0


def test_linearreg_angle_input_validation():
    """Test LINEARREG_ANGLE validates input"""
    from talib_pure import LINEARREG_ANGLE

    close = np.array([100, 105])

    with pytest.raises(ValueError):
        LINEARREG_ANGLE(close, timeperiod=1)  # timeperiod must be >= 2
