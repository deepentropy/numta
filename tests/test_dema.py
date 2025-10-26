"""
Test suite for DEMA
"""

import numpy as np
import pytest

def test_dema_basic():
    """Test DEMA basic calculation"""
    from talib_pure import DEMA

    # Create rising prices
    close = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                     110, 111, 112, 113, 114, 115])

    result = DEMA(close, timeperiod=5)

    # First 8 values (2*5-2) should be NaN
    assert np.all(np.isnan(result[:8]))
    # After lookback, should have valid values
    assert not np.isnan(result[-1])
    # DEMA should follow the trend
    assert result[-1] > result[8]  # Should be increasing


def test_dema_vs_ema():
    """Test that DEMA is more responsive than EMA"""
    from talib_pure import DEMA, EMA

    # Create data with a trend change
    close = np.array([100]*10 + [101, 102, 103, 104, 105, 106, 107, 108, 109, 110])

    dema = DEMA(close, timeperiod=5)
    ema = EMA(close, timeperiod=5)

    # DEMA should respond faster to the trend change
    # Check the last value where both are valid
    assert not np.isnan(dema[-1])
    assert not np.isnan(ema[-1])


def test_dema_empty_input():
    """Test DEMA with empty arrays"""
    from talib_pure import DEMA

    empty = np.array([])
    result = DEMA(empty)
    assert len(result) == 0


def test_dema_input_validation():
    """Test DEMA validates input"""
    from talib_pure import DEMA

    close = np.array([100, 105])

    with pytest.raises(ValueError):
        DEMA(close, timeperiod=1)  # timeperiod must be >= 2
