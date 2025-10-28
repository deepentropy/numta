"""
Test suite for CDLRISEFALL3METHODS
"""

import numpy as np
import pytest

def test_cdlrisefall3methods_rising():
    """Test Rising Three Methods pattern detection"""
    from talib_pure import CDLRISEFALL3METHODS

    # Build context + pattern
    open_ = np.array([90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                      100, 109.5, 108.5, 107.5, 105])  # Long white, 3 small black, long white
    high = np.array([91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                     110, 109.8, 109, 108, 116])  # Candle 2 high must be < first high
    low = np.array([90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                    100, 108.5, 107.5, 106.5, 105])
    close = np.array([91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                      110, 109, 108, 107, 116])  # Long white, 3 small black, long white

    result = CDLRISEFALL3METHODS(open_, high, low, close)
    assert result[-1] == 100, f"Expected 100, got {result[-1]}"


def test_cdlrisefall3methods_falling():
    """Test Falling Three Methods pattern detection"""
    from talib_pure import CDLRISEFALL3METHODS

    # Build context + pattern
    open_ = np.array([90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                      110, 100.5, 101.5, 102.5, 95])  # Long black, 3 small white, long black
    high = np.array([91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                     110, 101.5, 102.5, 103.5, 95])
    low = np.array([90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                    100, 100.5, 101, 102, 84])  # Lows must be > first low
    close = np.array([91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                      100, 101, 102, 103, 84])  # Long black, 3 small white, long black

    result = CDLRISEFALL3METHODS(open_, high, low, close)
    assert result[-1] == -100, f"Expected -100, got {result[-1]}"


def test_cdlrisefall3methods_no_pattern():
    """Test no pattern when conditions not met"""
    from talib_pure import CDLRISEFALL3METHODS

    # Wrong colors
    open_ = np.array([90, 91, 100, 101, 102, 103, 104])
    high = np.array([91, 92, 110, 102, 103, 104, 115])
    low = np.array([90, 91, 100, 100, 101, 102, 104])
    close = np.array([91, 92, 110, 100.5, 101.5, 102.5, 115])  # Middle candles wrong color

    result = CDLRISEFALL3METHODS(open_, high, low, close)
    assert result[-1] == 0
