"""
Test suite for CDLSEPARATINGLINES
"""

import numpy as np
import pytest

def test_cdlseparatinglines_bullish():
    """Test Separating Lines bullish pattern detection"""
    from talib_pure import CDLSEPARATINGLINES

    # Build context + pattern
    open_ = np.array([90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                      110, 110])  # Black, white with matching opens
    high = np.array([91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                     110, 120])
    low = np.array([90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                    100, 110])
    close = np.array([91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                      100, 120])  # Long black, long white, matching opens

    result = CDLSEPARATINGLINES(open_, high, low, close)
    assert result[-1] == 100, f"Expected 100, got {result[-1]}"


def test_cdlseparatinglines_bearish():
    """Test Separating Lines bearish pattern detection"""
    from talib_pure import CDLSEPARATINGLINES

    # Build context + pattern
    open_ = np.array([90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                      100, 100])  # White, black with matching opens
    high = np.array([91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                     110, 100])
    low = np.array([90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                    100, 90])
    close = np.array([91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                      110, 90])  # Long white, long black, matching opens

    result = CDLSEPARATINGLINES(open_, high, low, close)
    assert result[-1] == -100, f"Expected -100, got {result[-1]}"


def test_cdlseparatinglines_no_pattern():
    """Test no pattern when opens don't match"""
    from talib_pure import CDLSEPARATINGLINES

    open_ = np.array([90, 91, 110, 105])  # Opens don't match
    high = np.array([91, 92, 110, 115])
    low = np.array([90, 91, 100, 105])
    close = np.array([91, 92, 100, 115])

    result = CDLSEPARATINGLINES(open_, high, low, close)
    assert result[-1] == 0
