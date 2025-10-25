"""
Test suite for CDLXSIDEGAP3METHODS
"""

import numpy as np
import pytest

def test_cdlxsidegap3methods_upside():
    """Test Upside Gap Three Methods pattern"""
    from talib_pure import CDLXSIDEGAP3METHODS

    # Build context + upside gap three methods
    open_ = np.array([90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                      100, 106, 108])  # White, gap up white, black closes gap
    high = np.array([91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                     105, 110, 109])
    low = np.array([90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                    100, 106, 101])  # Gap: low[11]=106 > high[10]=105
    close = np.array([91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                      105, 110, 101])  # White(5), white(4), black(-7) closes gap

    result = CDLXSIDEGAP3METHODS(open_, high, low, close)
    assert result[-1] == 100, f"Expected 100, got {result[-1]}"


def test_cdlxsidegap3methods_downside():
    """Test Downside Gap Three Methods pattern"""
    from talib_pure import CDLXSIDEGAP3METHODS

    # Build context + downside gap three methods
    open_ = np.array([90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                      115, 104, 102])  # Black, gap down black, white closes gap
    high = np.array([91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                     115, 104, 112])  # Gap: high[11]=104 < low[10]=110
    low = np.array([90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                    110, 100, 101])
    close = np.array([91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                      110, 100, 112])  # Black(-5), black(-4), white(10) closes gap (112 > 110)

    result = CDLXSIDEGAP3METHODS(open_, high, low, close)
    assert result[-1] == -100, f"Expected -100, got {result[-1]}"


def test_cdlxsidegap3methods_no_pattern():
    """Test no pattern when gap not closed"""
    from talib_pure import CDLXSIDEGAP3METHODS

    open_ = np.array([90, 91, 100, 106, 107])
    high = np.array([91, 92, 105, 110, 108])
    low = np.array([90, 91, 100, 106, 107])
    close = np.array([91, 92, 105, 110, 108])  # Gap not closed

    result = CDLXSIDEGAP3METHODS(open_, high, low, close)
    assert result[-1] == 0
