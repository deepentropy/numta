"""
Test suite for CDLTASUKIGAP
"""

import numpy as np
import pytest

def test_cdltasukigap_upside():
    """Test Upside Tasuki Gap pattern"""
    from numta import CDLTASUKIGAP

    # Build context + upside tasuki gap (bodies must be similar size)
    open_ = np.array([90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                      100, 106, 109.8])  # White, gap up white, black (opens within 2nd body)
    high = np.array([91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                     105, 110, 110])
    low = np.array([90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                    100, 106, 107])  # Gap: low[11]=106 > high[10]=105
    close = np.array([91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                      105, 110, 106])  # White(5), white(4), black(3.8); close[12]=106 > high[10]=105

    result = CDLTASUKIGAP(open_, high, low, close)
    assert result[-1] == 100, f"Expected 100, got {result[-1]}"


def test_cdltasukigap_downside():
    """Test Downside Tasuki Gap pattern"""
    from numta import CDLTASUKIGAP

    # Build context + downside tasuki gap (bodies must be similar size)
    open_ = np.array([90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                      110, 104, 100.2])  # Black, gap down black, white (opens within 2nd body)
    high = np.array([91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                     110, 104, 104])  # Gap: high[11]=104 < low[10]=110
    low = np.array([90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                    110, 100, 100])
    close = np.array([91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                      105, 100, 104])  # Black(5), black(4), white(3.8); close[12]=104 < low[10]=110

    result = CDLTASUKIGAP(open_, high, low, close)
    assert result[-1] == -100, f"Expected -100, got {result[-1]}"


def test_cdltasukigap_no_pattern():
    """Test no pattern when gap is closed"""
    from numta import CDLTASUKIGAP

    open_ = np.array([90, 91, 100, 106, 107])
    high = np.array([91, 92, 105, 110, 108])
    low = np.array([90, 91, 100, 106, 107])
    close = np.array([91, 92, 105, 110, 99])  # Closes gap

    result = CDLTASUKIGAP(open_, high, low, close)
    assert result[-1] == 0
