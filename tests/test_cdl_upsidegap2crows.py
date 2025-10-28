"""
Test suite for CDLUPSIDEGAP2CROWS
"""

import numpy as np
import pytest

def test_cdlupsidegap2crows_pattern():
    """Test Upside Gap Two Crows pattern detection"""
    from talib_pure import CDLUPSIDEGAP2CROWS

    # Build context + upside gap two crows (white, black gaps up, black engulfs)
    open_ = np.array([90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                      100, 112, 113.5])  # Long white, black gaps up, black engulfs
    high = np.array([91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                     110, 111, 113])
    low = np.array([90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                    100, 111, 110])  # Gap up: low[11]=111 > high[10]=110
    close = np.array([91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                      110, 111, 110.5])  # White, black, black engulfs (close[12]=110.5 > close[10]=110)

    result = CDLUPSIDEGAP2CROWS(open_, high, low, close)
    assert result[-1] == -100, f"Expected -100, got {result[-1]}"


def test_cdlupsidegap2crows_no_pattern():
    """Test no pattern when no gap"""
    from talib_pure import CDLUPSIDEGAP2CROWS

    open_ = np.array([90, 91, 100, 101, 102])
    high = np.array([91, 92, 110, 102, 103])
    low = np.array([90, 91, 100, 100, 101])  # No gap
    close = np.array([91, 92, 110, 100, 101])

    result = CDLUPSIDEGAP2CROWS(open_, high, low, close)
    assert result[-1] == 0
