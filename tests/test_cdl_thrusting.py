"""
Test suite for CDLTHRUSTING
"""

import numpy as np
import pytest

def test_cdlthrusting_pattern():
    """Test Thrusting Pattern detection"""
    from numta import CDLTHRUSTING

    # Build context + thrusting pattern (black, white closes below midpoint)
    open_ = np.array([90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                      110, 99])  # Black, white opens below low
    high = np.array([91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                     110, 105])
    low = np.array([90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                    100, 98])  # Opens below low[10]=100
    close = np.array([91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                      100, 104.9])  # Closes below midpoint (100 + 10/2 = 105)

    result = CDLTHRUSTING(open_, high, low, close)
    assert result[-1] == -100, f"Expected -100, got {result[-1]}"


def test_cdlthrusting_no_pattern():
    """Test no pattern when white closes above midpoint"""
    from numta import CDLTHRUSTING

    open_ = np.array([90, 91, 110, 101])
    high = np.array([91, 92, 110, 106])
    low = np.array([90, 91, 100, 100])
    close = np.array([91, 92, 100, 106])  # Closes above midpoint

    result = CDLTHRUSTING(open_, high, low, close)
    assert result[-1] == 0
