"""
Test suite for CDLSTICKSANDWICH
"""

import numpy as np
import pytest

def test_cdlsticksandwich_pattern():
    """Test Stick Sandwich pattern detection"""
    from numta import CDLSTICKSANDWICH

    # Build context + stick sandwich (black-white-black with matching closes)
    open_ = np.array([90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                      110, 101, 110])  # Black, white gap up, black
    high = np.array([91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                     110, 105, 110])
    low = np.array([90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                    100, 101, 100])  # Second gaps above first close (101 > 100)
    close = np.array([91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                      100, 105, 100])  # First and third match

    result = CDLSTICKSANDWICH(open_, high, low, close)
    assert result[-1] == 100, f"Expected 100, got {result[-1]}"


def test_cdlsticksandwich_no_pattern():
    """Test no pattern when closes don't match"""
    from numta import CDLSTICKSANDWICH

    open_ = np.array([90, 91, 110, 95, 110])
    high = np.array([91, 92, 110, 105, 110])
    low = np.array([90, 91, 100, 95, 100])
    close = np.array([91, 92, 100, 105, 95])  # Closes don't match

    result = CDLSTICKSANDWICH(open_, high, low, close)
    assert result[-1] == 0
