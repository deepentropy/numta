"""
Test suite for CDLSTALLEDPATTERN
"""

import numpy as np
import pytest

def test_cdlstalledpattern_pattern():
    """Test Stalled Pattern detection"""
    from numta import CDLSTALLEDPATTERN

    # Build context + three white soldiers with stall
    open_ = np.array([90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                      100, 105, 109.8])  # Three rising white, third opens high
    high = np.array([91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                     105, 110, 110.3])  # Second has very short upper shadow
    low = np.array([90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                    100, 105, 109.5])
    close = np.array([91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                      105, 110, 110.1])  # Long white, long white, small white (higher closes!)

    result = CDLSTALLEDPATTERN(open_, high, low, close)
    assert result[-1] == -100, f"Expected -100, got {result[-1]}"


def test_cdlstalledpattern_no_pattern():
    """Test no pattern when not three white"""
    from numta import CDLSTALLEDPATTERN

    open_ = np.array([90, 91, 100, 105, 109])
    high = np.array([91, 92, 105, 110, 114])
    low = np.array([90, 91, 100, 105, 109])
    close = np.array([91, 92, 105, 110, 108])  # Last is black

    result = CDLSTALLEDPATTERN(open_, high, low, close)
    assert result[-1] == 0
