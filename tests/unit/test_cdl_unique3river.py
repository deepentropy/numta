"""
Test suite for CDLUNIQUE3RIVER
"""

import numpy as np


def test_cdlunique3river_pattern():
    """Test Unique 3 River pattern detection"""
    from numta import CDLUNIQUE3RIVER

    # Build context + unique 3 river (long black, black harami, small white)
    open_ = np.array([90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                      110, 102, 100.1])  # Long black, black harami, small white
    high = np.array([91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                     110, 103, 100.3])
    low = np.array([90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                    100, 99, 100])  # low[11] < low[10]
    close = np.array([91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                      100, 100.5, 100.3])  # Long black, small black harami (close>prev), small white

    result = CDLUNIQUE3RIVER(open_, high, low, close)
    assert result[-1] == 100, f"Expected 100, got {result[-1]}"


def test_cdlunique3river_no_pattern():
    """Test no pattern when not matching criteria"""
    from numta import CDLUNIQUE3RIVER

    open_ = np.array([90, 91, 110, 105, 104])
    high = np.array([91, 92, 110, 106, 105])
    low = np.array([90, 91, 100, 104, 104])
    close = np.array([91, 92, 100, 105, 105])  # Not matching pattern

    result = CDLUNIQUE3RIVER(open_, high, low, close)
    assert result[-1] == 0
