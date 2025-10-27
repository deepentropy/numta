"""
Test suite for CDLTRISTAR
"""

import numpy as np


def test_cdltristar_bearish():
    """Test Bearish Tristar (gap up, three dojis)"""
    from numta import CDLTRISTAR

    # Build context + tristar (three dojis with gap up)
    open_ = np.array([90, 100, 92, 102, 94, 104, 96, 106, 98, 108,
                      100.0, 105.0, 105.1])  # Three dojis, second gaps up
    high = np.array([95, 110, 97, 112, 99, 114, 101, 116, 103, 118,
                     100.1, 105.1, 105.0])  # Third not higher
    low = np.array([85, 95, 87, 97, 89, 99, 91, 101, 93, 103,
                    99.9, 104.9, 105.0])
    close = np.array([91, 105, 93, 107, 95, 109, 97, 111, 99, 113,
                      100.0, 105.0, 105.1])  # All dojis, gap up between 1st and 2nd

    result = CDLTRISTAR(open_, high, low, close)
    assert result[-1] == -100, f"Expected -100, got {result[-1]}"


def test_cdltristar_bullish():
    """Test Bullish Tristar (gap down, three dojis)"""
    from numta import CDLTRISTAR

    open_ = np.array([90, 100, 92, 102, 94, 104, 96, 106, 98, 108,
                      105.0, 100.0, 100.1])  # Three dojis, second gaps down
    high = np.array([95, 110, 97, 112, 99, 114, 101, 116, 103, 118,
                     105.1, 100.1, 100.2])
    low = np.array([85, 95, 87, 97, 89, 99, 91, 101, 93, 103,
                    104.9, 99.9, 100.0])
    close = np.array([91, 105, 93, 107, 95, 109, 97, 111, 99, 113,
                      105.0, 100.0, 100.1])  # All dojis, gap down

    result = CDLTRISTAR(open_, high, low, close)
    assert result[-1] == 100, f"Expected 100, got {result[-1]}"


def test_cdltristar_no_pattern():
    """Test no pattern when not dojis"""
    from numta import CDLTRISTAR

    open_ = np.array([90, 91, 100, 105, 110])
    high = np.array([91, 92, 105, 110, 115])
    low = np.array([90, 91, 100, 105, 110])
    close = np.array([91, 92, 105, 110, 115])  # Not dojis

    result = CDLTRISTAR(open_, high, low, close)
    assert result[-1] == 0
