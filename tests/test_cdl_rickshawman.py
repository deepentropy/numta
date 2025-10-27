"""
Test suite for CDLRICKSHAWMAN
"""

import numpy as np
import pytest

def test_cdlrickshawman_pattern():
    """Test Rickshaw Man pattern detection"""
    from numta import CDLRICKSHAWMAN

    # Build context + pattern (need larger body candles in context for proper doji detection)
    open_ = np.array([90, 100, 92, 102, 94, 104, 96, 106, 98, 108,
                      100.05])  # Very small body (0.05)
    high = np.array([91, 105, 93, 107, 95, 109, 97, 111, 99, 113,
                     115])  # Very long upper shadow
    low = np.array([90, 100, 92, 102, 94, 104, 96, 106, 98, 108,
                    85])  # Very long lower shadow
    close = np.array([91, 105, 93, 107, 95, 109, 97, 111, 99, 113,
                      100])  # Doji (body = 0.05), avg_body ~7, body < 0.1*7 = 0.7

    result = CDLRICKSHAWMAN(open_, high, low, close)
    assert result[-1] == 100, f"Expected 100, got {result[-1]}"


def test_cdlrickshawman_no_pattern():
    """Test no pattern when conditions not met"""
    from numta import CDLRICKSHAWMAN

    # No long shadows
    open_ = np.array([90, 91, 100])
    high = np.array([91, 92, 101])  # Short upper shadow
    low = np.array([90, 91, 99])  # Short lower shadow
    close = np.array([91, 92, 100])

    result = CDLRICKSHAWMAN(open_, high, low, close)
    assert result[-1] == 0
