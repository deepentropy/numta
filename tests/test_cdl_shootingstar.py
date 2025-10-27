"""
Test suite for CDLSHOOTINGSTAR
"""

import numpy as np
import pytest

def test_cdlshootingstar_pattern():
    """Test Shooting Star pattern detection"""
    from numta import CDLSHOOTINGSTAR

    # Build context + pattern (need larger body context candles)
    open_ = np.array([90, 100, 92, 102, 94, 104, 96, 106, 98, 108,
                      114])  # Gap up from prior high (114 > 113)
    high = np.array([91, 105, 93, 107, 95, 109, 97, 111, 99, 113,
                     123])  # Long upper shadow
    low = np.array([90, 100, 92, 102, 94, 104, 96, 106, 98, 108,
                    114])  # Very short lower shadow (0)
    close = np.array([91, 105, 93, 107, 95, 109, 97, 111, 99, 113,
                      114.2])  # Very small body (0.2), upper shadow (8.8)

    result = CDLSHOOTINGSTAR(open_, high, low, close)
    assert result[-1] == -100, f"Expected -100, got {result[-1]}"


def test_cdlshootingstar_no_gap():
    """Test no pattern without gap"""
    from numta import CDLSHOOTINGSTAR

    open_ = np.array([90, 91, 100])  # No gap
    high = np.array([91, 92, 115])
    low = np.array([90, 91, 99])
    close = np.array([91, 92, 102])

    result = CDLSHOOTINGSTAR(open_, high, low, close)
    assert result[-1] == 0
