"""
Test suite for CDLTAKURI
"""

import numpy as np
import pytest

def test_cdltakuri_pattern():
    """Test Takuri (Dragonfly Doji) pattern detection"""
    from talib_pure import CDLTAKURI

    # Build context with shadows, then takuri
    open_ = np.array([90, 100, 92, 102, 94, 104, 96, 106, 98, 108,
                      100.001])  # Very small body (0.001)
    high = np.array([95, 110, 97, 112, 99, 114, 101, 116, 103, 118,
                     100.005])  # Very short upper shadow (0.004), context has ~5-10 shadows
    low = np.array([85, 95, 87, 97, 89, 99, 91, 101, 93, 103,
                    70])  # Very long lower shadow (30), avg_shadow ~7.5, need > 15
    close = np.array([91, 105, 93, 107, 95, 109, 97, 111, 99, 113,
                      100])  # Doji

    result = CDLTAKURI(open_, high, low, close)
    assert result[-1] == 100, f"Expected 100, got {result[-1]}"


def test_cdltakuri_no_pattern():
    """Test no pattern when not doji"""
    from talib_pure import CDLTAKURI

    open_ = np.array([90, 91, 100])  # Large body
    high = np.array([91, 92, 101])
    low = np.array([90, 91, 80])
    close = np.array([91, 92, 105])

    result = CDLTAKURI(open_, high, low, close)
    assert result[-1] == 0
