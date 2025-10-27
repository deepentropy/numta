"""
Test suite for CDLSPINNINGTOP
"""

import numpy as np
import pytest

def test_cdlspinningtop_white():
    """Test Spinning Top white pattern"""
    from numta import CDLSPINNINGTOP

    # Build context + spinning top
    open_ = np.array([90, 100, 92, 102, 94, 104, 96, 106, 98, 108,
                      100])  # Small body
    high = np.array([91, 105, 93, 107, 95, 109, 97, 111, 99, 113,
                     103])  # Long upper shadow (3)
    low = np.array([90, 100, 92, 102, 94, 104, 96, 106, 98, 108,
                    97])  # Long lower shadow (3)
    close = np.array([91, 105, 93, 107, 95, 109, 97, 111, 99, 113,
                      100.5])  # Small white body (0.5), shadows > body

    result = CDLSPINNINGTOP(open_, high, low, close)
    assert result[-1] == 100, f"Expected 100, got {result[-1]}"


def test_cdlspinningtop_black():
    """Test Spinning Top black pattern"""
    from numta import CDLSPINNINGTOP

    open_ = np.array([90, 100, 92, 102, 94, 104, 96, 106, 98, 108,
                      100.5])
    high = np.array([91, 105, 93, 107, 95, 109, 97, 111, 99, 113,
                     103])
    low = np.array([90, 100, 92, 102, 94, 104, 96, 106, 98, 108,
                    97])
    close = np.array([91, 105, 93, 107, 95, 109, 97, 111, 99, 113,
                      100])  # Small black body

    result = CDLSPINNINGTOP(open_, high, low, close)
    assert result[-1] == -100, f"Expected -100, got {result[-1]}"


def test_cdlspinningtop_no_pattern():
    """Test no pattern when shadows not long enough"""
    from numta import CDLSPINNINGTOP

    open_ = np.array([90, 91, 100])
    high = np.array([91, 92, 100.05])  # Upper shadow = 0.05
    low = np.array([90, 91, 99.96])  # Lower shadow = 0.04, body = 0.1
    close = np.array([91, 92, 100.1])  # Shadows NOT > body (0.05 < 0.1, 0.04 < 0.1)

    result = CDLSPINNINGTOP(open_, high, low, close)
    assert result[-1] == 0
