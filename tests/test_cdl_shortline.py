"""
Test suite for CDLSHORTLINE
"""

import numpy as np
import pytest

def test_cdlshortline_white():
    """Test Short Line white candle detection"""
    from talib_pure import CDLSHORTLINE

    # Build context with larger candles + short line pattern (need shadows in context)
    open_ = np.array([90, 100, 92, 102, 94, 104, 96, 106, 98, 108,
                      100])  # Short body
    high = np.array([92, 106, 94, 108, 96, 110, 98, 112, 100, 115,
                     100.2])  # Context has shadows
    low = np.array([89, 99, 91, 101, 93, 103, 95, 105, 97, 107,
                    99.95])  # Context has shadows
    close = np.array([91, 105, 93, 107, 95, 109, 97, 111, 99, 113,
                      100.1])  # Short white candle (body=0.1, shadows=0.1/0.05)

    result = CDLSHORTLINE(open_, high, low, close)
    assert result[-1] == 100, f"Expected 100, got {result[-1]}"


def test_cdlshortline_black():
    """Test Short Line black candle detection"""
    from talib_pure import CDLSHORTLINE

    open_ = np.array([90, 100, 92, 102, 94, 104, 96, 106, 98, 108,
                      100.1])  # Short body
    high = np.array([92, 106, 94, 108, 96, 110, 98, 112, 100, 115,
                     100.2])
    low = np.array([89, 99, 91, 101, 93, 103, 95, 105, 97, 107,
                    99.95])
    close = np.array([91, 105, 93, 107, 95, 109, 97, 111, 99, 113,
                      100])  # Short black candle (body=0.1)

    result = CDLSHORTLINE(open_, high, low, close)
    assert result[-1] == -100, f"Expected -100, got {result[-1]}"


def test_cdlshortline_no_pattern():
    """Test no pattern when candle is long"""
    from talib_pure import CDLSHORTLINE

    open_ = np.array([90, 91, 100])  # Long candle
    high = np.array([91, 92, 120])
    low = np.array([90, 91, 100])
    close = np.array([91, 92, 120])

    result = CDLSHORTLINE(open_, high, low, close)
    assert result[-1] == 0
