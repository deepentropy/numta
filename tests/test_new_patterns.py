"""
Test suite for CDLMORNINGDOJISTAR, CDLMORNINGSTAR, CDLONNECK, and CDLPIERCING
"""

import numpy as np
import pytest


# ==================== CDLMORNINGDOJISTAR Tests ====================

def test_cdlmorningdojistar_pattern():
    """Test Morning Doji Star pattern detection"""
    from talib_pure import CDLMORNINGDOJISTAR

    # Build context + pattern
    open_ = np.array([90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                      110, 99.95, 111])  # Long black, doji gap down, white
    high = np.array([91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                     110, 100, 115])
    low = np.array([90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                    100, 99, 110])
    close = np.array([91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                      100, 100, 114])  # Black, doji (gap down), white penetrating

    result = CDLMORNINGDOJISTAR(open_, high, low, close, penetration=0.3)
    assert result[-1] == 100, f"Expected 100, got {result[-1]}"


def test_cdlmorningdojistar_no_pattern():
    """Test no pattern when conditions not met"""
    from talib_pure import CDLMORNINGDOJISTAR

    # No doji in middle
    open_ = np.array([90, 91, 110, 105, 111])
    high = np.array([91, 92, 110, 106, 115])
    low = np.array([90, 91, 100, 104, 110])
    close = np.array([91, 92, 100, 105.5, 114])  # Second candle not a doji

    result = CDLMORNINGDOJISTAR(open_, high, low, close)
    assert result[-1] == 0


# ==================== CDLMORNINGSTAR Tests ====================

def test_cdlmorningstar_pattern():
    """Test Morning Star pattern detection"""
    from talib_pure import CDLMORNINGSTAR

    # Build context + pattern
    open_ = np.array([90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                      110, 99, 111])  # Long black, small gap down, white
    high = np.array([91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                     110, 99.5, 115])
    low = np.array([90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                    100, 98.5, 110])
    close = np.array([91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                      100, 98.8, 114])  # Black, small (gap down), white penetrating

    result = CDLMORNINGSTAR(open_, high, low, close, penetration=0.3)
    assert result[-1] == 100, f"Expected 100, got {result[-1]}"


def test_cdlmorningstar_no_gap():
    """Test no pattern without gap"""
    from talib_pure import CDLMORNINGSTAR

    open_ = np.array([90, 91, 110, 100.5, 111])  # No gap
    high = np.array([91, 92, 110, 101, 115])
    low = np.array([90, 91, 100, 100, 110])
    close = np.array([91, 92, 100, 100.5, 114])

    result = CDLMORNINGSTAR(open_, high, low, close)
    assert result[-1] == 0


# ==================== CDLONNECK Tests ====================

def test_cdlonneck_pattern():
    """Test On-Neck pattern detection"""
    from talib_pure import CDLONNECK

    # Build context + pattern
    open_ = np.array([90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                      110, 94.5])  # Long black, white at prior low
    high = np.array([91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                     110, 100.5])
    low = np.array([90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                    100, 94])
    close = np.array([91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                      100, 100])  # Black, white closing at prior low

    result = CDLONNECK(open_, high, low, close)
    assert result[-1] == -100, f"Expected -100, got {result[-1]}"


def test_cdlonneck_no_pattern():
    """Test no pattern when close doesn't match low"""
    from talib_pure import CDLONNECK

    open_ = np.array([90, 91, 110, 95])
    high = np.array([91, 92, 110, 102])
    low = np.array([90, 91, 100, 94])
    close = np.array([91, 92, 100, 101.5])  # Doesn't match prior low

    result = CDLONNECK(open_, high, low, close)
    assert result[-1] == 0


# ==================== CDLPIERCING Tests ====================

def test_cdlpiercing_pattern():
    """Test Piercing Pattern detection"""
    from talib_pure import CDLPIERCING

    # Build context + pattern
    open_ = np.array([90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                      110, 94])  # Long black, long white piercing
    high = np.array([91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                     110, 108])
    low = np.array([90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                    100, 94])
    close = np.array([91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                      100, 106])  # Black, white closing above midpoint (105)

    result = CDLPIERCING(open_, high, low, close)
    assert result[-1] == 100, f"Expected 100, got {result[-1]}"


def test_cdlpiercing_no_pattern_insufficient_penetration():
    """Test no pattern when penetration insufficient"""
    from talib_pure import CDLPIERCING

    open_ = np.array([90, 91, 110, 95])
    high = np.array([91, 92, 110, 103])
    low = np.array([90, 91, 100, 94])
    close = np.array([91, 92, 100, 102])  # Doesn't reach midpoint (105)

    result = CDLPIERCING(open_, high, low, close)
    assert result[-1] == 0


# ==================== Common Tests ====================

def test_all_patterns_empty_input():
    """Test all patterns with empty arrays"""
    from talib_pure import CDLMORNINGDOJISTAR, CDLMORNINGSTAR, CDLONNECK, CDLPIERCING

    empty = np.array([])

    assert len(CDLMORNINGDOJISTAR(empty, empty, empty, empty)) == 0
    assert len(CDLMORNINGSTAR(empty, empty, empty, empty)) == 0
    assert len(CDLONNECK(empty, empty, empty, empty)) == 0
    assert len(CDLPIERCING(empty, empty, empty, empty)) == 0


def test_all_patterns_input_validation():
    """Test all patterns validate input lengths"""
    from talib_pure import CDLMORNINGDOJISTAR, CDLMORNINGSTAR, CDLONNECK, CDLPIERCING

    open_ = np.array([100, 105])
    high = np.array([105, 106])
    low = np.array([100, 100])
    close = np.array([105])  # Wrong length

    with pytest.raises(ValueError, match="same length"):
        CDLMORNINGDOJISTAR(open_, high, low, close)
    with pytest.raises(ValueError, match="same length"):
        CDLMORNINGSTAR(open_, high, low, close)
    with pytest.raises(ValueError, match="same length"):
        CDLONNECK(open_, high, low, close)
    with pytest.raises(ValueError, match="same length"):
        CDLPIERCING(open_, high, low, close)


def test_all_patterns_backend_consistency():
    from talib_pure import (CDLMORNINGDOJISTAR, CDLMORNINGSTAR, CDLONNECK, CDLPIERCING,


    # Generate test data
    n = 50
    np.random.seed(789)
    high = np.random.uniform(100, 200, n)
    low = high - np.random.uniform(1, 10, n)
    open_ = low + np.random.uniform(0, 1, n) * (high - low)
    close = low + np.random.uniform(0, 1, n) * (high - low)

    # Test each pattern
    for pattern_func in [CDLMORNINGDOJISTAR, CDLMORNINGSTAR, CDLONNECK, CDLPIERCING]:
        set_backend("cpu")
        if pattern_func in [CDLMORNINGDOJISTAR, CDLMORNINGSTAR]:
            cpu_result = pattern_func(open_, high, low, close, penetration=0.3)
        else:
            cpu_result = pattern_func(open_, high, low, close)

        if pattern_func in [CDLMORNINGDOJISTAR, CDLMORNINGSTAR]:
        else:


    set_backend("cpu")
