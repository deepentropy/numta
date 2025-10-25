"""
Test suite for CORREL
"""

import numpy as np
import pytest

def test_correl_perfect_positive():
    """Test CORREL with perfect positive correlation"""
    from talib_pure import CORREL

    # High and low perfectly correlated
    high = np.array([105, 106, 107, 108, 109, 110, 111, 112, 113, 114])
    low = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])

    result = CORREL(high, low, timeperiod=5)

    # First 4 values should be NaN (timeperiod - 1)
    assert np.all(np.isnan(result[:4]))
    # Perfect positive correlation should give ~1.0
    assert abs(result[-1] - 1.0) < 0.01, f"Expected ~1.0, got {result[-1]}"


def test_correl_no_correlation():
    """Test CORREL with no correlation"""
    from talib_pure import CORREL

    # Random, uncorrelated data
    np.random.seed(42)
    high = np.random.uniform(100, 110, 50)
    low = np.random.uniform(90, 100, 50)

    result = CORREL(high, low, timeperiod=30)

    # Correlation should be between -1 and 1
    assert -1.0 <= result[-1] <= 1.0


def test_correl_empty_input():
    """Test CORREL with empty arrays"""
    from talib_pure import CORREL

    empty = np.array([])
    result = CORREL(empty, empty)
    assert len(result) == 0


def test_correl_input_validation():
    """Test CORREL validates input lengths"""
    from talib_pure import CORREL

    high = np.array([105, 106])
    low = np.array([100])  # Wrong length

    with pytest.raises(ValueError, match="same length"):
        CORREL(high, low)
