"""
Test suite for MEDPRICE
"""

import numpy as np
import pytest

def test_medprice_basic():
    """Test MEDPRICE basic calculation"""
    from talib_pure import MEDPRICE

    high = np.array([105, 106, 108, 107, 109])
    low = np.array([100, 101, 103, 102, 104])

    result = MEDPRICE(high, low)

    expected = (high + low) / 2
    np.testing.assert_array_equal(result, expected)


def test_medprice_empty_input():
    """Test MEDPRICE with empty arrays"""
    from talib_pure import MEDPRICE

    empty = np.array([])
    result = MEDPRICE(empty, empty)
    assert len(result) == 0


def test_medprice_input_validation():
    """Test MEDPRICE validates input lengths"""
    from talib_pure import MEDPRICE

    high = np.array([105, 106])
    low = np.array([100])

    with pytest.raises(ValueError, match="same length"):
        MEDPRICE(high, low)
