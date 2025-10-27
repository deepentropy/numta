"""
Test suite for TYPPRICE
"""

import numpy as np

from numta import TYPPRICE


class TestTYPPRICE:
    """Tests for TYPPRICE"""

    def test_typprice_basic(self):
        """Test basic TYPPRICE calculation"""
        high = np.array([110, 112, 114, 113, 115], dtype=np.float64)
        low = np.array([100, 102, 104, 103, 105], dtype=np.float64)
        close = np.array([105, 107, 109, 108, 110], dtype=np.float64)

        result = TYPPRICE(high, low, close)

        assert len(result) == len(close)
        # No NaN values
        assert not np.any(np.isnan(result))

        # Verify first value: (110 + 100 + 105) / 3 = 105
        assert abs(result[0] - 105.0) < 0.01

    def test_typprice_between_high_low(self):
        """Test that TYPPRICE is between high and low"""
        high = np.array([110, 112, 114, 113, 115], dtype=np.float64)
        low = np.array([100, 102, 104, 103, 105], dtype=np.float64)
        close = np.array([105, 107, 109, 108, 110], dtype=np.float64)

        typprice = TYPPRICE(high, low, close)

        # All typical prices should be between low and high
        assert np.all(typprice >= low)
        assert np.all(typprice <= high)

    def test_typprice_formula(self):
        """Test TYPPRICE formula directly"""
        high = np.array([120.0, 125.0], dtype=np.float64)
        low = np.array([110.0, 115.0], dtype=np.float64)
        close = np.array([115.0, 120.0], dtype=np.float64)

        typprice = TYPPRICE(high, low, close)

        # Verify formula: (H + L + C) / 3
        expected = (high + low + close) / 3.0
        np.testing.assert_array_almost_equal(typprice, expected)
