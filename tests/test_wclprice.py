"""
Test suite for WCLPRICE
"""

import numpy as np

from numta import WCLPRICE


class TestWCLPRICE:
    """Tests for WCLPRICE"""

    def test_wclprice_basic(self):
        """Test basic WCLPRICE calculation"""
        high = np.array([110, 112, 114, 113, 115], dtype=np.float64)
        low = np.array([100, 102, 104, 103, 105], dtype=np.float64)
        close = np.array([105, 107, 109, 108, 110], dtype=np.float64)

        result = WCLPRICE(high, low, close)

        assert len(result) == len(close)
        # No NaN values
        assert not np.any(np.isnan(result))

        # Verify first value: (110 + 100 + 2*105) / 4 = 105
        assert abs(result[0] - 105.0) < 0.01

    def test_wclprice_formula(self):
        """Test WCLPRICE formula directly"""
        high = np.array([120.0, 125.0], dtype=np.float64)
        low = np.array([110.0, 115.0], dtype=np.float64)
        close = np.array([115.0, 120.0], dtype=np.float64)

        wclprice = WCLPRICE(high, low, close)

        # Verify formula: (H + L + 2*C) / 4
        expected = (high + low + 2.0 * close) / 4.0
        np.testing.assert_array_almost_equal(wclprice, expected)

    def test_wclprice_close_emphasis(self):
        """Test that WCLPRICE emphasizes close"""
        from numta import TYPPRICE

        high = np.array([110, 112, 114], dtype=np.float64)
        low = np.array([100, 102, 104], dtype=np.float64)
        close = np.array([105, 107, 109], dtype=np.float64)

        wclprice = WCLPRICE(high, low, close)
        typprice = TYPPRICE(high, low, close)

        # WCLPRICE should be closer to close than TYPPRICE
        # because close is weighted 2x in WCLPRICE
        for i in range(len(close)):
            wc_dist = abs(wclprice[i] - close[i])
            typ_dist = abs(typprice[i] - close[i])
            assert wc_dist <= typ_dist
