"""
Test suite for SUM
"""

import numpy as np

from numta import SUM


class TestSUM:
    """Tests for SUM"""

    def test_sum_basic(self):
        """Test basic SUM calculation"""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
        result = SUM(data, timeperiod=5)

        assert np.isnan(result[:4]).all()
        # Index 4: 1+2+3+4+5 = 15
        assert abs(result[4] - 15.0) < 0.01
        # Index 9: 6+7+8+9+10 = 40
        assert abs(result[9] - 40.0) < 0.01

    def test_sum_relationship_to_sma(self):
        """Test SUM = SMA * timeperiod"""
        from numta import SMA

        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
        sum_result = SUM(data, timeperiod=5)
        sma_result = SMA(data, timeperiod=5)

        # SUM = SMA * timeperiod
        expected = sma_result * 5.0
        np.testing.assert_array_almost_equal(sum_result, expected)
