"""
Test suite for MINMAX
"""

import numpy as np

from numta import MINMAX


class TestMINMAX:
    """Tests for MINMAX (Lowest and highest values)"""

    def test_minmax_basic(self):
        """Test basic MINMAX calculation"""
        data = np.array([100, 95, 103, 98, 106, 92, 104], dtype=np.float64)
        min_result, max_result = MINMAX(data, timeperiod=5)

        # First 4 values should be NaN
        assert np.isnan(min_result[:4]).all()
        assert np.isnan(max_result[:4]).all()

        # Index 4: min=95, max=106
        assert abs(min_result[4] - 95.0) < 0.01
        assert abs(max_result[4] - 106.0) < 0.01

        # Index 5: min=92, max=106
        assert abs(min_result[5] - 92.0) < 0.01
        assert abs(max_result[5] - 106.0) < 0.01

    def test_minmax_returns_tuple(self):
        """Test that MINMAX returns a tuple"""
        data = np.array([100, 95, 103, 98, 106], dtype=np.float64)
        result = MINMAX(data, timeperiod=3)
        assert isinstance(result, tuple)
        assert len(result) == 2
