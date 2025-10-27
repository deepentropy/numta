"""
Test suite for MIN
"""

import numpy as np
import pytest
from numta import MIN

class TestMIN:
    """Tests for MIN (Lowest value over period)"""

    def test_min_basic(self):
        """Test basic MIN calculation"""
        data = np.array([100, 95, 103, 98, 106, 92, 104], dtype=np.float64)
        result = MIN(data, timeperiod=5)

        # First 4 values should be NaN
        assert np.isnan(result[:4]).all()
        # Index 4: min(100,95,103,98,106) = 95
        assert abs(result[4] - 95.0) < 0.01
        # Index 5: min(95,103,98,106,92) = 92
        assert abs(result[5] - 92.0) < 0.01

    def test_min_decreasing(self):
        """Test MIN with decreasing values"""
        data = np.array([100, 90, 80, 70, 60], dtype=np.float64)
        result = MIN(data, timeperiod=3)

        assert abs(result[2] - 80.0) < 0.01
        assert abs(result[3] - 70.0) < 0.01
        assert abs(result[4] - 60.0) < 0.01
