"""
Test suite for MININDEX
"""

import numpy as np
import pytest
from talib_pure import MININDEX

class TestMININDEX:
    """Tests for MININDEX (Index of lowest value)"""

    def test_minindex_basic(self):
        """Test basic MININDEX calculation"""
        data = np.array([100, 95, 103, 98, 106], dtype=np.float64)
        result = MININDEX(data, timeperiod=5)

        # First 4 values should be NaN
        assert np.isnan(result[:4]).all()
        # Index 4: min is 95 at position 1, distance = 4-1 = 3
        assert abs(result[4] - 3.0) < 0.01

    def test_minindex_recent_low(self):
        """Test MININDEX with recent low"""
        data = np.array([100, 105, 103, 98, 106], dtype=np.float64)
        result = MININDEX(data, timeperiod=5)

        # Index 4: min is 98 at position 3, distance = 4-3 = 1
        assert abs(result[4] - 1.0) < 0.01
