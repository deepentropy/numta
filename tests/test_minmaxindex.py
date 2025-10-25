"""
Test suite for MINMAXINDEX
"""

import numpy as np
import pytest
from talib_pure import MINMAXINDEX

class TestMINMAXINDEX:
    """Tests for MINMAXINDEX (Indexes of lowest and highest values)"""

    def test_minmaxindex_basic(self):
        """Test basic MINMAXINDEX calculation"""
        data = np.array([100, 95, 108, 98, 106], dtype=np.float64)
        minidx_result, maxidx_result = MINMAXINDEX(data, timeperiod=5)

        # First 4 values should be NaN
        assert np.isnan(minidx_result[:4]).all()
        assert np.isnan(maxidx_result[:4]).all()

        # Index 4: min is 95 at position 1 (distance=3), max is 108 at position 2 (distance=2)
        assert abs(minidx_result[4] - 3.0) < 0.01
        assert abs(maxidx_result[4] - 2.0) < 0.01

    def test_minmaxindex_current_extreme(self):
        """Test MINMAXINDEX when current bar is extreme"""
        data = np.array([100, 95, 103, 98, 110], dtype=np.float64)
        minidx_result, maxidx_result = MINMAXINDEX(data, timeperiod=5)

        # Index 4: max is 110 at current position (distance=0)
        assert abs(maxidx_result[4] - 0.0) < 0.01
