"""
Test suite for STDDEV
"""

import numpy as np
import pytest
from numta import STDDEV

class TestSTDDEV:
    """Tests for STDDEV (Standard Deviation)"""

    def test_stddev_basic(self):
        """Test basic STDDEV calculation"""
        data = np.array([2, 4, 4, 4, 5, 5, 7, 9], dtype=np.float64)
        result = STDDEV(data, timeperiod=5, nbdev=1.0)

        # First 4 values should be NaN
        assert np.isnan(result[:4]).all()
        # Values should be positive
        assert np.all(result[~np.isnan(result)] >= 0)

    def test_stddev_constant(self):
        """Test STDDEV with constant values"""
        data = np.array([5, 5, 5, 5, 5, 5], dtype=np.float64)
        result = STDDEV(data, timeperiod=3, nbdev=1.0)

        # Std dev of constant values should be 0
        valid_values = result[~np.isnan(result)]
        assert np.all(np.abs(valid_values) < 0.01)

    def test_stddev_nbdev(self):
        """Test STDDEV with different nbdev"""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)

        stddev1 = STDDEV(data, timeperiod=5, nbdev=1.0)
        stddev2 = STDDEV(data, timeperiod=5, nbdev=2.0)

        # stddev2 should be exactly 2 * stddev1
        np.testing.assert_array_almost_equal(stddev2, stddev1 * 2.0)
