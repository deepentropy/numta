"""
Test suite for TRANGE
"""

import numpy as np
import pytest
from talib_pure import TRANGE

class TestTRANGE:
    """Tests for TRANGE"""

    def test_trange_basic(self):
        """Test basic TRANGE calculation"""
        high = np.array([110, 112, 114, 113, 115], dtype=np.float64)
        low = np.array([100, 102, 104, 103, 105], dtype=np.float64)
        close = np.array([105, 107, 109, 108, 110], dtype=np.float64)

        result = TRANGE(high, low, close)

        # No NaN values
        assert not np.any(np.isnan(result))
        # All values should be positive
        assert np.all(result >= 0)
        # First value is just high - low
        assert abs(result[0] - 10.0) < 0.01

    def test_trange_with_gaps(self):
        """Test TRANGE with price gaps"""
        high = np.array([110, 105, 115], dtype=np.float64)
        low = np.array([100, 95, 105], dtype=np.float64)
        close = np.array([105, 100, 110], dtype=np.float64)

        result = TRANGE(high, low, close)

        # TRANGE should capture gaps
        # Index 1: max(105-95, |105-105|, |95-105|) = max(10, 0, 10) = 10
        assert abs(result[1] - 10.0) < 0.01
