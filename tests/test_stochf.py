"""
Test suite for STOCHF
"""

import numpy as np
import pytest
from talib_pure import STOCHF

class TestSTOCHF:
    """Tests for STOCHF (Stochastic Fast)"""

    def test_stochf_basic(self):
        """Test basic STOCHF calculation"""
        high = np.array([110, 112, 114, 113, 115, 117, 116, 118, 120], dtype=np.float64)
        low = np.array([100, 102, 104, 103, 105, 107, 106, 108, 110], dtype=np.float64)
        close = np.array([105, 107, 109, 108, 110, 112, 111, 113, 115], dtype=np.float64)

        fastk, fastd = STOCHF(high, low, close, fastk_period=5, fastd_period=3)

        # Should return two arrays
        assert len(fastk) == len(high)
        assert len(fastd) == len(high)

        # Values should be between 0 and 100
        valid_k = fastk[~np.isnan(fastk)]
        valid_d = fastd[~np.isnan(fastd)]

        assert np.all(valid_k >= 0)
        assert np.all(valid_k <= 100)
        assert np.all(valid_d >= 0)
        assert np.all(valid_d <= 100)

    def test_stochf_extremes(self):
        """Test STOCHF at extreme values"""
        # Close at high (should give 100)
        high = np.array([110, 110, 110, 110, 110], dtype=np.float64)
        low = np.array([100, 100, 100, 100, 100], dtype=np.float64)
        close = np.array([110, 110, 110, 110, 110], dtype=np.float64)

        fastk, _ = STOCHF(high, low, close, fastk_period=3, fastd_period=3)

        valid_k = fastk[~np.isnan(fastk)]
        assert np.all(np.abs(valid_k - 100.0) < 0.01)
