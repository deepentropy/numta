"""
Test suite for STOCH
"""

import numpy as np
import pytest
from talib_pure import STOCH, STOCHF

class TestSTOCH:
    """Tests for STOCH (Stochastic)"""

    def test_stoch_basic(self):
        """Test basic STOCH calculation"""
        high = np.array([110, 112, 114, 113, 115, 117, 116, 118, 120], dtype=np.float64)
        low = np.array([100, 102, 104, 103, 105, 107, 106, 108, 110], dtype=np.float64)
        close = np.array([105, 107, 109, 108, 110, 112, 111, 113, 115], dtype=np.float64)

        slowk, slowd = STOCH(high, low, close, fastk_period=5, slowk_period=3, slowd_period=3)

        # Should return two arrays
        assert len(slowk) == len(high)
        assert len(slowd) == len(high)

        # Values should be between 0 and 100
        valid_k = slowk[~np.isnan(slowk)]
        valid_d = slowd[~np.isnan(slowd)]

        assert np.all(valid_k >= 0)
        assert np.all(valid_k <= 100)
        assert np.all(valid_d >= 0)
        assert np.all(valid_d <= 100)

    def test_stoch_smoother_than_stochf(self):
        """Test that STOCH is smoother than STOCHF"""
        np.random.seed(42)
        high = np.abs(np.cumsum(np.random.randn(50))) + 110
        low = high - 10
        close = (high + low) / 2 + np.random.randn(50) * 2

        fastk, _ = STOCHF(high, low, close, fastk_period=5, fastd_period=3)
        slowk, _ = STOCH(high, low, close, fastk_period=5, slowk_period=3, slowd_period=3)

        # Calculate standard deviation (measure of smoothness)
        # Slow should have lower std dev (smoother)
        valid_fast = fastk[~np.isnan(fastk)]
        valid_slow = slowk[~np.isnan(slowk)]

        if len(valid_fast) > 0 and len(valid_slow) > 0:
            # Take common length
            min_len = min(len(valid_fast), len(valid_slow))
            std_fast = np.std(valid_fast[-min_len:])
            std_slow = np.std(valid_slow[-min_len:])

            # Slow should generally be smoother (lower std)
            # Allow some tolerance
            assert std_slow <= std_fast * 1.5
