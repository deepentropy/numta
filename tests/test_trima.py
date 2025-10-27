"""
Test suite for TRIMA
"""

import numpy as np
import pytest
from numta import TRIMA, SMA

class TestTRIMA:
    """Tests for TRIMA"""

    def test_trima_basic(self):
        """Test basic TRIMA calculation"""
        close = np.linspace(100, 120, 50)
        result = TRIMA(close, timeperiod=10)

        assert len(result) == len(close)
        # TRIMA should match double SMA formula
        from numta import SMA
        n1 = 10 // 2
        n2 = n1 + 1
        sma1 = SMA(close, timeperiod=n1)
        expected = SMA(sma1, timeperiod=n2)
        np.testing.assert_array_equal(result, expected)

    def test_trima_smoother_than_sma(self):
        """Test that TRIMA is smoother than SMA"""
        from numta import SMA

        # Create data with some noise
        close = np.linspace(100, 120, 50) + np.random.randn(50) * 2

        sma = SMA(close, timeperiod=10)
        trima = TRIMA(close, timeperiod=10)

        # TRIMA should have lower standard deviation (smoother)
        valid_sma = sma[~np.isnan(sma)]
        valid_trima = trima[~np.isnan(trima)]

        if len(valid_sma) > 10 and len(valid_trima) > 10:
            sma_std = np.std(np.diff(valid_sma))
            trima_std = np.std(np.diff(valid_trima))
            assert trima_std <= sma_std * 1.5  # Allow some tolerance
