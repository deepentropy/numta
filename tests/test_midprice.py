"""
Test suite for MIDPRICE
"""

import numpy as np
import pytest
from talib_pure import MIDPRICE

class TestMIDPRICE:
    """Tests for MIDPRICE (Midpoint Price over period)"""

    def test_midprice_basic(self):
        """Test basic MIDPRICE calculation"""
        high = np.array([105, 106, 108, 107, 109, 110, 111, 112], dtype=np.float64)
        low = np.array([100, 101, 103, 102, 104, 105, 106, 107], dtype=np.float64)
        result = MIDPRICE(high, low, timeperiod=5)

        # First 4 values should be NaN (lookback = timeperiod - 1)
        assert np.isnan(result[:4]).all()
        # Values should be between low and high
        assert np.all(result[4:] >= 101.0)
        assert np.all(result[4:] <= 111.0)

    def test_midprice_calculation(self):
        """Test MIDPRICE calculation accuracy"""
        high = np.array([110, 112, 108, 115, 113], dtype=np.float64)
        low = np.array([100, 102, 98, 105, 103], dtype=np.float64)
        result = MIDPRICE(high, low, timeperiod=3)

        # At index 2: MAX(110,112,108)=112, MIN(100,102,98)=98, MIDPRICE=(112+98)/2=105
        assert abs(result[2] - 105.0) < 0.01
        # At index 3: MAX(112,108,115)=115, MIN(102,98,105)=98, MIDPRICE=(115+98)/2=106.5
        assert abs(result[3] - 106.5) < 0.01

    def test_midprice_invalid_timeperiod(self):
        """Test MIDPRICE with invalid timeperiod"""
        high = np.array([105, 106, 108], dtype=np.float64)
        low = np.array([100, 101, 103], dtype=np.float64)
        with pytest.raises(ValueError):
            MIDPRICE(high, low, timeperiod=1)

    def test_midprice_mismatched_arrays(self):
        """Test MIDPRICE with mismatched array lengths"""
        high = np.array([105, 106, 108], dtype=np.float64)
        low = np.array([100, 101], dtype=np.float64)
        with pytest.raises(ValueError):
            MIDPRICE(high, low, timeperiod=3)
