"""
Test suite for SMA
"""

import numpy as np
import pytest
from talib_pure import SMA

class TestSMA:
    """Tests for SMA (Simple Moving Average)"""

    def test_sma_basic(self):
        """Test basic SMA calculation"""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
        result = SMA(data, timeperiod=5)

        # First 4 values should be NaN
        assert np.isnan(result[:4]).all()
        # Index 4: (1+2+3+4+5)/5 = 3.0
        assert abs(result[4] - 3.0) < 0.01
        # Index 9: (6+7+8+9+10)/5 = 8.0
        assert abs(result[9] - 8.0) < 0.01

    def test_sma_invalid_timeperiod(self):
        """Test SMA with invalid timeperiod"""
        data = np.array([1, 2, 3], dtype=np.float64)
        with pytest.raises(ValueError):
            SMA(data, timeperiod=1)
