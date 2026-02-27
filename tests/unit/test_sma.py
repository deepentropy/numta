"""
Test suite for SMA
"""

import numpy as np
import pytest
from numta import SMA

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

    def test_sma_insufficient_data(self):
        """Test SMA returns all NaN when input length < timeperiod (issue #30)"""
        data = np.array([1.0, 2.0, 3.0])
        result = SMA(data, timeperiod=20)
        assert len(result) == len(data)
        assert np.isnan(result).all()

    def test_sma_insufficient_data_off_by_one(self):
        """Test SMA when input length == timeperiod - 1"""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        result = SMA(data, timeperiod=5)
        assert len(result) == len(data)
        assert np.isnan(result).all()

    def test_sma_exact_length(self):
        """Test SMA when input length == timeperiod"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = SMA(data, timeperiod=5)
        assert len(result) == len(data)
        assert np.isnan(result[:4]).all()
        assert abs(result[4] - 3.0) < 0.01

    def test_sma_single_element(self):
        """Test SMA with single element array"""
        data = np.array([42.0])
        result = SMA(data, timeperiod=5)
        assert len(result) == 1
        assert np.isnan(result[0])

    def test_sma_empty(self):
        """Test SMA with empty array"""
        data = np.array([], dtype=np.float64)
        result = SMA(data, timeperiod=5)
        assert len(result) == 0

    def test_sma_invalid_timeperiod(self):
        """Test SMA with invalid timeperiod"""
        data = np.array([1, 2, 3], dtype=np.float64)
        with pytest.raises(ValueError):
            SMA(data, timeperiod=1)
