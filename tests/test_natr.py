"""
Test suite for NATR
"""

import numpy as np
import pytest
from talib_pure import NATR

class TestNATR:
    """Tests for NATR (Normalized Average True Range)"""

    def test_natr_basic(self):
        """Test basic NATR calculation"""
        high = np.array([105, 106, 108, 107, 109, 110, 111, 112, 113, 114,
                        115, 116, 117, 118, 119], dtype=np.float64)
        low = np.array([100, 101, 103, 102, 104, 105, 106, 107, 108, 109,
                       110, 111, 112, 113, 114], dtype=np.float64)
        close = np.array([103, 104, 106, 105, 107, 108, 109, 110, 111, 112,
                         113, 114, 115, 116, 117], dtype=np.float64)
        result = NATR(high, low, close, timeperiod=14)

        # Should have NaN at start (ATR lookback)
        assert np.any(np.isnan(result[:13]))
        # Values should be positive percentages
        assert np.all(result[~np.isnan(result)] > 0)

    def test_natr_percentage(self):
        """Test that NATR is expressed as percentage"""
        high = np.array([110, 112, 111, 113, 115, 114, 116, 118, 117, 119,
                        120, 122, 121, 123, 125], dtype=np.float64)
        low = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
                       110, 112, 111, 113, 115], dtype=np.float64)
        close = np.array([105, 107, 106, 108, 110, 109, 111, 113, 112, 114,
                         115, 117, 116, 118, 120], dtype=np.float64)
        result = NATR(high, low, close, timeperiod=14)

        # NATR should be reasonable percentage (typically 0-10% for most stocks)
        valid_values = result[~np.isnan(result)]
        assert np.all(valid_values < 50)  # Should be reasonable percentage
