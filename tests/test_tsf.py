"""
Test suite for TSF
"""

import numpy as np
import pytest
from numta import TSF, SMA

class TestTSF:
    """Tests for TSF"""

    def test_tsf_basic(self):
        """Test basic TSF calculation"""
        close = np.linspace(100, 120, 50)
        result = TSF(close, timeperiod=14)

        assert len(result) == len(close)
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0

    def test_tsf_linear_trend(self):
        """Test TSF with perfect linear trend"""
        close = np.linspace(100, 150, 50)
        tsf = TSF(close, timeperiod=10)

        valid_tsf = tsf[~np.isnan(tsf)]

        # TSF should forecast ahead of current price in uptrend
        if len(valid_tsf) > 5:
            # Last TSF value should be higher than current price
            assert valid_tsf[-1] > close[len(close) - 2]

    def test_tsf_faster_than_sma(self):
        """Test that TSF reacts faster than SMA"""
        from numta import SMA

        # Create data with sudden jump
        close = np.concatenate([
            np.full(50, 100.0),
            np.full(50, 120.0)
        ])

        sma = SMA(close, timeperiod=14)
        tsf = TSF(close, timeperiod=14)

        # After the jump, TSF should be closer to new level
        if not np.isnan(tsf[70]) and not np.isnan(sma[70]):
            # TSF should be higher (closer to 120) than SMA
            assert tsf[70] >= sma[70]
