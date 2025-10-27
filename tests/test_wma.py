"""
Test suite for WMA
"""

import numpy as np

from numta import WMA


class TestWMA:
    """Tests for WMA"""

    def test_wma_basic(self):
        """Test basic WMA calculation"""
        close = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
        result = WMA(close, timeperiod=5)

        assert len(result) == len(close)
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0

    def test_wma_weights(self):
        """Test WMA weighting scheme"""
        # Simple data to verify weights
        data = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        wma = WMA(data, timeperiod=5)

        # WMA at index 4 should be:
        # (5*5 + 4*4 + 3*3 + 2*2 + 1*1) / (5+4+3+2+1)
        # = (25 + 16 + 9 + 4 + 1) / 15 = 55/15 = 3.666...
        expected = (5*5 + 4*4 + 3*3 + 2*2 + 1*1) / 15.0
        assert abs(wma[4] - expected) < 0.01

    def test_wma_vs_sma(self):
        """Test that WMA is more responsive than SMA"""
        from numta import SMA

        # Create data with sudden jump
        close = np.concatenate([
            np.full(20, 100.0),
            np.full(20, 120.0)
        ])

        sma = SMA(close, timeperiod=10)
        wma = WMA(close, timeperiod=10)

        # After the jump, WMA should be higher (closer to 120) than SMA
        if not np.isnan(wma[25]) and not np.isnan(sma[25]):
            # WMA should react faster
            assert wma[25] >= sma[25]

    def test_wma_linear_trend(self):
        """Test WMA with linear trend"""
        # Linear increasing data
        close = np.linspace(100, 120, 50)
        wma = WMA(close, timeperiod=10)

        valid_wma = wma[~np.isnan(wma)]

        # WMA should be mostly increasing for uptrend
        if len(valid_wma) > 10:
            diffs = np.diff(valid_wma)
            positive_diffs = np.sum(diffs > 0)
            # Most differences should be positive
            assert positive_diffs > len(diffs) * 0.8
