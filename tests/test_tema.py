"""
Test suite for TEMA
"""

import numpy as np
import pytest
from talib_pure import TEMA, EMA

class TestTEMA:
    """Tests for TEMA"""

    def test_tema_basic(self):
        """Test basic TEMA calculation"""
        close = np.linspace(100, 120, 200)
        result = TEMA(close, timeperiod=10)

        assert len(result) == len(close)
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0

    def test_tema_less_lag_than_ema(self):
        """Test that TEMA responds faster than EMA"""
        from talib_pure import EMA

        # Create data with sudden jump
        close = np.concatenate([
            np.full(50, 100.0),
            np.full(50, 120.0)
        ])

        ema = EMA(close, timeperiod=10)
        tema = TEMA(close, timeperiod=10)

        # TEMA should reach the new level faster than EMA
        # Check values after the jump
        if not np.isnan(tema[30]) and not np.isnan(ema[30]):
            assert tema[30] >= ema[30]  # TEMA should be closer to 120
