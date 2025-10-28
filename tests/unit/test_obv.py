"""
Test suite for OBV
"""

import numpy as np
import pytest
from numta import OBV

class TestOBV:
    """Tests for OBV (On Balance Volume)"""

    def test_obv_basic(self):
        """Test basic OBV calculation"""
        close = np.array([100, 102, 101, 103, 105], dtype=np.float64)
        volume = np.array([1000, 1500, 1200, 1800, 2000], dtype=np.float64)
        result = OBV(close, volume)

        # Index 0: 0
        assert result[0] == 1000.0
        # Index 1: 1000 + 1500 = 2500 (price up)
        assert result[1] == 2500.0
        # Index 2: 2500 - 1200 = 1300 (price down)
        assert result[2] == 1300.0
        # Index 3: 1300 + 1800 = 3100 (price up)
        assert result[3] == 3100.0
        # Index 4: 3100 + 2000 = 5100 (price up)
        assert result[4] == 5100.0

    def test_obv_flat(self):
        """Test OBV when price doesn't change"""
        close = np.array([100, 100, 100, 100], dtype=np.float64)
        volume = np.array([1000, 1500, 1200, 1800], dtype=np.float64)
        result = OBV(close, volume)

        # OBV should remain at 0 when price is flat
        assert result[0] == 1000.0
        assert result[1] == 1000.0
        assert result[2] == 1000.0
        assert result[3] == 1000.0
    def test_obv_downtrend(self):
        """Test OBV in downtrend"""
        close = np.array([105, 104, 103, 102, 101], dtype=np.float64)
        volume = np.array([1000, 1500, 1200, 1800, 2000], dtype=np.float64)
        result = OBV(close, volume)

        # OBV should be negative (declining)
        assert result[-1] < 0

    def test_obv_mismatched_arrays(self):
        """Test OBV with mismatched array lengths"""
        close = np.array([100, 102, 104], dtype=np.float64)
        volume = np.array([1000, 1500], dtype=np.float64)
        with pytest.raises(ValueError):
            OBV(close, volume)
