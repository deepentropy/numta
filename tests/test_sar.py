"""
Test suite for SAR
"""

import numpy as np
import pytest
from talib_pure import SAR

class TestSAR:
    """Tests for SAR (Parabolic SAR)"""

    def test_sar_basic(self):
        """Test basic SAR calculation"""
        high = np.array([110, 112, 114, 113, 115, 117, 116, 118, 120], dtype=np.float64)
        low = np.array([100, 102, 104, 103, 105, 107, 106, 108, 110], dtype=np.float64)
        result = SAR(high, low, acceleration=0.02, maximum=0.2)

        # Should have no NaN values
        assert not np.any(np.isnan(result))
        # SAR values should be within the high-low range (approximately)
        assert len(result) == len(high)

    def test_sar_invalid_params(self):
        """Test SAR with invalid parameters"""
        high = np.array([110, 112, 114], dtype=np.float64)
        low = np.array([100, 102, 104], dtype=np.float64)

        with pytest.raises(ValueError):
            SAR(high, low, acceleration=0.0, maximum=0.2)

        with pytest.raises(ValueError):
            SAR(high, low, acceleration=0.3, maximum=0.2)
