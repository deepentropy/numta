"""
Test suite for SAREXT
"""

import numpy as np

from numta import SAREXT


class TestSAREXT:
    """Tests for SAREXT (Parabolic SAR Extended)"""

    def test_sarext_basic(self):
        """Test basic SAREXT calculation"""
        high = np.array([110, 112, 114, 113, 115, 117, 116, 118, 120], dtype=np.float64)
        low = np.array([100, 102, 104, 103, 105, 107, 106, 108, 110], dtype=np.float64)
        result = SAREXT(high, low)

        # Should have no NaN values
        assert not np.any(np.isnan(result))
        assert len(result) == len(high)

    def test_sarext_asymmetric(self):
        """Test SAREXT with asymmetric parameters"""
        high = np.array([110, 112, 114, 113, 115, 117, 116, 118, 120], dtype=np.float64)
        low = np.array([100, 102, 104, 103, 105, 107, 106, 108, 110], dtype=np.float64)

        result = SAREXT(high, low,
                       accelerationinit_long=0.01,
                       accelerationlong=0.01,
                       accelerationmax_long=0.1,
                       accelerationinit_short=0.03,
                       accelerationshort=0.03,
                       accelerationmax_short=0.3)

        assert len(result) == len(high)
