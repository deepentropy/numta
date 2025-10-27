"""
Test suite for T3
"""

import numpy as np

from numta import T3


class TestT3:
    """Tests for T3"""

    def test_t3_basic(self):
        """Test basic T3 calculation"""
        close = np.linspace(100, 120, 200)
        result = T3(close, timeperiod=5, vfactor=0.7)

        assert len(result) == len(close)
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0

    def test_t3_vfactor_range(self):
        """Test T3 with different vfactor values"""
        close = np.linspace(100, 120, 200)

        t3_low = T3(close, timeperiod=5, vfactor=0.3)
        t3_mid = T3(close, timeperiod=5, vfactor=0.7)
        t3_high = T3(close, timeperiod=5, vfactor=0.9)

        # All should produce valid outputs
        assert len(t3_low[~np.isnan(t3_low)]) > 0
        assert len(t3_mid[~np.isnan(t3_mid)]) > 0
        assert len(t3_high[~np.isnan(t3_high)]) > 0
