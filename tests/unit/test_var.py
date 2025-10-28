"""
Test suite for VAR
"""

import numpy as np

from numta import VAR


class TestVAR:
    """Tests for VAR"""

    def test_var_basic(self):
        """Test basic VAR calculation"""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
        result = VAR(data, timeperiod=5)

        assert len(result) == len(data)
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0

    def test_var_constant(self):
        """Test VAR with constant values"""
        data = np.full(10, 100.0)
        var = VAR(data, timeperiod=5)

        valid_var = var[~np.isnan(var)]
        # Variance of constant should be 0
        assert np.all(np.abs(valid_var) < 0.01)

    def test_var_relationship_to_stddev(self):
        """Test VAR = STDDEV²"""
        from numta import STDDEV

        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)

        var = VAR(data, timeperiod=5, nbdev=1.0)
        stddev = STDDEV(data, timeperiod=5, nbdev=1.0)

        # VAR should equal STDDEV squared
        expected_var = stddev ** 2

        # Only compare valid values
        valid_mask = ~np.isnan(var)
        np.testing.assert_array_almost_equal(
            var[valid_mask],
            expected_var[valid_mask],
            decimal=10
        )
