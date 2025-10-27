"""
Test suite for HT_DCPHASE
"""

import numpy as np
import pytest

def test_ht_dcphase_basic():
    """Test HT_DCPHASE basic calculation"""
    from numta import HT_DCPHASE

    # Create cyclic data
    n = 100
    close = np.sin(np.linspace(0, 4*np.pi, n)) * 10 + 100

    result = HT_DCPHASE(close)

    # First 32 values should be NaN
    assert np.all(np.isnan(result[:32]))
    # Phase values should be between 0 and 360
    valid_values = result[~np.isnan(result)]
    assert np.all(valid_values >= 0)
    assert np.all(valid_values <= 360)


def test_ht_dcphase_empty_input():
    """Test HT_DCPHASE with empty array"""
    from numta import HT_DCPHASE

    empty = np.array([])
    result = HT_DCPHASE(empty)
    assert len(result) == 0
