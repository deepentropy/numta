"""
Test suite for MIDPOINT
"""

import numpy as np
import pytest

def test_midpoint_basic():
    """Test MIDPOINT basic calculation"""
    from numta import MIDPOINT

    close = np.array([100, 105, 103, 108, 106, 110, 107, 109, 104, 111, 113])
    result = MIDPOINT(close, timeperiod=5)

    assert np.all(np.isnan(result[:4]))
    assert not np.isnan(result[-1])


def test_midpoint_calculation():
    """Test MIDPOINT calculates correctly"""
    from numta import MIDPOINT

    # Simple test case
    data = np.array([100, 110, 105, 115, 100])
    result = MIDPOINT(data, timeperiod=5)

    # MIDPOINT should be (max + min) / 2 = (115 + 100) / 2 = 107.5
    assert result[-1] == 107.5


def test_midpoint_empty_input():
    """Test MIDPOINT with empty array"""
    from numta import MIDPOINT

    empty = np.array([])
    result = MIDPOINT(empty)
    assert len(result) == 0


def test_midpoint_input_validation():
    """Test MIDPOINT validates input"""
    from numta import MIDPOINT

    data = np.array([100, 105])
    with pytest.raises(ValueError):
        MIDPOINT(data, timeperiod=1)
