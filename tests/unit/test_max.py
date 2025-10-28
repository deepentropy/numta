"""
Test suite for MAX
"""

import numpy as np
import pytest

def test_max_basic():
    """Test MAX basic calculation"""
    from numta import MAX

    data = np.array([100, 105, 103, 108, 106, 110, 107, 109, 104, 111])
    result = MAX(data, timeperiod=5)

    assert np.all(np.isnan(result[:4]))
    assert result[4] == 108  # max of [100, 105, 103, 108, 106]
    assert result[-1] == 111  # max of last 5 values


def test_max_empty_input():
    """Test MAX with empty array"""
    from numta import MAX

    empty = np.array([])
    result = MAX(empty)
    assert len(result) == 0


def test_max_input_validation():
    """Test MAX validates input"""
    from numta import MAX

    data = np.array([100, 105])
    with pytest.raises(ValueError):
        MAX(data, timeperiod=1)
