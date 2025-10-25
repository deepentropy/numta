"""
Test suite for MAXINDEX
"""

import numpy as np
import pytest

def test_maxindex_basic():
    """Test MAXINDEX basic calculation"""
    from talib_pure import MAXINDEX

    data = np.array([100, 105, 103, 108, 106, 110, 107, 109, 104, 111])
    result = MAXINDEX(data, timeperiod=5)

    assert np.all(np.isnan(result[:4]))
    assert result[-1] == 0  # Current bar (111) is highest


def test_maxindex_aging_high():
    """Test MAXINDEX with aging high"""
    from talib_pure import MAXINDEX

    data = np.array([100, 105, 103, 110, 106, 107, 108, 109, 108, 107])
    result = MAXINDEX(data, timeperiod=7)

    # The high of 110 at index 3 should show increasing age
    assert result[-1] > 0  # High was some bars ago


def test_maxindex_empty_input():
    """Test MAXINDEX with empty array"""
    from talib_pure import MAXINDEX

    empty = np.array([])
    result = MAXINDEX(empty)
    assert len(result) == 0
