"""
Test suite for MAMA
"""

import numpy as np


def test_mama_basic():
    """Test MAMA basic calculation"""
    from numta import MAMA

    close = np.linspace(100, 150, 50)
    mama, fama = MAMA(close, fastlimit=0.5, slowlimit=0.05)

    assert len(mama) == len(close)
    assert len(fama) == len(close)
    assert not np.isnan(mama[-1])
    assert not np.isnan(fama[-1])


def test_mama_empty_input():
    """Test MAMA with empty array"""
    from numta import MAMA

    empty = np.array([])
    mama, fama = MAMA(empty)
    assert len(mama) == 0
    assert len(fama) == 0
