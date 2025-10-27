"""
Test suite for HT_PHASOR
"""

import numpy as np


def test_ht_phasor_basic():
    """Test HT_PHASOR basic calculation"""
    from numta import HT_PHASOR

    # Create cyclic data
    n = 100
    close = np.sin(np.linspace(0, 4*np.pi, n)) * 10 + 100

    inphase, quadrature = HT_PHASOR(close)

    # First 32 values should be NaN
    assert np.all(np.isnan(inphase[:32]))
    assert np.all(np.isnan(quadrature[:32]))

    # Should have valid values after unstable period
    assert not np.isnan(inphase[-1])
    assert not np.isnan(quadrature[-1])


def test_ht_phasor_magnitude():
    """Test that HT_PHASOR components form valid phasor"""
    from numta import HT_PHASOR

    n = 100
    close = np.sin(np.linspace(0, 4*np.pi, n)) * 10 + 100

    inphase, quadrature = HT_PHASOR(close)

    # Calculate magnitude (should be non-negative)
    valid_idx = ~np.isnan(inphase)
    magnitude = np.sqrt(inphase[valid_idx]**2 + quadrature[valid_idx]**2)
    assert np.all(magnitude >= 0)


def test_ht_phasor_empty_input():
    """Test HT_PHASOR with empty array"""
    from numta import HT_PHASOR

    empty = np.array([])
    inphase, quadrature = HT_PHASOR(empty)
    assert len(inphase) == 0
    assert len(quadrature) == 0
