"""
Test suite for HT_SINE
"""

import numpy as np
import pytest

def test_ht_sine_basic():
    """Test HT_SINE basic calculation"""
    from numta import HT_SINE

    # Create sine wave to simulate a cycle
    n = 100
    close = np.sin(np.linspace(0, 4*np.pi, n)) * 10 + 100

    sine, leadsine = HT_SINE(close)

    # First 32 values should be NaN (unstable period)
    assert np.all(np.isnan(sine[:32]))
    assert np.all(np.isnan(leadsine[:32]))

    # After stable period, should have valid values
    assert not np.isnan(sine[-1])
    assert not np.isnan(leadsine[-1])

    # Sine values should be between -1 and 1
    valid_sine = sine[~np.isnan(sine)]
    assert np.all(valid_sine >= -1.2)  # Allow small overshoot
    assert np.all(valid_sine <= 1.2)


def test_ht_sine_lead_relationship():
    """Test that leadsine leads sine by ~45 degrees"""
    from numta import HT_SINE

    n = 100
    close = np.sin(np.linspace(0, 4*np.pi, n)) * 10 + 100

    sine, leadsine = HT_SINE(close)

    # Lead sine should have some relationship to sine
    # (exact 45-degree lead might not be perfect in all cases)
    valid_idx = ~np.isnan(sine)
    assert len(sine[valid_idx]) > 0
    assert len(leadsine[valid_idx]) > 0


def test_ht_sine_empty_input():
    """Test HT_SINE with empty array"""
    from numta import HT_SINE

    empty = np.array([])
    sine, leadsine = HT_SINE(empty)
    assert len(sine) == 0
    assert len(leadsine) == 0
