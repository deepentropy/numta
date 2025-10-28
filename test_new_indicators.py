#!/usr/bin/env python3
"""Test script for newly implemented indicators"""

import numpy as np
import sys
sys.path.insert(0, 'src')

from numta import (
    MACDFIX, MFI, MINUS_DI, MINUS_DM, MOM,
    PLUS_DI, PLUS_DM, PPO
)

def test_macdfix():
    """Test MACDFIX indicator"""
    print("Testing MACDFIX...")
    close = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
                      110, 112, 111, 113, 115, 114, 116, 118, 117, 119,
                      120, 122, 121, 123, 125, 124, 126, 128, 127, 129,
                      130, 132, 131, 133, 135], dtype=np.float64)

    macd, signal, hist = MACDFIX(close, signalperiod=9)

    assert len(macd) == len(close), "MACDFIX: output length mismatch"
    assert isinstance(macd, np.ndarray), "MACDFIX: macd should be ndarray"
    assert isinstance(signal, np.ndarray), "MACDFIX: signal should be ndarray"
    assert isinstance(hist, np.ndarray), "MACDFIX: hist should be ndarray"

    # Check that early values are NaN
    assert np.isnan(macd[0]), "MACDFIX: first value should be NaN"

    # Check that later values are not NaN
    assert not np.isnan(macd[-1]), "MACDFIX: last value should not be NaN"

    print("  ✓ MACDFIX passed")
    print(f"    Last MACD: {macd[-1]:.4f}, Signal: {signal[-1]:.4f}, Hist: {hist[-1]:.4f}")


def test_mfi():
    """Test MFI indicator"""
    print("Testing MFI...")
    high = np.array([110, 112, 114, 113, 115, 117, 116, 118, 120, 119, 121,
                     123, 122, 124, 126, 125, 127, 129, 128, 130], dtype=np.float64)
    low = np.array([100, 102, 104, 103, 105, 107, 106, 108, 110, 109, 111,
                    113, 112, 114, 116, 115, 117, 119, 118, 120], dtype=np.float64)
    close = np.array([105, 107, 109, 108, 110, 112, 111, 113, 115, 114, 116,
                      118, 117, 119, 121, 120, 122, 124, 123, 125], dtype=np.float64)
    volume = np.array([1000, 1200, 1100, 1300, 1400, 1250, 1350, 1450, 1300, 1400,
                       1500, 1350, 1450, 1550, 1400, 1500, 1600, 1450, 1550, 1650], dtype=np.float64)

    mfi = MFI(high, low, close, volume, timeperiod=14)

    assert len(mfi) == len(close), "MFI: output length mismatch"
    assert isinstance(mfi, np.ndarray), "MFI: should be ndarray"

    # Check that early values are NaN
    assert np.isnan(mfi[0]), "MFI: first value should be NaN"

    # Check that later values are valid and in range [0, 100]
    assert not np.isnan(mfi[-1]), "MFI: last value should not be NaN"
    assert 0 <= mfi[-1] <= 100, "MFI: value should be between 0 and 100"

    print("  ✓ MFI passed")
    print(f"    Last MFI: {mfi[-1]:.4f}")


def test_minus_di():
    """Test MINUS_DI indicator"""
    print("Testing MINUS_DI...")
    high = np.array([110, 112, 114, 113, 115, 117, 116, 118, 120, 119, 121,
                     123, 122, 124, 126, 125], dtype=np.float64)
    low = np.array([100, 102, 104, 103, 105, 107, 106, 108, 110, 109, 111,
                    113, 112, 114, 116, 115], dtype=np.float64)
    close = np.array([105, 107, 109, 108, 110, 112, 111, 113, 115, 114, 116,
                      118, 117, 119, 121, 120], dtype=np.float64)

    minus_di = MINUS_DI(high, low, close, timeperiod=14)

    assert len(minus_di) == len(close), "MINUS_DI: output length mismatch"
    assert isinstance(minus_di, np.ndarray), "MINUS_DI: should be ndarray"

    # Check that early values are NaN
    assert np.isnan(minus_di[0]), "MINUS_DI: first value should be NaN"

    # Check that later values are valid
    assert not np.isnan(minus_di[-1]), "MINUS_DI: last value should not be NaN"
    assert minus_di[-1] >= 0, "MINUS_DI: value should be >= 0"

    print("  ✓ MINUS_DI passed")
    print(f"    Last MINUS_DI: {minus_di[-1]:.4f}")


def test_minus_dm():
    """Test MINUS_DM indicator"""
    print("Testing MINUS_DM...")
    high = np.array([110, 112, 114, 113, 115, 117, 116, 118, 120, 119, 121,
                     123, 122, 124, 126, 125], dtype=np.float64)
    low = np.array([100, 102, 104, 103, 105, 107, 106, 108, 110, 109, 111,
                    113, 112, 114, 116, 115], dtype=np.float64)

    minus_dm = MINUS_DM(high, low, timeperiod=14)

    assert len(minus_dm) == len(high), "MINUS_DM: output length mismatch"
    assert isinstance(minus_dm, np.ndarray), "MINUS_DM: should be ndarray"

    # Check that early values are NaN
    assert np.isnan(minus_dm[0]), "MINUS_DM: first value should be NaN"

    # Check that later values are valid
    assert not np.isnan(minus_dm[-1]), "MINUS_DM: last value should not be NaN"
    assert minus_dm[-1] >= 0, "MINUS_DM: value should be >= 0"

    print("  ✓ MINUS_DM passed")
    print(f"    Last MINUS_DM: {minus_dm[-1]:.4f}")


def test_mom():
    """Test MOM indicator"""
    print("Testing MOM...")
    close = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 110,
                      112, 111, 113, 115], dtype=np.float64)

    mom = MOM(close, timeperiod=10)

    assert len(mom) == len(close), "MOM: output length mismatch"
    assert isinstance(mom, np.ndarray), "MOM: should be ndarray"

    # Check that early values are NaN
    assert np.isnan(mom[0]), "MOM: first value should be NaN"

    # Check that later values are valid
    assert not np.isnan(mom[-1]), "MOM: last value should not be NaN"

    # Manually verify: MOM[14] = close[14] - close[4] = 115 - 105 = 10
    assert abs(mom[-1] - 10.0) < 0.001, "MOM: calculation error"

    print("  ✓ MOM passed")
    print(f"    Last MOM: {mom[-1]:.4f}")


def test_plus_di():
    """Test PLUS_DI indicator"""
    print("Testing PLUS_DI...")
    high = np.array([110, 112, 114, 113, 115, 117, 116, 118, 120, 119, 121,
                     123, 122, 124, 126, 125], dtype=np.float64)
    low = np.array([100, 102, 104, 103, 105, 107, 106, 108, 110, 109, 111,
                    113, 112, 114, 116, 115], dtype=np.float64)
    close = np.array([105, 107, 109, 108, 110, 112, 111, 113, 115, 114, 116,
                      118, 117, 119, 121, 120], dtype=np.float64)

    plus_di = PLUS_DI(high, low, close, timeperiod=14)

    assert len(plus_di) == len(close), "PLUS_DI: output length mismatch"
    assert isinstance(plus_di, np.ndarray), "PLUS_DI: should be ndarray"

    # Check that early values are NaN
    assert np.isnan(plus_di[0]), "PLUS_DI: first value should be NaN"

    # Check that later values are valid
    assert not np.isnan(plus_di[-1]), "PLUS_DI: last value should not be NaN"
    assert plus_di[-1] >= 0, "PLUS_DI: value should be >= 0"

    print("  ✓ PLUS_DI passed")
    print(f"    Last PLUS_DI: {plus_di[-1]:.4f}")


def test_plus_dm():
    """Test PLUS_DM indicator"""
    print("Testing PLUS_DM...")
    high = np.array([110, 112, 114, 113, 115, 117, 116, 118, 120, 119, 121,
                     123, 122, 124, 126, 125], dtype=np.float64)
    low = np.array([100, 102, 104, 103, 105, 107, 106, 108, 110, 109, 111,
                    113, 112, 114, 116, 115], dtype=np.float64)

    plus_dm = PLUS_DM(high, low, timeperiod=14)

    assert len(plus_dm) == len(high), "PLUS_DM: output length mismatch"
    assert isinstance(plus_dm, np.ndarray), "PLUS_DM: should be ndarray"

    # Check that early values are NaN
    assert np.isnan(plus_dm[0]), "PLUS_DM: first value should be NaN"

    # Check that later values are valid
    assert not np.isnan(plus_dm[-1]), "PLUS_DM: last value should not be NaN"
    assert plus_dm[-1] >= 0, "PLUS_DM: value should be >= 0"

    print("  ✓ PLUS_DM passed")
    print(f"    Last PLUS_DM: {plus_dm[-1]:.4f}")


def test_ppo():
    """Test PPO indicator"""
    print("Testing PPO...")
    close = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
                      110, 112, 111, 113, 115, 114, 116, 118, 117, 119,
                      120, 122, 121, 123, 125, 124, 126, 128, 127, 129], dtype=np.float64)

    ppo = PPO(close, fastperiod=12, slowperiod=26)

    assert len(ppo) == len(close), "PPO: output length mismatch"
    assert isinstance(ppo, np.ndarray), "PPO: should be ndarray"

    # Check that early values are NaN
    assert np.isnan(ppo[0]), "PPO: first value should be NaN"

    # Check that later values are valid
    assert not np.isnan(ppo[-1]), "PPO: last value should not be NaN"

    print("  ✓ PPO passed")
    print(f"    Last PPO: {ppo[-1]:.4f}")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("Testing newly implemented indicators")
    print("="*60 + "\n")

    try:
        test_macdfix()
        test_mfi()
        test_minus_di()
        test_minus_dm()
        test_mom()
        test_plus_di()
        test_plus_dm()
        test_ppo()

        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60 + "\n")
        return 0

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
