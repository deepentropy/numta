"""
Test suite comparing talib-pure AD outputs with original TA-Lib
"""

import numpy as np
import pytest


def test_ad_basic_comparison():
    """Test AD output matches TA-Lib for basic case"""
    try:
        import talib
    except ImportError:
        pytest.skip("TA-Lib not installed, skipping comparison tests")

    from talib_pure import AD as AD_pure

    # Test data with realistic price ranges
    np.random.seed(42)
    n = 100
    base_price = 100
    high = base_price + np.random.uniform(0, 2, n)
    low = base_price - np.random.uniform(0, 2, n)
    close = np.random.uniform(low, high)
    volume = np.random.uniform(100000, 200000, n)

    # Compare with TA-Lib
    talib_result = talib.AD(high, low, close, volume)
    pure_result = AD_pure(high, low, close, volume)

    np.testing.assert_array_almost_equal(
        talib_result, pure_result, decimal=8,
        err_msg="AD output differs from TA-Lib"
    )


def test_ad_simple_case():
    """Test AD with simple known values"""
    try:
        import talib
    except ImportError:
        pytest.skip("TA-Lib not installed, skipping comparison tests")

    from talib_pure import AD as AD_pure

    # Simple test case
    high = np.array([10, 11, 12, 11, 13], dtype=np.float64)
    low = np.array([9, 10, 10, 9, 11], dtype=np.float64)
    close = np.array([9.5, 10.5, 11, 10, 12], dtype=np.float64)
    volume = np.array([1000, 1100, 1200, 900, 1300], dtype=np.float64)

    talib_result = talib.AD(high, low, close, volume)
    pure_result = AD_pure(high, low, close, volume)

    np.testing.assert_array_almost_equal(
        talib_result, pure_result, decimal=10,
        err_msg="AD output differs from TA-Lib for simple case"
    )


def test_ad_edge_case_same_high_low():
    """Test AD when high equals low (division by zero case)"""
    try:
        import talib
    except ImportError:
        pytest.skip("TA-Lib not installed, skipping comparison tests")

    from talib_pure import AD as AD_pure

    # Case where high == low (should not change AD line)
    high = np.array([10, 10, 10], dtype=np.float64)
    low = np.array([10, 10, 10], dtype=np.float64)
    close = np.array([10, 10, 10], dtype=np.float64)
    volume = np.array([1000, 1000, 1000], dtype=np.float64)

    talib_result = talib.AD(high, low, close, volume)
    pure_result = AD_pure(high, low, close, volume)

    np.testing.assert_array_almost_equal(
        talib_result, pure_result, decimal=10,
        err_msg="AD output differs from TA-Lib when high == low"
    )


def test_ad_mixed_scenarios():
    """Test AD with various price scenarios"""
    try:
        import talib
    except ImportError:
        pytest.skip("TA-Lib not installed, skipping comparison tests")

    from talib_pure import AD as AD_pure

    # Mix of normal bars and bars where high == low
    high = np.array([10, 11, 12, 12, 13, 14, 15], dtype=np.float64)
    low = np.array([9, 10, 12, 9, 11, 12, 13], dtype=np.float64)
    close = np.array([9.5, 10.5, 12, 10, 12, 13, 14], dtype=np.float64)
    volume = np.array([1000, 1100, 1200, 900, 1300, 1400, 1500], dtype=np.float64)

    talib_result = talib.AD(high, low, close, volume)
    pure_result = AD_pure(high, low, close, volume)

    np.testing.assert_array_almost_equal(
        talib_result, pure_result, decimal=10,
        err_msg="AD output differs from TA-Lib for mixed scenarios"
    )


def test_ad_accumulation_pattern():
    """Test AD with strong accumulation pattern (close near high)"""
    try:
        import talib
    except ImportError:
        pytest.skip("TA-Lib not installed, skipping comparison tests")

    from talib_pure import AD as AD_pure

    # Close near high (accumulation)
    high = np.array([10, 11, 12, 13, 14], dtype=np.float64)
    low = np.array([9, 10, 11, 12, 13], dtype=np.float64)
    close = np.array([9.9, 10.9, 11.9, 12.9, 13.9], dtype=np.float64)
    volume = np.array([1000, 1000, 1000, 1000, 1000], dtype=np.float64)

    talib_result = talib.AD(high, low, close, volume)
    pure_result = AD_pure(high, low, close, volume)

    np.testing.assert_array_almost_equal(
        talib_result, pure_result, decimal=10,
        err_msg="AD output differs from TA-Lib for accumulation pattern"
    )

    # AD line should be increasing (positive money flow)
    assert np.all(np.diff(pure_result) > 0), "AD should increase with accumulation"


def test_ad_distribution_pattern():
    """Test AD with strong distribution pattern (close near low)"""
    try:
        import talib
    except ImportError:
        pytest.skip("TA-Lib not installed, skipping comparison tests")

    from talib_pure import AD as AD_pure

    # Close near low (distribution)
    high = np.array([10, 11, 12, 13, 14], dtype=np.float64)
    low = np.array([9, 10, 11, 12, 13], dtype=np.float64)
    close = np.array([9.1, 10.1, 11.1, 12.1, 13.1], dtype=np.float64)
    volume = np.array([1000, 1000, 1000, 1000, 1000], dtype=np.float64)

    talib_result = talib.AD(high, low, close, volume)
    pure_result = AD_pure(high, low, close, volume)

    np.testing.assert_array_almost_equal(
        talib_result, pure_result, decimal=10,
        err_msg="AD output differs from TA-Lib for distribution pattern"
    )

    # AD line should be decreasing (negative money flow)
    assert np.all(np.diff(pure_result) < 0), "AD should decrease with distribution"


def test_ad_list_input():
    """Test AD works with list input (not just numpy arrays)"""
    from talib_pure import AD as AD_pure

    high = [10, 11, 12, 11, 13]
    low = [9, 10, 10, 9, 11]
    close = [9.5, 10.5, 11, 10, 12]
    volume = [1000, 1100, 1200, 900, 1300]

    result = AD_pure(high, low, close, volume)

    assert isinstance(result, np.ndarray)
    assert len(result) == len(high)


def test_ad_empty_input():
    """Test AD handles empty input gracefully"""
    from talib_pure import AD as AD_pure

    high = np.array([])
    low = np.array([])
    close = np.array([])
    volume = np.array([])

    result = AD_pure(high, low, close, volume)

    assert len(result) == 0


def test_ad_mismatched_lengths():
    """Test AD raises error for mismatched input lengths"""
    from talib_pure import AD as AD_pure

    high = np.array([10, 11, 12])
    low = np.array([9, 10])  # Different length
    close = np.array([9.5, 10.5, 11])
    volume = np.array([1000, 1100, 1200])

    with pytest.raises(ValueError, match="same length"):
        AD_pure(high, low, close, volume)


def test_ad_large_dataset():
    """Test AD with large dataset for performance validation"""
    try:
        import talib
    except ImportError:
        pytest.skip("TA-Lib not installed, skipping comparison tests")

    from talib_pure import AD as AD_pure

    # Large dataset
    np.random.seed(123)
    n = 10000
    base_price = 100
    high = base_price + np.random.uniform(0, 5, n)
    low = base_price - np.random.uniform(0, 5, n)
    close = np.random.uniform(low, high)
    volume = np.random.uniform(100000, 500000, n)

    talib_result = talib.AD(high, low, close, volume)
    pure_result = AD_pure(high, low, close, volume)

    np.testing.assert_array_almost_equal(
        talib_result, pure_result, decimal=6,
        err_msg="AD output differs from TA-Lib for large dataset"
    )


def test_ad_extreme_values():
    """Test AD with extreme volume values"""
    try:
        import talib
    except ImportError:
        pytest.skip("TA-Lib not installed, skipping comparison tests")

    from talib_pure import AD as AD_pure

    # Extreme volume values
    high = np.array([100, 101, 102], dtype=np.float64)
    low = np.array([99, 100, 101], dtype=np.float64)
    close = np.array([99.5, 100.5, 101.5], dtype=np.float64)
    volume = np.array([1e9, 1e10, 1e11], dtype=np.float64)

    talib_result = talib.AD(high, low, close, volume)
    pure_result = AD_pure(high, low, close, volume)

    np.testing.assert_array_almost_equal(
        talib_result, pure_result, decimal=2,
        err_msg="AD output differs from TA-Lib for extreme volumes"
    )