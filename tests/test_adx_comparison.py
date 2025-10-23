"""
Test suite comparing talib-pure ADX outputs with original TA-Lib
"""

import numpy as np
import pytest


def test_adx_basic_comparison():
    """Test ADX output matches TA-Lib for basic case"""
    try:
        import talib
    except ImportError:
        pytest.skip("TA-Lib not installed, skipping comparison tests")

    from talib_pure import ADX as ADX_pure

    # Test data
    np.random.seed(42)
    n = 100
    base_price = 100
    high = base_price + np.random.uniform(0, 2, n)
    low = base_price - np.random.uniform(0, 2, n)
    close = np.random.uniform(low, high)

    # Compare with default timeperiod
    talib_result = talib.ADX(high, low, close)
    pure_result = ADX_pure(high, low, close)

    np.testing.assert_array_almost_equal(
        talib_result, pure_result, decimal=10,
        err_msg="ADX output differs from TA-Lib with default timeperiod"
    )


def test_adx_various_timeperiods():
    """Test ADX output matches TA-Lib for various timeperiods"""
    try:
        import talib
    except ImportError:
        pytest.skip("TA-Lib not installed, skipping comparison tests")

    from talib_pure import ADX as ADX_pure

    np.random.seed(123)
    n = 100
    base_price = 100
    high = base_price + np.random.uniform(0, 2, n)
    low = base_price - np.random.uniform(0, 2, n)
    close = np.random.uniform(low, high)

    # Test various timeperiods
    for timeperiod in [5, 10, 14, 20, 25]:
        talib_result = talib.ADX(high, low, close, timeperiod=timeperiod)
        pure_result = ADX_pure(high, low, close, timeperiod=timeperiod)

        np.testing.assert_array_almost_equal(
            talib_result, pure_result, decimal=10,
            err_msg=f"ADX output differs from TA-Lib with timeperiod={timeperiod}"
        )


def test_adx_trending_market():
    """Test ADX with strong trending market (should show high ADX)"""
    try:
        import talib
    except ImportError:
        pytest.skip("TA-Lib not installed, skipping comparison tests")

    from talib_pure import ADX as ADX_pure

    # Create strong uptrend
    n = 50
    high = np.arange(10, 10 + n, dtype=np.float64)
    low = np.arange(9, 9 + n, dtype=np.float64)
    close = np.arange(9.5, 9.5 + n, dtype=np.float64)

    talib_result = talib.ADX(high, low, close, timeperiod=14)
    pure_result = ADX_pure(high, low, close, timeperiod=14)

    np.testing.assert_array_almost_equal(
        talib_result, pure_result, decimal=10,
        err_msg="ADX differs for trending market"
    )

    # ADX should be relatively high for strong trend
    assert np.nanmax(pure_result) > 20, "ADX should be > 20 for strong trend"


def test_adx_ranging_market():
    """Test ADX with ranging market (should show low ADX)"""
    try:
        import talib
    except ImportError:
        pytest.skip("TA-Lib not installed, skipping comparison tests")

    from talib_pure import ADX as ADX_pure

    # Create ranging market (oscillating prices)
    n = 50
    high = np.array([10 + (i % 2) for i in range(n)], dtype=np.float64)
    low = np.array([9 + (i % 2) for i in range(n)], dtype=np.float64)
    close = np.array([9.5 + (i % 2) for i in range(n)], dtype=np.float64)

    talib_result = talib.ADX(high, low, close, timeperiod=14)
    pure_result = ADX_pure(high, low, close, timeperiod=14)

    np.testing.assert_array_almost_equal(
        talib_result, pure_result, decimal=10,
        err_msg="ADX differs for ranging market"
    )


def test_adx_nan_handling():
    """Test that NaN values are placed correctly during lookback period"""
    from talib_pure import ADX as ADX_pure

    n = 50
    high = np.arange(10, 10 + n, dtype=np.float64)
    low = np.arange(9, 9 + n, dtype=np.float64)
    close = np.arange(9.5, 9.5 + n, dtype=np.float64)

    result = ADX_pure(high, low, close, timeperiod=14)

    # First (2 * timeperiod - 1) = 27 values should be NaN
    assert all(np.isnan(result[:27]))

    # 28th value (index 27) should not be NaN
    assert not np.isnan(result[27])
    assert not np.isnan(result[-1])


def test_adx_list_input():
    """Test ADX works with list input (not just numpy arrays)"""
    from talib_pure import ADX as ADX_pure

    high = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    low = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    close = [9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5]

    result = ADX_pure(high, low, close, timeperiod=5)

    assert isinstance(result, np.ndarray)
    assert len(result) == len(high)


def test_adx_empty_input():
    """Test ADX handles empty input gracefully"""
    from talib_pure import ADX as ADX_pure

    high = np.array([])
    low = np.array([])
    close = np.array([])

    result = ADX_pure(high, low, close, timeperiod=14)

    assert len(result) == 0


def test_adx_insufficient_data():
    """Test ADX when data length is less than lookback period"""
    from talib_pure import ADX as ADX_pure

    high = np.array([10, 11, 12])
    low = np.array([9, 10, 11])
    close = np.array([9.5, 10.5, 11.5])

    result = ADX_pure(high, low, close, timeperiod=14)

    # All values should be NaN (need at least 2*14 = 28 values)
    assert all(np.isnan(result))
    assert len(result) == 3


def test_adx_mismatched_lengths():
    """Test ADX raises error for mismatched input lengths"""
    from talib_pure import ADX as ADX_pure

    high = np.array([10, 11, 12])
    low = np.array([9, 10])  # Different length
    close = np.array([9.5, 10.5, 11])

    with pytest.raises(ValueError, match="same length"):
        ADX_pure(high, low, close)


def test_adx_invalid_timeperiod():
    """Test ADX raises error for invalid timeperiod"""
    from talib_pure import ADX as ADX_pure

    high = np.array([10, 11, 12, 13, 14])
    low = np.array([9, 10, 11, 12, 13])
    close = np.array([9.5, 10.5, 11.5, 12.5, 13.5])

    with pytest.raises(ValueError, match="timeperiod must be >= 2"):
        ADX_pure(high, low, close, timeperiod=1)

    with pytest.raises(ValueError, match="timeperiod must be >= 2"):
        ADX_pure(high, low, close, timeperiod=0)


def test_adx_small_timeperiod():
    """Test ADX with small timeperiod"""
    try:
        import talib
    except ImportError:
        pytest.skip("TA-Lib not installed, skipping comparison tests")

    from talib_pure import ADX as ADX_pure

    np.random.seed(42)
    n = 30
    high = np.arange(10, 10 + n, dtype=np.float64)
    low = np.arange(9, 9 + n, dtype=np.float64)
    close = np.arange(9.5, 9.5 + n, dtype=np.float64)

    # Test with minimum viable timeperiod
    talib_result = talib.ADX(high, low, close, timeperiod=2)
    pure_result = ADX_pure(high, low, close, timeperiod=2)

    np.testing.assert_array_almost_equal(
        talib_result, pure_result, decimal=10,
        err_msg="ADX differs with timeperiod=2"
    )


def test_adx_large_dataset():
    """Test ADX with large dataset for performance validation"""
    try:
        import talib
    except ImportError:
        pytest.skip("TA-Lib not installed, skipping comparison tests")

    from talib_pure import ADX as ADX_pure

    # Large dataset
    np.random.seed(456)
    n = 10000
    base_price = 100
    high = base_price + np.random.uniform(0, 5, n)
    low = base_price - np.random.uniform(0, 5, n)
    close = np.random.uniform(low, high)

    talib_result = talib.ADX(high, low, close, timeperiod=14)
    pure_result = ADX_pure(high, low, close, timeperiod=14)

    np.testing.assert_array_almost_equal(
        talib_result, pure_result, decimal=8,
        err_msg="ADX output differs from TA-Lib for large dataset"
    )


def test_adx_value_range():
    """Test that ADX values are in the expected range [0, 100]"""
    from talib_pure import ADX as ADX_pure

    np.random.seed(789)
    n = 100
    base_price = 100
    high = base_price + np.random.uniform(0, 5, n)
    low = base_price - np.random.uniform(0, 5, n)
    close = np.random.uniform(low, high)

    result = ADX_pure(high, low, close, timeperiod=14)

    # Remove NaN values
    valid_values = result[~np.isnan(result)]

    # ADX should be between 0 and 100
    assert np.all(valid_values >= 0), "ADX values should be >= 0"
    assert np.all(valid_values <= 100), "ADX values should be <= 100"