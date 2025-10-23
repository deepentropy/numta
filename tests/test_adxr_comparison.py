"""
Test suite comparing talib-pure ADXR outputs with original TA-Lib
"""

import numpy as np
import pytest


def test_adxr_basic_comparison():
    """Test ADXR output matches TA-Lib for basic case"""
    try:
        import talib
    except ImportError:
        pytest.skip("TA-Lib not installed, skipping comparison tests")

    from talib_pure import ADXR as ADXR_pure

    # Test data
    np.random.seed(42)
    n = 100
    base_price = 100
    high = base_price + np.random.uniform(0, 2, n)
    low = base_price - np.random.uniform(0, 2, n)
    close = np.random.uniform(low, high)

    # Compare with default timeperiod
    talib_result = talib.ADXR(high, low, close)
    pure_result = ADXR_pure(high, low, close)

    np.testing.assert_array_almost_equal(
        talib_result, pure_result, decimal=10,
        err_msg="ADXR output differs from TA-Lib with default timeperiod"
    )


def test_adxr_various_timeperiods():
    """Test ADXR output matches TA-Lib for various timeperiods"""
    try:
        import talib
    except ImportError:
        pytest.skip("TA-Lib not installed, skipping comparison tests")

    from talib_pure import ADXR as ADXR_pure

    np.random.seed(123)
    n = 100
    base_price = 100
    high = base_price + np.random.uniform(0, 2, n)
    low = base_price - np.random.uniform(0, 2, n)
    close = np.random.uniform(low, high)

    # Test various timeperiods
    for timeperiod in [5, 10, 14, 20, 25]:
        talib_result = talib.ADXR(high, low, close, timeperiod=timeperiod)
        pure_result = ADXR_pure(high, low, close, timeperiod=timeperiod)

        np.testing.assert_array_almost_equal(
            talib_result, pure_result, decimal=10,
            err_msg=f"ADXR output differs from TA-Lib with timeperiod={timeperiod}"
        )


def test_adxr_trending_market():
    """Test ADXR with strong trending market"""
    try:
        import talib
    except ImportError:
        pytest.skip("TA-Lib not installed, skipping comparison tests")

    from talib_pure import ADXR as ADXR_pure

    # Create strong uptrend
    n = 70
    high = np.arange(10, 10 + n, dtype=np.float64)
    low = np.arange(9, 9 + n, dtype=np.float64)
    close = np.arange(9.5, 9.5 + n, dtype=np.float64)

    talib_result = talib.ADXR(high, low, close, timeperiod=14)
    pure_result = ADXR_pure(high, low, close, timeperiod=14)

    np.testing.assert_array_almost_equal(
        talib_result, pure_result, decimal=10,
        err_msg="ADXR differs for trending market"
    )

    # ADXR should be relatively high for strong trend
    assert np.nanmax(pure_result) > 20, "ADXR should be > 20 for strong trend"


def test_adxr_vs_adx_smoother():
    """Test that ADXR is smoother than ADX"""
    from talib_pure import ADX, ADXR

    np.random.seed(456)
    n = 100
    base_price = 100
    high = base_price + np.random.uniform(0, 2, n)
    low = base_price - np.random.uniform(0, 2, n)
    close = np.random.uniform(low, high)

    adx = ADX(high, low, close, timeperiod=14)
    adxr = ADXR(high, low, close, timeperiod=14)

    # Remove NaN values
    adx_valid = adx[~np.isnan(adx)]
    adxr_valid = adxr[~np.isnan(adxr)]

    # Calculate volatility (standard deviation of differences)
    adx_volatility = np.std(np.diff(adx_valid))
    adxr_volatility = np.std(np.diff(adxr_valid))

    # ADXR should be smoother (lower volatility) than ADX
    assert adxr_volatility < adx_volatility, "ADXR should be smoother than ADX"


def test_adxr_nan_handling():
    """Test that NaN values are placed correctly during lookback period"""
    from talib_pure import ADXR as ADXR_pure

    n = 70
    high = np.arange(10, 10 + n, dtype=np.float64)
    low = np.arange(9, 9 + n, dtype=np.float64)
    close = np.arange(9.5, 9.5 + n, dtype=np.float64)

    result = ADXR_pure(high, low, close, timeperiod=14)

    # First (3 * timeperiod - 2) = 40 values should be NaN
    assert all(np.isnan(result[:40]))

    # 41st value (index 40) should not be NaN
    assert not np.isnan(result[40])
    assert not np.isnan(result[-1])


def test_adxr_list_input():
    """Test ADXR works with list input (not just numpy arrays)"""
    from talib_pure import ADXR as ADXR_pure

    n = 50
    high = list(range(10, 10 + n))
    low = list(range(9, 9 + n))
    close = [i + 0.5 for i in range(9, 9 + n)]

    result = ADXR_pure(high, low, close, timeperiod=10)

    assert isinstance(result, np.ndarray)
    assert len(result) == len(high)


def test_adxr_empty_input():
    """Test ADXR handles empty input gracefully"""
    from talib_pure import ADXR as ADXR_pure

    high = np.array([])
    low = np.array([])
    close = np.array([])

    result = ADXR_pure(high, low, close, timeperiod=14)

    assert len(result) == 0


def test_adxr_insufficient_data():
    """Test ADXR when data length is less than lookback period"""
    from talib_pure import ADXR as ADXR_pure

    high = np.array([10, 11, 12])
    low = np.array([9, 10, 11])
    close = np.array([9.5, 10.5, 11.5])

    result = ADXR_pure(high, low, close, timeperiod=14)

    # All values should be NaN (need at least 3*14-2 = 40 values)
    assert all(np.isnan(result))
    assert len(result) == 3


def test_adxr_invalid_timeperiod():
    """Test ADXR raises error for invalid timeperiod"""
    from talib_pure import ADXR as ADXR_pure

    high = np.array([10, 11, 12, 13, 14])
    low = np.array([9, 10, 11, 12, 13])
    close = np.array([9.5, 10.5, 11.5, 12.5, 13.5])

    with pytest.raises(ValueError, match="timeperiod must be >= 2"):
        ADXR_pure(high, low, close, timeperiod=1)

    with pytest.raises(ValueError, match="timeperiod must be >= 2"):
        ADXR_pure(high, low, close, timeperiod=0)


def test_adxr_small_timeperiod():
    """Test ADXR with small timeperiod"""
    try:
        import talib
    except ImportError:
        pytest.skip("TA-Lib not installed, skipping comparison tests")

    from talib_pure import ADXR as ADXR_pure

    np.random.seed(42)
    n = 30
    high = np.arange(10, 10 + n, dtype=np.float64)
    low = np.arange(9, 9 + n, dtype=np.float64)
    close = np.arange(9.5, 9.5 + n, dtype=np.float64)

    # Test with minimum viable timeperiod
    talib_result = talib.ADXR(high, low, close, timeperiod=2)
    pure_result = ADXR_pure(high, low, close, timeperiod=2)

    np.testing.assert_array_almost_equal(
        talib_result, pure_result, decimal=10,
        err_msg="ADXR differs with timeperiod=2"
    )


def test_adxr_large_dataset():
    """Test ADXR with large dataset for performance validation"""
    try:
        import talib
    except ImportError:
        pytest.skip("TA-Lib not installed, skipping comparison tests")

    from talib_pure import ADXR as ADXR_pure

    # Large dataset
    np.random.seed(789)
    n = 10000
    base_price = 100
    high = base_price + np.random.uniform(0, 5, n)
    low = base_price - np.random.uniform(0, 5, n)
    close = np.random.uniform(low, high)

    talib_result = talib.ADXR(high, low, close, timeperiod=14)
    pure_result = ADXR_pure(high, low, close, timeperiod=14)

    np.testing.assert_array_almost_equal(
        talib_result, pure_result, decimal=8,
        err_msg="ADXR output differs from TA-Lib for large dataset"
    )


def test_adxr_value_range():
    """Test that ADXR values are in the expected range [0, 100]"""
    from talib_pure import ADXR as ADXR_pure

    np.random.seed(321)
    n = 100
    base_price = 100
    high = base_price + np.random.uniform(0, 5, n)
    low = base_price - np.random.uniform(0, 5, n)
    close = np.random.uniform(low, high)

    result = ADXR_pure(high, low, close, timeperiod=14)

    # Remove NaN values
    valid_values = result[~np.isnan(result)]

    # ADXR should be between 0 and 100
    assert np.all(valid_values >= 0), "ADXR values should be >= 0"
    assert np.all(valid_values <= 100), "ADXR values should be <= 100"


def test_adxr_formula_verification():
    """Test that ADXR correctly implements the formula"""
    from talib_pure import ADX, ADXR

    np.random.seed(999)
    n = 100
    base_price = 100
    high = base_price + np.random.uniform(0, 2, n)
    low = base_price - np.random.uniform(0, 2, n)
    close = np.random.uniform(low, high)

    timeperiod = 14
    adx = ADX(high, low, close, timeperiod=timeperiod)
    adxr = ADXR(high, low, close, timeperiod=timeperiod)

    # Verify formula: ADXR[i] = (ADX[i] + ADX[i - (timeperiod-1)]) / 2
    lookback = 3 * timeperiod - 2
    lag = timeperiod - 1

    for i in range(lookback, min(lookback + 10, n)):
        manual_adxr = (adx[i] + adx[i - lag]) / 2.0
        assert np.isclose(manual_adxr, adxr[i]), \
            f"ADXR[{i}] formula verification failed"
