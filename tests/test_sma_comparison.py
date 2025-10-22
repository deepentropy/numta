"""
Test suite comparing talib-pure outputs with original TA-Lib
"""

import numpy as np
import pytest


def test_sma_basic_comparison():
    """Test SMA output matches TA-Lib for basic case"""
    try:
        import talib
    except ImportError:
        pytest.skip("TA-Lib not installed, skipping comparison tests")

    from talib_pure import SMA as SMA_pure

    # Test data
    close = np.random.uniform(100, 200, 100)

    # Compare with default timeperiod
    talib_result = talib.SMA(close)
    pure_result = SMA_pure(close)

    np.testing.assert_array_almost_equal(
        talib_result, pure_result, decimal=10,
        err_msg="SMA output differs from TA-Lib with default timeperiod"
    )


def test_sma_various_timeperiods():
    """Test SMA output matches TA-Lib for various timeperiods"""
    try:
        import talib
    except ImportError:
        pytest.skip("TA-Lib not installed, skipping comparison tests")

    from talib_pure import SMA as SMA_pure

    close = np.random.uniform(100, 200, 100)

    # Test various timeperiods
    for timeperiod in [5, 10, 20, 30, 50]:
        talib_result = talib.SMA(close, timeperiod=timeperiod)
        pure_result = SMA_pure(close, timeperiod=timeperiod)

        np.testing.assert_array_almost_equal(
            talib_result, pure_result, decimal=10,
            err_msg=f"SMA output differs from TA-Lib with timeperiod={timeperiod}"
        )


def test_sma_edge_cases():
    """Test SMA edge cases match TA-Lib behavior"""
    try:
        import talib
    except ImportError:
        pytest.skip("TA-Lib not installed, skipping comparison tests")

    from talib_pure import SMA as SMA_pure

    # Small dataset
    close_small = np.array([100, 101, 102, 103, 104], dtype=np.float64)
    for timeperiod in [2, 3, 5]:
        if len(close_small) >= timeperiod:
            talib_result = talib.SMA(close_small, timeperiod=timeperiod)
            pure_result = SMA_pure(close_small, timeperiod=timeperiod)
            np.testing.assert_array_almost_equal(
                talib_result, pure_result, decimal=10,
                err_msg=f"SMA differs for small dataset with timeperiod={timeperiod}"
            )

    # Test with minimum valid timeperiod (TA-Lib requires timeperiod >= 2)
    close_two = np.array([100.0, 101.0], dtype=np.float64)
    talib_result = talib.SMA(close_two, timeperiod=2)
    pure_result = SMA_pure(close_two, timeperiod=2)
    np.testing.assert_array_almost_equal(
        talib_result, pure_result, decimal=10,
        err_msg="SMA differs for minimum timeperiod=2"
    )


def test_sma_nan_handling():
    """Test that NaN values are placed correctly during lookback period"""
    from talib_pure import SMA as SMA_pure

    close = np.arange(1, 11, dtype=np.float64)
    result = SMA_pure(close, timeperiod=3)

    # First (timeperiod - 1) values should be NaN
    assert np.isnan(result[0])
    assert np.isnan(result[1])

    # Subsequent values should not be NaN
    assert not np.isnan(result[2])
    assert not np.isnan(result[-1])

    # Check actual values (using approximate equality for floating point)
    np.testing.assert_almost_equal(result[2], 2.0, decimal=10)  # (1+2+3)/3
    np.testing.assert_almost_equal(result[3], 3.0, decimal=10)  # (2+3+4)/3
    np.testing.assert_almost_equal(result[4], 4.0, decimal=10)  # (3+4+5)/3


def test_sma_list_input():
    """Test SMA works with list input (not just numpy arrays)"""
    from talib_pure import SMA as SMA_pure

    close_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = SMA_pure(close_list, timeperiod=3)

    assert isinstance(result, np.ndarray)
    assert len(result) == len(close_list)
    assert np.isnan(result[0])
    assert np.isnan(result[1])
    np.testing.assert_almost_equal(result[2], 2.0, decimal=10)


def test_sma_empty_input():
    """Test SMA handles empty input gracefully"""
    from talib_pure import SMA as SMA_pure

    close_empty = np.array([])
    result = SMA_pure(close_empty, timeperiod=5)

    assert len(result) == 0


def test_sma_insufficient_data():
    """Test SMA when data length is less than timeperiod"""
    from talib_pure import SMA as SMA_pure

    close = np.array([1, 2, 3])
    result = SMA_pure(close, timeperiod=5)

    # All values should be NaN
    assert all(np.isnan(result))
    assert len(result) == 3


def test_sma_invalid_timeperiod():
    """Test SMA raises error for invalid timeperiod"""
    from talib_pure import SMA as SMA_pure

    close = np.array([1, 2, 3, 4, 5])

    # TA-Lib requires timeperiod >= 2
    with pytest.raises(ValueError):
        SMA_pure(close, timeperiod=0)

    with pytest.raises(ValueError):
        SMA_pure(close, timeperiod=1)

    with pytest.raises(ValueError):
        SMA_pure(close, timeperiod=-1)
