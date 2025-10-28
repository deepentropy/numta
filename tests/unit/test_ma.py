"""
Test suite for MA
"""

import numpy as np
import pytest

def test_ma_sma():
    """Test MA with SMA (matype=0)"""
    from numta import MA, SMA

    close = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])

    ma_result = MA(close, timeperiod=5, matype=0)
    sma_result = SMA(close, timeperiod=5)

    np.testing.assert_array_equal(ma_result, sma_result)


def test_ma_ema():
    """Test MA with EMA (matype=1)"""
    from numta import MA, EMA

    close = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])

    ma_result = MA(close, timeperiod=5, matype=1)
    ema_result = EMA(close, timeperiod=5)

    np.testing.assert_array_equal(ma_result, ema_result)


def test_ma_dema():
    """Test MA with DEMA (matype=3)"""
    from numta import MA, DEMA

    close = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 110, 112])

    ma_result = MA(close, timeperiod=5, matype=3)
    dema_result = DEMA(close, timeperiod=5)

    np.testing.assert_array_equal(ma_result, dema_result)


def test_ma_kama():
    """Test MA with KAMA (matype=6)"""
    from numta import MA, KAMA

    close = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
                     110, 112, 111, 113, 115])

    ma_result = MA(close, timeperiod=10, matype=6)
    kama_result = KAMA(close, timeperiod=10)

    np.testing.assert_array_equal(ma_result, kama_result)


def test_ma_wma():
    """Test MA with WMA (matype=2)"""
    from numta import MA, WMA

    close = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])

    ma_result = MA(close, timeperiod=5, matype=2)
    wma_result = WMA(close, timeperiod=5)

    np.testing.assert_array_equal(ma_result, wma_result)


def test_ma_tema():
    """Test MA with TEMA (matype=4)"""
    from numta import MA, TEMA

    close = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
                     110, 112, 111, 113, 115])

    ma_result = MA(close, timeperiod=5, matype=4)
    tema_result = TEMA(close, timeperiod=5)

    np.testing.assert_array_equal(ma_result, tema_result)


def test_ma_trima():
    """Test MA with TRIMA (matype=5)"""
    from numta import MA, TRIMA

    close = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
                     110, 112, 111, 113, 115])

    ma_result = MA(close, timeperiod=5, matype=5)
    trima_result = TRIMA(close, timeperiod=5)

    np.testing.assert_array_equal(ma_result, trima_result)


def test_ma_t3():
    """Test MA with T3 (matype=8)"""
    from numta import MA, T3

    close = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
                     110, 112, 111, 113, 115, 114, 116, 118, 117, 119])

    ma_result = MA(close, timeperiod=5, matype=8)
    t3_result = T3(close, timeperiod=5)

    np.testing.assert_array_equal(ma_result, t3_result)


def test_ma_invalid_matype():
    """Test MA validates matype"""
    from numta import MA

    close = np.array([100, 102, 101, 103, 105])

    with pytest.raises(ValueError):
        MA(close, timeperiod=3, matype=99)
