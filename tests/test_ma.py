"""
Test suite for MA
"""

import numpy as np
import pytest

def test_ma_sma():
    """Test MA with SMA (matype=0)"""
    from talib_pure import MA, SMA

    close = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])

    ma_result = MA(close, timeperiod=5, matype=0)
    sma_result = SMA(close, timeperiod=5)

    np.testing.assert_array_equal(ma_result, sma_result)


def test_ma_ema():
    """Test MA with EMA (matype=1)"""
    from talib_pure import MA, EMA

    close = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])

    ma_result = MA(close, timeperiod=5, matype=1)
    ema_result = EMA(close, timeperiod=5)

    np.testing.assert_array_equal(ma_result, ema_result)


def test_ma_dema():
    """Test MA with DEMA (matype=3)"""
    from talib_pure import MA, DEMA

    close = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 110, 112])

    ma_result = MA(close, timeperiod=5, matype=3)
    dema_result = DEMA(close, timeperiod=5)

    np.testing.assert_array_equal(ma_result, dema_result)


def test_ma_kama():
    """Test MA with KAMA (matype=6)"""
    from talib_pure import MA, KAMA

    close = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
                     110, 112, 111, 113, 115])

    ma_result = MA(close, timeperiod=10, matype=6)
    kama_result = KAMA(close, timeperiod=10)

    np.testing.assert_array_equal(ma_result, kama_result)


def test_ma_not_implemented():
    """Test MA raises NotImplementedError for unsupported types"""
    from talib_pure import MA

    close = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])

    # WMA (matype=2) not implemented
    with pytest.raises(NotImplementedError):
        MA(close, timeperiod=5, matype=2)

    # TEMA (matype=4) not implemented
    with pytest.raises(NotImplementedError):
        MA(close, timeperiod=5, matype=4)


def test_ma_invalid_matype():
    """Test MA validates matype"""
    from talib_pure import MA

    close = np.array([100, 102, 101, 103, 105])

    with pytest.raises(ValueError):
        MA(close, timeperiod=3, matype=99)
