"""
GPU Batch API - Run indicators on thousands of tickers simultaneously.

Each function accepts 2D arrays of shape (num_tickers, num_bars) and returns
results of the same shape. One CUDA thread processes one ticker's full time series.
"""

import numpy as np
from ..gpu import HAS_CUDA

if HAS_CUDA:
    from numba import cuda
    from ..gpu.memory import to_device_2d, allocate_output_2d, compute_grid_1d
    from ..gpu.overlap import (
        _sma_batch_cuda, _ema_batch_cuda, _bbands_batch_cuda,
        _dema_batch_cuda, _kama_batch_cuda, _wma_batch_cuda,
        _tema_batch_cuda, _t3_batch_cuda, _trima_batch_cuda,
        _mama_batch_cuda, _sar_batch_cuda, _sarext_batch_cuda,
    )
    from ..gpu.momentum_indicators import (
        _rsi_batch_cuda, _macd_batch_cuda, _adx_batch_cuda,
        _atr_batch_cuda, _cci_batch_cuda, _cmo_batch_cuda,
        _dx_batch_cuda, _mom_batch_cuda,
        _roc_batch_cuda, _rocp_batch_cuda, _rocr_batch_cuda, _rocr100_batch_cuda,
        _willr_batch_cuda, _stoch_fastk_batch_cuda,
        _bop_batch_cuda, _aroon_batch_cuda, _aroonosc_batch_cuda,
        _mfi_batch_cuda, _minus_dm_batch_cuda, _plus_dm_batch_cuda,
        _minus_di_batch_cuda, _plus_di_batch_cuda, _ultosc_batch_cuda,
    )
    from ..gpu.volatility_indicators import _trange_batch_cuda, _natr_batch_cuda
    from ..gpu.volume_indicators import _ad_batch_cuda, _obv_batch_cuda, _adosc_batch_cuda
    from ..gpu.statistics import _stddev_batch_cuda, _var_batch_cuda, _tsf_batch_cuda
    from ..gpu.statistic_functions import (
        _beta_batch_cuda, _correl_batch_cuda, _linearreg_batch_cuda,
        _linearreg_angle_batch_cuda, _linearreg_intercept_batch_cuda, _linearreg_slope_batch_cuda,
    )
    from ..gpu.math_operators import (
        _max_batch_cuda, _maxindex_batch_cuda, _min_batch_cuda, _minindex_batch_cuda,
        _minmax_batch_cuda, _minmaxindex_batch_cuda, _sum_batch_cuda,
    )
    from ..gpu.price_transform import (
        _medprice_batch_cuda, _midpoint_batch_cuda, _midprice_batch_cuda,
        _typprice_batch_cuda, _wclprice_batch_cuda,
    )
    from ..gpu.cycle_indicators import (
        _ht_trendline_batch_cuda, _ht_trendmode_batch_cuda,
        _ht_dcperiod_batch_cuda, _ht_dcphase_batch_cuda,
        _ht_phasor_batch_cuda, _ht_sine_batch_cuda,
    )
    from ..gpu.pattern_recognition import (
        _cdl2crows_batch_cuda, _cdl3blackcrows_batch_cuda,
        _cdl3inside_batch_cuda, _cdl3outside_batch_cuda,
        _cdl3starsinsouth_batch_cuda, _cdl3whitesoldiers_batch_cuda,
        _cdlabandonedbaby_batch_cuda, _cdladvanceblock_batch_cuda,
        _cdlbelthold_batch_cuda, _cdlbreakaway_batch_cuda,
        _cdlclosingmarubozu_batch_cuda, _cdlconcealbabyswall_batch_cuda,
        _cdlcounterattack_batch_cuda, _cdldarkcloudcover_batch_cuda,
        _cdldoji_batch_cuda, _cdldojistar_batch_cuda,
        _cdldragonflydoji_batch_cuda, _cdlengulfing_batch_cuda,
        _cdleveningdojistar_batch_cuda, _cdleveningstar_batch_cuda,
        _cdlgapsidesidewhite_batch_cuda, _cdlgravestonedoji_batch_cuda,
        _cdlhammer_batch_cuda, _cdlhangingman_batch_cuda,
        _cdlharami_batch_cuda, _cdlharamicross_batch_cuda,
        _cdlhighwave_batch_cuda, _cdlhikkake_batch_cuda,
        _cdlhikkakemod_batch_cuda, _cdlhomingpigeon_batch_cuda,
        _cdlidentical3crows_batch_cuda, _cdlinneck_batch_cuda,
        _cdlinvertedhammer_batch_cuda, _cdlkicking_batch_cuda,
        _cdlkickingbylength_batch_cuda, _cdlladderbottom_batch_cuda,
        _cdllongleggeddoji_batch_cuda, _cdllongline_batch_cuda,
        _cdlmarubozu_batch_cuda, _cdlmatchinglow_batch_cuda,
        _cdlmathold_batch_cuda, _cdlmorningdojistar_batch_cuda,
        _cdlmorningstar_batch_cuda, _cdlonneck_batch_cuda,
        _cdlpiercing_batch_cuda, _cdlrickshawman_batch_cuda,
        _cdlrisefall3methods_batch_cuda, _cdlseparatinglines_batch_cuda,
        _cdlshootingstar_batch_cuda, _cdlshortline_batch_cuda,
        _cdlspinningtop_batch_cuda, _cdlstalledpattern_batch_cuda,
        _cdlsticksandwich_batch_cuda, _cdltakuri_batch_cuda,
        _cdltasukigap_batch_cuda, _cdlthrusting_batch_cuda,
        _cdltristar_batch_cuda, _cdlunique3river_batch_cuda,
        _cdlupsidegap2crows_batch_cuda, _cdlxsidegap3methods_batch_cuda,
    )


def _require_cuda():
    if not HAS_CUDA:
        raise RuntimeError(
            "CUDA is not available. Install numba-cuda and ensure a CUDA-capable GPU is present."
        )


def _prepare_input(data, dtype=np.float64):
    """Convert to 2D contiguous float64 array. Accept 1D (single ticker) or 2D."""
    data = np.asarray(data, dtype=dtype)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got {data.ndim}D")
    return np.ascontiguousarray(data)


def _is_device_array(arr):
    """Check if arr is already a Numba CUDA DeviceNDArray."""
    return HAS_CUDA and isinstance(arr, cuda.devicearray.DeviceNDArray)


def _to_device(data, dtype=np.float64):
    """Transfer to GPU if not already there."""
    if _is_device_array(data):
        return data
    return to_device_2d(data, dtype=dtype)


# =============================================================================
# Overlap Studies - Batch
# =============================================================================

def SMA_batch(close, timeperiod=30):
    """Batch SMA: (num_tickers, num_bars) -> (num_tickers, num_bars)"""
    _require_cuda()
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    d_close = _to_device(close)
    d_out = allocate_output_2d(num_tickers, num_bars)
    blocks, tpb = compute_grid_1d(num_tickers)
    _sma_batch_cuda[blocks, tpb](d_close, timeperiod, d_out)
    return d_out.copy_to_host()


def EMA_batch(close, timeperiod=30):
    """Batch EMA: (num_tickers, num_bars) -> (num_tickers, num_bars)"""
    _require_cuda()
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    d_close = _to_device(close)
    d_out = allocate_output_2d(num_tickers, num_bars)
    blocks, tpb = compute_grid_1d(num_tickers)
    _ema_batch_cuda[blocks, tpb](d_close, timeperiod, d_out)
    return d_out.copy_to_host()


def BBANDS_batch(close, timeperiod=5, nbdevup=2.0, nbdevdn=2.0):
    """Batch BBANDS: returns (upper, middle, lower) each (num_tickers, num_bars)"""
    _require_cuda()
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    d_close = _to_device(close)
    d_upper = allocate_output_2d(num_tickers, num_bars)
    d_middle = allocate_output_2d(num_tickers, num_bars)
    d_lower = allocate_output_2d(num_tickers, num_bars)
    blocks, tpb = compute_grid_1d(num_tickers)
    _bbands_batch_cuda[blocks, tpb](d_close, timeperiod, nbdevup, nbdevdn,
                                     d_upper, d_middle, d_lower)
    return d_upper.copy_to_host(), d_middle.copy_to_host(), d_lower.copy_to_host()


def DEMA_batch(close, timeperiod=30):
    """Batch DEMA: (num_tickers, num_bars) -> (num_tickers, num_bars)"""
    _require_cuda()
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    d_close = _to_device(close)
    d_ema1 = allocate_output_2d(num_tickers, num_bars)
    d_out = allocate_output_2d(num_tickers, num_bars)
    blocks, tpb = compute_grid_1d(num_tickers)
    _dema_batch_cuda[blocks, tpb](d_close, timeperiod, d_ema1, d_out)
    return d_out.copy_to_host()


def KAMA_batch(close, timeperiod=30):
    """Batch KAMA: (num_tickers, num_bars) -> (num_tickers, num_bars)"""
    _require_cuda()
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    d_close = _to_device(close)
    d_out = allocate_output_2d(num_tickers, num_bars)
    blocks, tpb = compute_grid_1d(num_tickers)
    _kama_batch_cuda[blocks, tpb](d_close, timeperiod, d_out)
    return d_out.copy_to_host()


def WMA_batch(close, timeperiod=30):
    """Batch WMA: (num_tickers, num_bars) -> (num_tickers, num_bars)"""
    _require_cuda()
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    d_close = _to_device(close)
    d_out = allocate_output_2d(num_tickers, num_bars)
    blocks, tpb = compute_grid_1d(num_tickers)
    _wma_batch_cuda[blocks, tpb](d_close, timeperiod, d_out)
    return d_out.copy_to_host()


def TEMA_batch(close, timeperiod=30):
    """Batch TEMA: (num_tickers, num_bars) -> (num_tickers, num_bars)"""
    _require_cuda()
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    d_close = _to_device(close)
    d_ema1 = allocate_output_2d(num_tickers, num_bars)
    d_ema2 = allocate_output_2d(num_tickers, num_bars)
    d_ema3 = allocate_output_2d(num_tickers, num_bars)
    d_out = allocate_output_2d(num_tickers, num_bars)
    blocks, tpb = compute_grid_1d(num_tickers)
    _tema_batch_cuda[blocks, tpb](d_close, timeperiod, d_ema1, d_ema2, d_ema3, d_out)
    return d_out.copy_to_host()


def T3_batch(close, timeperiod=5, vfactor=0.7):
    """Batch T3: (num_tickers, num_bars) -> (num_tickers, num_bars)"""
    _require_cuda()
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    d_close = _to_device(close)
    d_ema1 = allocate_output_2d(num_tickers, num_bars)
    d_ema2 = allocate_output_2d(num_tickers, num_bars)
    d_ema3 = allocate_output_2d(num_tickers, num_bars)
    d_ema4 = allocate_output_2d(num_tickers, num_bars)
    d_ema5 = allocate_output_2d(num_tickers, num_bars)
    d_ema6 = allocate_output_2d(num_tickers, num_bars)
    d_out = allocate_output_2d(num_tickers, num_bars)
    blocks, tpb = compute_grid_1d(num_tickers)
    _t3_batch_cuda[blocks, tpb](d_close, timeperiod, vfactor,
                                 d_ema1, d_ema2, d_ema3, d_ema4, d_ema5, d_ema6, d_out)
    return d_out.copy_to_host()


def TRIMA_batch(close, timeperiod=30):
    """Batch TRIMA: (num_tickers, num_bars) -> (num_tickers, num_bars)"""
    _require_cuda()
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    d_close = _to_device(close)
    d_sma1 = allocate_output_2d(num_tickers, num_bars)
    d_out = allocate_output_2d(num_tickers, num_bars)
    blocks, tpb = compute_grid_1d(num_tickers)
    _trima_batch_cuda[blocks, tpb](d_close, timeperiod, d_sma1, d_out)
    return d_out.copy_to_host()


def MAMA_batch(close, fastlimit=0.5, slowlimit=0.05):
    """Batch MAMA: returns (mama, fama) each (num_tickers, num_bars)"""
    _require_cuda()
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    d_close = _to_device(close)
    d_mama = allocate_output_2d(num_tickers, num_bars)
    d_fama = allocate_output_2d(num_tickers, num_bars)
    blocks, tpb = compute_grid_1d(num_tickers)
    _mama_batch_cuda[blocks, tpb](d_close, fastlimit, slowlimit, d_mama, d_fama)
    return d_mama.copy_to_host(), d_fama.copy_to_host()


def SAR_batch(high, low, acceleration=0.02, maximum=0.2):
    """Batch SAR: (num_tickers, num_bars) -> (num_tickers, num_bars)"""
    _require_cuda()
    high = _prepare_input(high)
    low = _prepare_input(low)
    num_tickers, num_bars = high.shape
    d_high = _to_device(high)
    d_low = _to_device(low)
    d_out = allocate_output_2d(num_tickers, num_bars)
    blocks, tpb = compute_grid_1d(num_tickers)
    _sar_batch_cuda[blocks, tpb](d_high, d_low, acceleration, maximum, d_out)
    return d_out.copy_to_host()


def SAREXT_batch(high, low, startvalue=0.0, offsetonreverse=0.0,
                 accelerationinit_long=0.02, accelerationlong=0.02, accelerationmax_long=0.2,
                 accelerationinit_short=0.02, accelerationshort=0.02, accelerationmax_short=0.2):
    """Batch SAREXT: (num_tickers, num_bars) -> (num_tickers, num_bars)"""
    _require_cuda()
    high = _prepare_input(high)
    low = _prepare_input(low)
    num_tickers, num_bars = high.shape
    d_high = _to_device(high)
    d_low = _to_device(low)
    d_out = allocate_output_2d(num_tickers, num_bars)
    blocks, tpb = compute_grid_1d(num_tickers)
    _sarext_batch_cuda[blocks, tpb](d_high, d_low,
                                     startvalue, offsetonreverse,
                                     accelerationinit_long, accelerationlong, accelerationmax_long,
                                     accelerationinit_short, accelerationshort, accelerationmax_short,
                                     d_out)
    return d_out.copy_to_host()


# =============================================================================
# Momentum Indicators - Batch
# =============================================================================

def RSI_batch(close, timeperiod=14):
    """Batch RSI: (num_tickers, num_bars) -> (num_tickers, num_bars)"""
    _require_cuda()
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    d_close = _to_device(close)
    d_out = allocate_output_2d(num_tickers, num_bars)
    blocks, tpb = compute_grid_1d(num_tickers)
    _rsi_batch_cuda[blocks, tpb](d_close, timeperiod, d_out)
    return d_out.copy_to_host()


def MACD_batch(close, fastperiod=12, slowperiod=26, signalperiod=9):
    """Batch MACD: returns (macd, signal, histogram) each (num_tickers, num_bars)"""
    _require_cuda()
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    d_close = _to_device(close)
    d_macd = allocate_output_2d(num_tickers, num_bars)
    d_signal = allocate_output_2d(num_tickers, num_bars)
    d_hist = allocate_output_2d(num_tickers, num_bars)
    blocks, tpb = compute_grid_1d(num_tickers)
    _macd_batch_cuda[blocks, tpb](d_close, fastperiod, slowperiod, signalperiod,
                                   d_macd, d_signal, d_hist)
    return d_macd.copy_to_host(), d_signal.copy_to_host(), d_hist.copy_to_host()


def ADX_batch(high, low, close, timeperiod=14):
    """Batch ADX: (num_tickers, num_bars) -> (num_tickers, num_bars)"""
    _require_cuda()
    high, low, close = _prepare_input(high), _prepare_input(low), _prepare_input(close)
    num_tickers, num_bars = high.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _adx_batch_cuda[blocks, tpb](_to_device(high), _to_device(low), _to_device(close), timeperiod, d_out)
    return d_out.copy_to_host()


def ATR_batch(high, low, close, timeperiod=14):
    """Batch ATR: (num_tickers, num_bars) -> (num_tickers, num_bars)"""
    _require_cuda()
    high, low, close = _prepare_input(high), _prepare_input(low), _prepare_input(close)
    num_tickers, num_bars = high.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _atr_batch_cuda[blocks, tpb](_to_device(high), _to_device(low), _to_device(close), timeperiod, d_out)
    return d_out.copy_to_host()


def CCI_batch(high, low, close, timeperiod=14):
    """Batch CCI: (num_tickers, num_bars) -> (num_tickers, num_bars)"""
    _require_cuda()
    high, low, close = _prepare_input(high), _prepare_input(low), _prepare_input(close)
    num_tickers, num_bars = high.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _cci_batch_cuda[blocks, tpb](_to_device(high), _to_device(low), _to_device(close), timeperiod, d_out)
    return d_out.copy_to_host()


def CMO_batch(close, timeperiod=14):
    """Batch CMO: (num_tickers, num_bars) -> (num_tickers, num_bars)"""
    _require_cuda()
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _cmo_batch_cuda[blocks, tpb](_to_device(close), timeperiod, d_out)
    return d_out.copy_to_host()


def DX_batch(high, low, close, timeperiod=14):
    """Batch DX: (num_tickers, num_bars) -> (num_tickers, num_bars)"""
    _require_cuda()
    high, low, close = _prepare_input(high), _prepare_input(low), _prepare_input(close)
    num_tickers, num_bars = high.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _dx_batch_cuda[blocks, tpb](_to_device(high), _to_device(low), _to_device(close), timeperiod, d_out)
    return d_out.copy_to_host()


def MOM_batch(close, timeperiod=10):
    """Batch MOM: (num_tickers, num_bars) -> (num_tickers, num_bars)"""
    _require_cuda()
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _mom_batch_cuda[blocks, tpb](_to_device(close), timeperiod, d_out)
    return d_out.copy_to_host()


def ROC_batch(close, timeperiod=10):
    """Batch ROC"""
    _require_cuda()
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _roc_batch_cuda[blocks, tpb](_to_device(close), timeperiod, d_out)
    return d_out.copy_to_host()


def ROCP_batch(close, timeperiod=10):
    """Batch ROCP"""
    _require_cuda()
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _rocp_batch_cuda[blocks, tpb](_to_device(close), timeperiod, d_out)
    return d_out.copy_to_host()


def ROCR_batch(close, timeperiod=10):
    """Batch ROCR"""
    _require_cuda()
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _rocr_batch_cuda[blocks, tpb](_to_device(close), timeperiod, d_out)
    return d_out.copy_to_host()


def ROCR100_batch(close, timeperiod=10):
    """Batch ROCR100"""
    _require_cuda()
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _rocr100_batch_cuda[blocks, tpb](_to_device(close), timeperiod, d_out)
    return d_out.copy_to_host()


def WILLR_batch(high, low, close, timeperiod=14):
    """Batch WILLR"""
    _require_cuda()
    high, low, close = _prepare_input(high), _prepare_input(low), _prepare_input(close)
    num_tickers, num_bars = high.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _willr_batch_cuda[blocks, tpb](_to_device(high), _to_device(low), _to_device(close), timeperiod, d_out)
    return d_out.copy_to_host()


def STOCH_batch(high, low, close, fastk_period=5, slowk_period=3, slowd_period=3):
    """Batch STOCH: returns (slowk, slowd)"""
    _require_cuda()
    high, low, close = _prepare_input(high), _prepare_input(low), _prepare_input(close)
    num_tickers, num_bars = high.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_fastk = allocate_output_2d(num_tickers, num_bars)
    _stoch_fastk_batch_cuda[blocks, tpb](_to_device(high), _to_device(low), _to_device(close), fastk_period, d_fastk)
    from ..gpu.overlap import _sma_batch_cuda
    d_slowk = allocate_output_2d(num_tickers, num_bars)
    _sma_batch_cuda[blocks, tpb](d_fastk, slowk_period, d_slowk)
    d_slowd = allocate_output_2d(num_tickers, num_bars)
    _sma_batch_cuda[blocks, tpb](d_slowk, slowd_period, d_slowd)
    return d_slowk.copy_to_host(), d_slowd.copy_to_host()


def STOCHF_batch(high, low, close, fastk_period=5, fastd_period=3):
    """Batch STOCHF: returns (fastk, fastd)"""
    _require_cuda()
    high, low, close = _prepare_input(high), _prepare_input(low), _prepare_input(close)
    num_tickers, num_bars = high.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_fastk = allocate_output_2d(num_tickers, num_bars)
    _stoch_fastk_batch_cuda[blocks, tpb](_to_device(high), _to_device(low), _to_device(close), fastk_period, d_fastk)
    from ..gpu.overlap import _sma_batch_cuda
    d_fastd = allocate_output_2d(num_tickers, num_bars)
    _sma_batch_cuda[blocks, tpb](d_fastk, fastd_period, d_fastd)
    return d_fastk.copy_to_host(), d_fastd.copy_to_host()


def BOP_batch(open_, high, low, close):
    """Batch BOP"""
    _require_cuda()
    open_, high = _prepare_input(open_), _prepare_input(high)
    low, close = _prepare_input(low), _prepare_input(close)
    num_tickers, num_bars = open_.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _bop_batch_cuda[blocks, tpb](_to_device(open_), _to_device(high), _to_device(low), _to_device(close), d_out)
    return d_out.copy_to_host()


def AROON_batch(high, low, timeperiod=14):
    """Batch AROON: returns (aroondown, aroonup)"""
    _require_cuda()
    high, low = _prepare_input(high), _prepare_input(low)
    num_tickers, num_bars = high.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_down = allocate_output_2d(num_tickers, num_bars)
    d_up = allocate_output_2d(num_tickers, num_bars)
    _aroon_batch_cuda[blocks, tpb](_to_device(high), _to_device(low), timeperiod, d_down, d_up)
    return d_down.copy_to_host(), d_up.copy_to_host()


def AROONOSC_batch(high, low, timeperiod=14):
    """Batch AROONOSC"""
    _require_cuda()
    high, low = _prepare_input(high), _prepare_input(low)
    num_tickers, num_bars = high.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _aroonosc_batch_cuda[blocks, tpb](_to_device(high), _to_device(low), timeperiod, d_out)
    return d_out.copy_to_host()


def MFI_batch(high, low, close, volume, timeperiod=14):
    """Batch MFI"""
    _require_cuda()
    high, low = _prepare_input(high), _prepare_input(low)
    close, volume = _prepare_input(close), _prepare_input(volume)
    num_tickers, num_bars = high.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _mfi_batch_cuda[blocks, tpb](_to_device(high), _to_device(low), _to_device(close), _to_device(volume), timeperiod, d_out)
    return d_out.copy_to_host()


def MINUS_DM_batch(high, low, timeperiod=14):
    """Batch MINUS_DM"""
    _require_cuda()
    high, low = _prepare_input(high), _prepare_input(low)
    num_tickers, num_bars = high.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _minus_dm_batch_cuda[blocks, tpb](_to_device(high), _to_device(low), timeperiod, d_out)
    return d_out.copy_to_host()


def PLUS_DM_batch(high, low, timeperiod=14):
    """Batch PLUS_DM"""
    _require_cuda()
    high, low = _prepare_input(high), _prepare_input(low)
    num_tickers, num_bars = high.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _plus_dm_batch_cuda[blocks, tpb](_to_device(high), _to_device(low), timeperiod, d_out)
    return d_out.copy_to_host()


def MINUS_DI_batch(high, low, close, timeperiod=14):
    """Batch MINUS_DI"""
    _require_cuda()
    high, low, close = _prepare_input(high), _prepare_input(low), _prepare_input(close)
    num_tickers, num_bars = high.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _minus_di_batch_cuda[blocks, tpb](_to_device(high), _to_device(low), _to_device(close), timeperiod, d_out)
    return d_out.copy_to_host()


def PLUS_DI_batch(high, low, close, timeperiod=14):
    """Batch PLUS_DI"""
    _require_cuda()
    high, low, close = _prepare_input(high), _prepare_input(low), _prepare_input(close)
    num_tickers, num_bars = high.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _plus_di_batch_cuda[blocks, tpb](_to_device(high), _to_device(low), _to_device(close), timeperiod, d_out)
    return d_out.copy_to_host()


def ULTOSC_batch(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28):
    """Batch ULTOSC"""
    _require_cuda()
    high, low, close = _prepare_input(high), _prepare_input(low), _prepare_input(close)
    num_tickers, num_bars = high.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _ultosc_batch_cuda[blocks, tpb](_to_device(high), _to_device(low), _to_device(close),
                                     timeperiod1, timeperiod2, timeperiod3, d_out)
    return d_out.copy_to_host()


# =============================================================================
# Volatility Indicators - Batch
# =============================================================================

def TRANGE_batch(high, low, close):
    """Batch TRANGE"""
    _require_cuda()
    high, low, close = _prepare_input(high), _prepare_input(low), _prepare_input(close)
    num_tickers, num_bars = high.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _trange_batch_cuda[blocks, tpb](_to_device(high), _to_device(low), _to_device(close), d_out)
    return d_out.copy_to_host()


def NATR_batch(high, low, close, timeperiod=14):
    """Batch NATR"""
    _require_cuda()
    high, low, close = _prepare_input(high), _prepare_input(low), _prepare_input(close)
    num_tickers, num_bars = high.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _natr_batch_cuda[blocks, tpb](_to_device(high), _to_device(low), _to_device(close), timeperiod, d_out)
    return d_out.copy_to_host()


# =============================================================================
# Volume Indicators - Batch
# =============================================================================

def AD_batch(high, low, close, volume):
    """Batch AD (Chaikin A/D Line)"""
    _require_cuda()
    high, low = _prepare_input(high), _prepare_input(low)
    close, volume = _prepare_input(close), _prepare_input(volume)
    num_tickers, num_bars = high.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _ad_batch_cuda[blocks, tpb](_to_device(high), _to_device(low), _to_device(close), _to_device(volume), d_out)
    return d_out.copy_to_host()


def OBV_batch(close, volume):
    """Batch OBV"""
    _require_cuda()
    close, volume = _prepare_input(close), _prepare_input(volume)
    num_tickers, num_bars = close.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _obv_batch_cuda[blocks, tpb](_to_device(close), _to_device(volume), d_out)
    return d_out.copy_to_host()


def ADOSC_batch(high, low, close, volume, fastperiod=3, slowperiod=10):
    """Batch ADOSC (Chaikin A/D Oscillator)"""
    _require_cuda()
    high, low = _prepare_input(high), _prepare_input(low)
    close, volume = _prepare_input(close), _prepare_input(volume)
    num_tickers, num_bars = high.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _adosc_batch_cuda[blocks, tpb](_to_device(high), _to_device(low), _to_device(close),
                                    _to_device(volume), fastperiod, slowperiod, d_out)
    return d_out.copy_to_host()


# =============================================================================
# Statistics - Batch
# =============================================================================

def STDDEV_batch(close, timeperiod=5, nbdev=1.0):
    """Batch STDDEV"""
    _require_cuda()
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _stddev_batch_cuda[blocks, tpb](_to_device(close), timeperiod, nbdev, d_out)
    return d_out.copy_to_host()


def VAR_batch(close, timeperiod=5, nbdev=1.0):
    """Batch VAR"""
    _require_cuda()
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _var_batch_cuda[blocks, tpb](_to_device(close), timeperiod, nbdev, d_out)
    return d_out.copy_to_host()


def TSF_batch(close, timeperiod=14):
    """Batch TSF"""
    _require_cuda()
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _tsf_batch_cuda[blocks, tpb](_to_device(close), timeperiod, d_out)
    return d_out.copy_to_host()


# =============================================================================
# Statistic Functions - Batch
# =============================================================================

def BETA_batch(high, low, timeperiod=5):
    """Batch BETA"""
    _require_cuda()
    high, low = _prepare_input(high), _prepare_input(low)
    num_tickers, num_bars = high.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _beta_batch_cuda[blocks, tpb](_to_device(high), _to_device(low), timeperiod, d_out)
    return d_out.copy_to_host()


def CORREL_batch(high, low, timeperiod=30):
    """Batch CORREL"""
    _require_cuda()
    high, low = _prepare_input(high), _prepare_input(low)
    num_tickers, num_bars = high.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _correl_batch_cuda[blocks, tpb](_to_device(high), _to_device(low), timeperiod, d_out)
    return d_out.copy_to_host()


def LINEARREG_batch(close, timeperiod=14):
    """Batch LINEARREG"""
    _require_cuda()
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _linearreg_batch_cuda[blocks, tpb](_to_device(close), timeperiod, d_out)
    return d_out.copy_to_host()


def LINEARREG_ANGLE_batch(close, timeperiod=14):
    """Batch LINEARREG_ANGLE"""
    _require_cuda()
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _linearreg_angle_batch_cuda[blocks, tpb](_to_device(close), timeperiod, d_out)
    return d_out.copy_to_host()


def LINEARREG_INTERCEPT_batch(close, timeperiod=14):
    """Batch LINEARREG_INTERCEPT"""
    _require_cuda()
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _linearreg_intercept_batch_cuda[blocks, tpb](_to_device(close), timeperiod, d_out)
    return d_out.copy_to_host()


def LINEARREG_SLOPE_batch(close, timeperiod=14):
    """Batch LINEARREG_SLOPE"""
    _require_cuda()
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _linearreg_slope_batch_cuda[blocks, tpb](_to_device(close), timeperiod, d_out)
    return d_out.copy_to_host()


# =============================================================================
# Math Operators - Batch
# =============================================================================

def MAX_batch(close, timeperiod=30):
    """Batch MAX"""
    _require_cuda()
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _max_batch_cuda[blocks, tpb](_to_device(close), timeperiod, d_out)
    return d_out.copy_to_host()


def MAXINDEX_batch(close, timeperiod=30):
    """Batch MAXINDEX"""
    _require_cuda()
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _maxindex_batch_cuda[blocks, tpb](_to_device(close), timeperiod, d_out)
    return d_out.copy_to_host()


def MIN_batch(close, timeperiod=30):
    """Batch MIN"""
    _require_cuda()
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _min_batch_cuda[blocks, tpb](_to_device(close), timeperiod, d_out)
    return d_out.copy_to_host()


def MININDEX_batch(close, timeperiod=30):
    """Batch MININDEX"""
    _require_cuda()
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _minindex_batch_cuda[blocks, tpb](_to_device(close), timeperiod, d_out)
    return d_out.copy_to_host()


def MINMAX_batch(close, timeperiod=30):
    """Batch MINMAX: returns (min_out, max_out)"""
    _require_cuda()
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_min = allocate_output_2d(num_tickers, num_bars)
    d_max = allocate_output_2d(num_tickers, num_bars)
    _minmax_batch_cuda[blocks, tpb](_to_device(close), timeperiod, d_min, d_max)
    return d_min.copy_to_host(), d_max.copy_to_host()


def MINMAXINDEX_batch(close, timeperiod=30):
    """Batch MINMAXINDEX: returns (min_idx, max_idx)"""
    _require_cuda()
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_min_idx = allocate_output_2d(num_tickers, num_bars)
    d_max_idx = allocate_output_2d(num_tickers, num_bars)
    _minmaxindex_batch_cuda[blocks, tpb](_to_device(close), timeperiod, d_min_idx, d_max_idx)
    return d_min_idx.copy_to_host(), d_max_idx.copy_to_host()


def SUM_batch(close, timeperiod=30):
    """Batch SUM"""
    _require_cuda()
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _sum_batch_cuda[blocks, tpb](_to_device(close), timeperiod, d_out)
    return d_out.copy_to_host()


# =============================================================================
# Price Transform - Batch
# =============================================================================

def MEDPRICE_batch(high, low):
    """Batch MEDPRICE"""
    _require_cuda()
    high, low = _prepare_input(high), _prepare_input(low)
    num_tickers, num_bars = high.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _medprice_batch_cuda[blocks, tpb](_to_device(high), _to_device(low), d_out)
    return d_out.copy_to_host()


def MIDPOINT_batch(close, timeperiod=14):
    """Batch MIDPOINT"""
    _require_cuda()
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _midpoint_batch_cuda[blocks, tpb](_to_device(close), timeperiod, d_out)
    return d_out.copy_to_host()


def MIDPRICE_batch(high, low, timeperiod=14):
    """Batch MIDPRICE"""
    _require_cuda()
    high, low = _prepare_input(high), _prepare_input(low)
    num_tickers, num_bars = high.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _midprice_batch_cuda[blocks, tpb](_to_device(high), _to_device(low), timeperiod, d_out)
    return d_out.copy_to_host()


def TYPPRICE_batch(high, low, close):
    """Batch TYPPRICE"""
    _require_cuda()
    high, low, close = _prepare_input(high), _prepare_input(low), _prepare_input(close)
    num_tickers, num_bars = high.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _typprice_batch_cuda[blocks, tpb](_to_device(high), _to_device(low), _to_device(close), d_out)
    return d_out.copy_to_host()


def WCLPRICE_batch(high, low, close):
    """Batch WCLPRICE"""
    _require_cuda()
    high, low, close = _prepare_input(high), _prepare_input(low), _prepare_input(close)
    num_tickers, num_bars = high.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _wclprice_batch_cuda[blocks, tpb](_to_device(high), _to_device(low), _to_device(close), d_out)
    return d_out.copy_to_host()


# =============================================================================
# Cycle Indicators - Batch
# =============================================================================

def _allocate_ht_workspace(num_tickers, num_bars):
    """Allocate zeroed workspace arrays for Hilbert Transform kernels."""
    ws = {}
    for name in ('smooth', 'detrender', 'i1', 'q1', 'ji', 'jq', 'i2', 'q2',
                 're', 'im', 'period', 'smooth_period'):
        host = np.zeros((num_tickers, num_bars), dtype=np.float64)
        ws[name] = cuda.to_device(host)
    return ws


def HT_TRENDLINE_batch(close):
    """Batch HT_TRENDLINE"""
    _require_cuda()
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _ht_trendline_batch_cuda[blocks, tpb](_to_device(close), d_out)
    return d_out.copy_to_host()


def HT_TRENDMODE_batch(close):
    """Batch HT_TRENDMODE"""
    _require_cuda()
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _ht_trendmode_batch_cuda[blocks, tpb](_to_device(close), d_out)
    return d_out.copy_to_host()


def HT_DCPERIOD_batch(close):
    """Batch HT_DCPERIOD"""
    _require_cuda()
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    ws = _allocate_ht_workspace(num_tickers, num_bars)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _ht_dcperiod_batch_cuda[blocks, tpb](
        _to_device(close), ws['smooth'], ws['detrender'], ws['i1'], ws['q1'],
        ws['ji'], ws['jq'], ws['i2'], ws['q2'], ws['re'], ws['im'],
        ws['period'], ws['smooth_period'], d_out)
    return d_out.copy_to_host()


def HT_DCPHASE_batch(close):
    """Batch HT_DCPHASE"""
    _require_cuda()
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    ws = _allocate_ht_workspace(num_tickers, num_bars)
    d_out = allocate_output_2d(num_tickers, num_bars)
    _ht_dcphase_batch_cuda[blocks, tpb](
        _to_device(close), ws['smooth'], ws['detrender'], ws['i1'], ws['q1'],
        ws['ji'], ws['jq'], ws['i2'], ws['q2'], ws['period'], d_out)
    return d_out.copy_to_host()


def HT_PHASOR_batch(close):
    """Batch HT_PHASOR: returns (inphase, quadrature)"""
    _require_cuda()
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    ws = _allocate_ht_workspace(num_tickers, num_bars)
    d_inphase = allocate_output_2d(num_tickers, num_bars)
    d_quadrature = allocate_output_2d(num_tickers, num_bars)
    _ht_phasor_batch_cuda[blocks, tpb](
        _to_device(close), ws['smooth'], ws['detrender'], ws['i1'], ws['q1'],
        ws['ji'], ws['jq'], ws['i2'], ws['q2'], ws['period'],
        d_inphase, d_quadrature)
    return d_inphase.copy_to_host(), d_quadrature.copy_to_host()


def HT_SINE_batch(close):
    """Batch HT_SINE: returns (sine, leadsine)"""
    _require_cuda()
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    blocks, tpb = compute_grid_1d(num_tickers)
    ws = _allocate_ht_workspace(num_tickers, num_bars)
    d_sine = allocate_output_2d(num_tickers, num_bars)
    d_leadsine = allocate_output_2d(num_tickers, num_bars)
    _ht_sine_batch_cuda[blocks, tpb](
        _to_device(close), ws['smooth'], ws['detrender'], ws['i1'], ws['q1'],
        ws['ji'], ws['jq'], ws['i2'], ws['q2'], ws['re'], ws['im'],
        ws['period'], ws['smooth_period'], d_sine, d_leadsine)
    return d_sine.copy_to_host(), d_leadsine.copy_to_host()


# --- Candlestick Pattern Recognition ---

def _cdl_batch(kernel, open_, high, low, close):
    """Common helper for standard 4-input candlestick pattern batch functions."""
    _require_cuda()
    open_ = _prepare_input(open_)
    high = _prepare_input(high)
    low = _prepare_input(low)
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    d_output = cuda.to_device(np.zeros((num_tickers, num_bars), dtype=np.float64))
    blocks, tpb = compute_grid_1d(num_tickers)
    kernel[blocks, tpb](_to_device(open_), _to_device(high), _to_device(low), _to_device(close), d_output)
    return d_output.copy_to_host()


def _cdl_batch_pen(kernel, open_, high, low, close, penetration):
    """Helper for candlestick patterns with penetration parameter."""
    _require_cuda()
    open_ = _prepare_input(open_)
    high = _prepare_input(high)
    low = _prepare_input(low)
    close = _prepare_input(close)
    num_tickers, num_bars = close.shape
    d_output = cuda.to_device(np.zeros((num_tickers, num_bars), dtype=np.float64))
    blocks, tpb = compute_grid_1d(num_tickers)
    kernel[blocks, tpb](_to_device(open_), _to_device(high), _to_device(low), _to_device(close), penetration, d_output)
    return d_output.copy_to_host()


def CDL2CROWS_batch(open_, high, low, close):
    return _cdl_batch(_cdl2crows_batch_cuda, open_, high, low, close)

def CDL3BLACKCROWS_batch(open_, high, low, close):
    return _cdl_batch(_cdl3blackcrows_batch_cuda, open_, high, low, close)

def CDL3INSIDE_batch(open_, high, low, close):
    return _cdl_batch(_cdl3inside_batch_cuda, open_, high, low, close)

def CDL3OUTSIDE_batch(open_, high, low, close):
    return _cdl_batch(_cdl3outside_batch_cuda, open_, high, low, close)

def CDL3STARSINSOUTH_batch(open_, high, low, close):
    return _cdl_batch(_cdl3starsinsouth_batch_cuda, open_, high, low, close)

def CDL3WHITESOLDIERS_batch(open_, high, low, close):
    return _cdl_batch(_cdl3whitesoldiers_batch_cuda, open_, high, low, close)

def CDLABANDONEDBABY_batch(open_, high, low, close):
    return _cdl_batch(_cdlabandonedbaby_batch_cuda, open_, high, low, close)

def CDLADVANCEBLOCK_batch(open_, high, low, close):
    return _cdl_batch(_cdladvanceblock_batch_cuda, open_, high, low, close)

def CDLBELTHOLD_batch(open_, high, low, close):
    return _cdl_batch(_cdlbelthold_batch_cuda, open_, high, low, close)

def CDLBREAKAWAY_batch(open_, high, low, close):
    return _cdl_batch(_cdlbreakaway_batch_cuda, open_, high, low, close)

def CDLCLOSINGMARUBOZU_batch(open_, high, low, close):
    return _cdl_batch(_cdlclosingmarubozu_batch_cuda, open_, high, low, close)

def CDLCONCEALBABYSWALL_batch(open_, high, low, close):
    return _cdl_batch(_cdlconcealbabyswall_batch_cuda, open_, high, low, close)

def CDLCOUNTERATTACK_batch(open_, high, low, close):
    return _cdl_batch(_cdlcounterattack_batch_cuda, open_, high, low, close)

def CDLDARKCLOUDCOVER_batch(open_, high, low, close):
    return _cdl_batch(_cdldarkcloudcover_batch_cuda, open_, high, low, close)

def CDLDOJI_batch(open_, high, low, close):
    return _cdl_batch(_cdldoji_batch_cuda, open_, high, low, close)

def CDLDOJISTAR_batch(open_, high, low, close):
    return _cdl_batch(_cdldojistar_batch_cuda, open_, high, low, close)

def CDLDRAGONFLYDOJI_batch(open_, high, low, close):
    return _cdl_batch(_cdldragonflydoji_batch_cuda, open_, high, low, close)

def CDLENGULFING_batch(open_, high, low, close):
    return _cdl_batch(_cdlengulfing_batch_cuda, open_, high, low, close)

def CDLEVENINGDOJISTAR_batch(open_, high, low, close):
    return _cdl_batch(_cdleveningdojistar_batch_cuda, open_, high, low, close)

def CDLEVENINGSTAR_batch(open_, high, low, close):
    return _cdl_batch(_cdleveningstar_batch_cuda, open_, high, low, close)

def CDLGAPSIDESIDEWHITE_batch(open_, high, low, close):
    return _cdl_batch(_cdlgapsidesidewhite_batch_cuda, open_, high, low, close)

def CDLGRAVESTONEDOJI_batch(open_, high, low, close):
    return _cdl_batch(_cdlgravestonedoji_batch_cuda, open_, high, low, close)

def CDLHAMMER_batch(open_, high, low, close):
    return _cdl_batch(_cdlhammer_batch_cuda, open_, high, low, close)

def CDLHANGINGMAN_batch(open_, high, low, close):
    return _cdl_batch(_cdlhangingman_batch_cuda, open_, high, low, close)

def CDLHARAMI_batch(open_, high, low, close):
    return _cdl_batch(_cdlharami_batch_cuda, open_, high, low, close)

def CDLHARAMICROSS_batch(open_, high, low, close):
    return _cdl_batch(_cdlharamicross_batch_cuda, open_, high, low, close)

def CDLHIGHWAVE_batch(open_, high, low, close):
    return _cdl_batch(_cdlhighwave_batch_cuda, open_, high, low, close)

def CDLHIKKAKE_batch(open_, high, low, close):
    return _cdl_batch(_cdlhikkake_batch_cuda, open_, high, low, close)

def CDLHIKKAKEMOD_batch(open_, high, low, close):
    return _cdl_batch(_cdlhikkakemod_batch_cuda, open_, high, low, close)

def CDLHOMINGPIGEON_batch(open_, high, low, close):
    return _cdl_batch(_cdlhomingpigeon_batch_cuda, open_, high, low, close)

def CDLIDENTICAL3CROWS_batch(open_, high, low, close):
    return _cdl_batch(_cdlidentical3crows_batch_cuda, open_, high, low, close)

def CDLINNECK_batch(open_, high, low, close):
    return _cdl_batch(_cdlinneck_batch_cuda, open_, high, low, close)

def CDLINVERTEDHAMMER_batch(open_, high, low, close):
    return _cdl_batch(_cdlinvertedhammer_batch_cuda, open_, high, low, close)

def CDLKICKING_batch(open_, high, low, close):
    return _cdl_batch(_cdlkicking_batch_cuda, open_, high, low, close)

def CDLKICKINGBYLENGTH_batch(open_, high, low, close):
    return _cdl_batch(_cdlkickingbylength_batch_cuda, open_, high, low, close)

def CDLLADDERBOTTOM_batch(open_, high, low, close):
    return _cdl_batch(_cdlladderbottom_batch_cuda, open_, high, low, close)

def CDLLONGLEGGEDDOJI_batch(open_, high, low, close):
    return _cdl_batch(_cdllongleggeddoji_batch_cuda, open_, high, low, close)

def CDLLONGLINE_batch(open_, high, low, close):
    return _cdl_batch(_cdllongline_batch_cuda, open_, high, low, close)

def CDLMARUBOZU_batch(open_, high, low, close):
    return _cdl_batch(_cdlmarubozu_batch_cuda, open_, high, low, close)

def CDLMATCHINGLOW_batch(open_, high, low, close):
    return _cdl_batch(_cdlmatchinglow_batch_cuda, open_, high, low, close)

def CDLMATHOLD_batch(open_, high, low, close, penetration=0.5):
    return _cdl_batch_pen(_cdlmathold_batch_cuda, open_, high, low, close, penetration)

def CDLMORNINGDOJISTAR_batch(open_, high, low, close, penetration=0.3):
    return _cdl_batch_pen(_cdlmorningdojistar_batch_cuda, open_, high, low, close, penetration)

def CDLMORNINGSTAR_batch(open_, high, low, close, penetration=0.3):
    return _cdl_batch_pen(_cdlmorningstar_batch_cuda, open_, high, low, close, penetration)

def CDLONNECK_batch(open_, high, low, close):
    return _cdl_batch(_cdlonneck_batch_cuda, open_, high, low, close)

def CDLPIERCING_batch(open_, high, low, close):
    return _cdl_batch(_cdlpiercing_batch_cuda, open_, high, low, close)

def CDLRICKSHAWMAN_batch(open_, high, low, close):
    return _cdl_batch(_cdlrickshawman_batch_cuda, open_, high, low, close)

def CDLRISEFALL3METHODS_batch(open_, high, low, close):
    return _cdl_batch(_cdlrisefall3methods_batch_cuda, open_, high, low, close)

def CDLSEPARATINGLINES_batch(open_, high, low, close):
    return _cdl_batch(_cdlseparatinglines_batch_cuda, open_, high, low, close)

def CDLSHOOTINGSTAR_batch(open_, high, low, close):
    return _cdl_batch(_cdlshootingstar_batch_cuda, open_, high, low, close)

def CDLSHORTLINE_batch(open_, high, low, close):
    return _cdl_batch(_cdlshortline_batch_cuda, open_, high, low, close)

def CDLSPINNINGTOP_batch(open_, high, low, close):
    return _cdl_batch(_cdlspinningtop_batch_cuda, open_, high, low, close)

def CDLSTALLEDPATTERN_batch(open_, high, low, close):
    return _cdl_batch(_cdlstalledpattern_batch_cuda, open_, high, low, close)

def CDLSTICKSANDWICH_batch(open_, high, low, close):
    return _cdl_batch(_cdlsticksandwich_batch_cuda, open_, high, low, close)

def CDLTAKURI_batch(open_, high, low, close):
    return _cdl_batch(_cdltakuri_batch_cuda, open_, high, low, close)

def CDLTASUKIGAP_batch(open_, high, low, close):
    return _cdl_batch(_cdltasukigap_batch_cuda, open_, high, low, close)

def CDLTHRUSTING_batch(open_, high, low, close):
    return _cdl_batch(_cdlthrusting_batch_cuda, open_, high, low, close)

def CDLTRISTAR_batch(open_, high, low, close):
    return _cdl_batch(_cdltristar_batch_cuda, open_, high, low, close)

def CDLUNIQUE3RIVER_batch(open_, high, low, close):
    return _cdl_batch(_cdlunique3river_batch_cuda, open_, high, low, close)

def CDLUPSIDEGAP2CROWS_batch(open_, high, low, close):
    return _cdl_batch(_cdlupsidegap2crows_batch_cuda, open_, high, low, close)

def CDLXSIDEGAP3METHODS_batch(open_, high, low, close):
    return _cdl_batch(_cdlxsidegap3methods_batch_cuda, open_, high, low, close)
