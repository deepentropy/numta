"""
talib-pure: Pure Python TA-Lib library with focus on performance

Supports both CPU (Numba) and GPU (CuPy/CUDA) backends for accelerated computation.
"""

# Add optimized versions to __all__ if available
if HAS_NUMBA:
    __all__.append("SMA_numba")

if HAS_CUPY:
    __all__.append("SMA_gpu")
from .overlap import SMA, EMA, DEMA, BBANDS, KAMA, MA, MAMA, SAR, SAREXT, T3, TEMA, TRIMA, WMA
from .statistics import STDDEV, TSF, VAR
from .volatility_indicators import NATR, TRANGE
from .volume_indicators import AD, OBV, ADOSC
from .momentum_indicators import ADX, ADXR, APO, AROON, AROONOSC, ATR, BOP, CCI, CMO, DX, MACD, MACDEXT, MACDFIX, MFI, MINUS_DI, MINUS_DM, MOM, PLUS_DI, PLUS_DM, PPO, ROC, ROCP, ROCR, ROCR100, RSI, STOCH, STOCHF, STOCHRSI, TRIX, ULTOSC, WILLR
from .cycle_indicators import HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, HT_SINE, HT_TRENDLINE, HT_TRENDMODE
from .statistic_functions import CORREL, LINEARREG, LINEARREG_ANGLE, LINEARREG_INTERCEPT, LINEARREG_SLOPE
from .math_operators import MAX, MAXINDEX, MIN, MININDEX, MINMAX, MINMAXINDEX, SUM
from .price_transform import MEDPRICE, MIDPOINT, MIDPRICE, TYPPRICE, WCLPRICE
from .pattern_recognition import (
    CDLMARUBOZU, CDLMATCHINGLOW, CDLMATHOLD,
    CDLMORNINGDOJISTAR, CDLMORNINGSTAR, CDLONNECK, CDLPIERCING,
    CDLRICKSHAWMAN, CDLRISEFALL3METHODS, CDLSEPARATINGLINES, CDLSHOOTINGSTAR,
    CDLSHORTLINE, CDLSPINNINGTOP, CDLSTALLEDPATTERN, CDLSTICKSANDWICH, CDLTAKURI, CDLTASUKIGAP,
    CDLTHRUSTING, CDLTRISTAR, CDLUNIQUE3RIVER, CDLUPSIDEGAP2CROWS, CDLXSIDEGAP3METHODS
)

# Backend configuration
from .backend import (
    set_backend,
    get_backend,
    is_gpu_available,
    get_backend_info
)

__version__ = "0.1.0"

__all__ = [
    # Indicators
    "SMA", "EMA", "DEMA", "BBANDS", "KAMA", "MA", "MAMA", "SAR", "SAREXT", "T3", "TEMA", "TRIMA", "WMA",
    "AD", "OBV", "ADOSC",
    "ADX", "ADXR", "APO", "AROON", "AROONOSC", "ATR", "BOP", "CCI", "CMO", "DX", "MACD", "MACDEXT", "MACDFIX", "MFI", "MINUS_DI", "MINUS_DM", "MOM", "NATR", "PLUS_DI", "PLUS_DM", "PPO", "ROC", "ROCP", "ROCR", "ROCR100", "RSI", "STDDEV", "STOCH", "STOCHF", "STOCHRSI", "TRANGE", "TRIX", "TSF", "ULTOSC", "VAR", "WILLR",
    "HT_DCPERIOD", "HT_DCPHASE", "HT_PHASOR", "HT_SINE", "HT_TRENDLINE", "HT_TRENDMODE",
    "CORREL", "LINEARREG", "LINEARREG_ANGLE", "LINEARREG_INTERCEPT", "LINEARREG_SLOPE",
    "MAX", "MAXINDEX", "MIN", "MININDEX", "MINMAX", "MINMAXINDEX", "SUM",
    "MEDPRICE", "MIDPOINT", "MIDPRICE", "TYPPRICE", "WCLPRICE",
    # Pattern Recognition
    "CDLMARUBOZU", "CDLMATCHINGLOW", "CDLMATHOLD",
    "CDLMORNINGDOJISTAR", "CDLMORNINGSTAR", "CDLONNECK", "CDLPIERCING",
    "CDLRICKSHAWMAN", "CDLRISEFALL3METHODS", "CDLSEPARATINGLINES", "CDLSHOOTINGSTAR",
    "CDLSHORTLINE", "CDLSPINNINGTOP", "CDLSTALLEDPATTERN", "CDLSTICKSANDWICH", "CDLTAKURI", "CDLTASUKIGAP",
    "CDLTHRUSTING", "CDLTRISTAR", "CDLUNIQUE3RIVER", "CDLUPSIDEGAP2CROWS", "CDLXSIDEGAP3METHODS",
    # Backend configuration
    "set_backend",
    "get_backend",
    "is_gpu_available",
    "get_backend_info"
]
