"""
numta: Pure Python TA-Lib library with focus on performance

Optimized with Numba JIT compilation for fast CPU computation.
"""
# Disable numba JIT caching on Windows to prevent segfaults from cache corruption.
# This must happen before any numba imports. See: https://github.com/deepentropy/numta/issues/31
import os as _os
import sys as _sys
if _sys.platform == "win32" and "NUMBA_DISABLE_CACHING" not in _os.environ:
    _os.environ["NUMBA_DISABLE_CACHING"] = "1"

# Import from API layer
from .api.overlap import SMA, EMA, DEMA, BBANDS, KAMA, MA, MAMA, SAR, SAREXT, T3, TEMA, TRIMA, WMA
from .api.statistics import STDDEV, TSF, VAR
from .api.volatility_indicators import NATR, TRANGE
from .api.volume_indicators import AD, OBV, ADOSC
from .api.momentum_indicators import ADX, ADXR, APO, AROON, AROONOSC, ATR, BOP, CCI, CMO, DX, MACD, MACDEXT, MACDFIX, MFI, MINUS_DI, MINUS_DM, MOM, PLUS_DI, PLUS_DM, PPO, ROC, ROCP, ROCR, ROCR100, RSI, STOCH, STOCHF, STOCHRSI, TRIX, ULTOSC, WILLR
from .api.cycle_indicators import HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, HT_SINE, HT_TRENDLINE, HT_TRENDMODE
from .api.statistic_functions import BETA, CORREL, LINEARREG, LINEARREG_ANGLE, LINEARREG_INTERCEPT, LINEARREG_SLOPE
from .api.math_operators import MAX, MAXINDEX, MIN, MININDEX, MINMAX, MINMAXINDEX, SUM
from .api.price_transform import MEDPRICE, MIDPOINT, MIDPRICE, TYPPRICE, WCLPRICE
from .api.pattern_recognition import (
    CDL2CROWS, CDL3BLACKCROWS, CDL3INSIDE, CDL3OUTSIDE, CDL3STARSINSOUTH, CDL3WHITESOLDIERS,
    CDLABANDONEDBABY, CDLADVANCEBLOCK, CDLBELTHOLD, CDLBREAKAWAY,
    CDLCLOSINGMARUBOZU, CDLCONCEALBABYSWALL, CDLCOUNTERATTACK, CDLDARKCLOUDCOVER, CDLDOJI, CDLDOJISTAR,
    CDLDRAGONFLYDOJI, CDLENGULFING, CDLEVENINGDOJISTAR, CDLEVENINGSTAR, CDLGAPSIDESIDEWHITE, CDLGRAVESTONEDOJI,
    CDLHAMMER, CDLHANGINGMAN, CDLHARAMI, CDLHARAMICROSS, CDLHIGHWAVE,
    CDLHIKKAKE, CDLHIKKAKEMOD, CDLHOMINGPIGEON, CDLIDENTICAL3CROWS, CDLINNECK,
    CDLINVERTEDHAMMER, CDLKICKING, CDLKICKINGBYLENGTH, CDLLADDERBOTTOM, CDLLONGLEGGEDDOJI, CDLLONGLINE,
    CDLMARUBOZU, CDLMATCHINGLOW, CDLMATHOLD,
    CDLMORNINGDOJISTAR, CDLMORNINGSTAR, CDLONNECK, CDLPIERCING,
    CDLRICKSHAWMAN, CDLRISEFALL3METHODS, CDLSEPARATINGLINES, CDLSHOOTINGSTAR,
    CDLSHORTLINE, CDLSPINNINGTOP, CDLSTALLEDPATTERN, CDLSTICKSANDWICH, CDLTAKURI, CDLTASUKIGAP,
    CDLTHRUSTING, CDLTRISTAR, CDLUNIQUE3RIVER, CDLUPSIDEGAP2CROWS, CDLXSIDEGAP3METHODS
)

# Chart Pattern Recognition
from .patterns import (
    # Swing detection
    find_swing_highs,
    find_swing_lows,
    find_swing_points,
    get_swing_high_indices,
    get_swing_low_indices,
    # Utilities
    fibonacci_retracement,
    fibonacci_extension,
    fit_trendline,
    # Chart pattern dataclasses
    HeadShouldersPattern,
    DoublePattern,
    TriplePattern,
    TrianglePattern,
    WedgePattern,
    FlagPattern,
    VCPPattern,
    # Chart pattern detection
    detect_head_shoulders,
    detect_inverse_head_shoulders,
    detect_double_top,
    detect_double_bottom,
    detect_triple_top,
    detect_triple_bottom,
    detect_triangle,
    detect_wedge,
    detect_flag,
    detect_vcp,
    # Harmonic patterns
    HarmonicPattern,
    detect_gartley,
    detect_butterfly,
    detect_bat,
    detect_crab,
    detect_harmonic_patterns,
)

# Backend configuration
from .backend import (
    set_backend,
    get_backend,
    get_backend_info
)

# Optimized implementations
from .optimized import (
    SMA_cumsum,
    SMA_auto,
    get_available_backends,
    HAS_NUMBA
)

# Conditional imports for optimized variants
if HAS_NUMBA:
    from .optimized import SMA_numba

# Version is automatically managed by setuptools-scm from git tags
try:
    from ._version import __version__
except ImportError:
    # Fallback for development installations without setuptools-scm
    __version__ = "0.0.0+unknown"

__all__ = [
    # Indicators
    "SMA", "EMA", "DEMA", "BBANDS", "KAMA", "MA", "MAMA", "SAR", "SAREXT", "T3", "TEMA", "TRIMA", "WMA",
    "AD", "OBV", "ADOSC",
    "ADX", "ADXR", "APO", "AROON", "AROONOSC", "ATR", "BOP", "CCI", "CMO", "DX", "MACD", "MACDEXT", "MACDFIX", "MFI", "MINUS_DI", "MINUS_DM", "MOM", "NATR", "PLUS_DI", "PLUS_DM", "PPO", "ROC", "ROCP", "ROCR", "ROCR100", "RSI", "STDDEV", "STOCH", "STOCHF", "STOCHRSI", "TRANGE", "TRIX", "TSF", "ULTOSC", "VAR", "WILLR",
    "HT_DCPERIOD", "HT_DCPHASE", "HT_PHASOR", "HT_SINE", "HT_TRENDLINE", "HT_TRENDMODE",
    "BETA", "CORREL", "LINEARREG", "LINEARREG_ANGLE", "LINEARREG_INTERCEPT", "LINEARREG_SLOPE",
    "MAX", "MAXINDEX", "MIN", "MININDEX", "MINMAX", "MINMAXINDEX", "SUM",
    "MEDPRICE", "MIDPOINT", "MIDPRICE", "TYPPRICE", "WCLPRICE",
    # Candlestick Pattern Recognition
    "CDL2CROWS", "CDL3BLACKCROWS", "CDL3INSIDE", "CDL3OUTSIDE", "CDL3STARSINSOUTH", "CDL3WHITESOLDIERS",
    "CDLABANDONEDBABY", "CDLADVANCEBLOCK", "CDLBELTHOLD", "CDLBREAKAWAY",
    "CDLCLOSINGMARUBOZU", "CDLCONCEALBABYSWALL", "CDLCOUNTERATTACK", "CDLDARKCLOUDCOVER", "CDLDOJI", "CDLDOJISTAR",
    "CDLDRAGONFLYDOJI", "CDLENGULFING", "CDLEVENINGDOJISTAR", "CDLEVENINGSTAR", "CDLGAPSIDESIDEWHITE", "CDLGRAVESTONEDOJI",
    "CDLHAMMER", "CDLHANGINGMAN", "CDLHARAMI", "CDLHARAMICROSS", "CDLHIGHWAVE",
    "CDLHIKKAKE", "CDLHIKKAKEMOD", "CDLHOMINGPIGEON", "CDLIDENTICAL3CROWS", "CDLINNECK",
    "CDLINVERTEDHAMMER", "CDLKICKING", "CDLKICKINGBYLENGTH", "CDLLADDERBOTTOM", "CDLLONGLEGGEDDOJI", "CDLLONGLINE",
    "CDLMARUBOZU", "CDLMATCHINGLOW", "CDLMATHOLD",
    "CDLMORNINGDOJISTAR", "CDLMORNINGSTAR", "CDLONNECK", "CDLPIERCING",
    "CDLRICKSHAWMAN", "CDLRISEFALL3METHODS", "CDLSEPARATINGLINES", "CDLSHOOTINGSTAR",
    "CDLSHORTLINE", "CDLSPINNINGTOP", "CDLSTALLEDPATTERN", "CDLSTICKSANDWICH", "CDLTAKURI", "CDLTASUKIGAP",
    "CDLTHRUSTING", "CDLTRISTAR", "CDLUNIQUE3RIVER", "CDLUPSIDEGAP2CROWS", "CDLXSIDEGAP3METHODS",
    # Chart Pattern Recognition
    "find_swing_highs", "find_swing_lows", "find_swing_points",
    "get_swing_high_indices", "get_swing_low_indices",
    "fibonacci_retracement", "fibonacci_extension", "fit_trendline",
    "HeadShouldersPattern", "DoublePattern", "TriplePattern", "TrianglePattern", "WedgePattern", "FlagPattern", "VCPPattern",
    "detect_head_shoulders", "detect_inverse_head_shoulders",
    "detect_double_top", "detect_double_bottom",
    "detect_triple_top", "detect_triple_bottom",
    "detect_triangle", "detect_wedge", "detect_flag", "detect_vcp",
    # Harmonic patterns
    "HarmonicPattern",
    "detect_gartley", "detect_butterfly", "detect_bat", "detect_crab", "detect_harmonic_patterns",
    # Backend configuration
    "set_backend",
    "get_backend",
    "get_backend_info",
    # Optimized implementations
    "SMA_cumsum",
    "SMA_auto",
    "get_available_backends",
    "HAS_NUMBA",
]

# Add optimized versions to __all__ if available
if HAS_NUMBA:
    __all__.append("SMA_numba")

# Streaming indicators (lazy import to avoid circular imports)
try:
    from . import streaming
    __all__.append("streaming")
except ImportError:
    pass

# Register pandas DataFrame extension accessor (if pandas is available)
from . import pandas_ext  # noqa: F401
