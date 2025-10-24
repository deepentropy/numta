"""
talib-pure: Pure Python TA-Lib library with focus on performance

Supports both CPU (Numba) and GPU (CuPy/CUDA) backends for accelerated computation.
"""

from .overlap import SMA, EMA, BBANDS
from .volume_indicators import AD, ADOSC
from .momentum_indicators import ADX, ADXR, APO, AROON, AROONOSC, ATR, BOP, CCI
from .statistic_functions import BETA
from .pattern_recognition import CDL2CROWS, CDL3BLACKCROWS, CDL3INSIDE, CDL3OUTSIDE, CDL3STARSINSOUTH, CDL3WHITESOLDIERS, CDLABANDONEDBABY, CDLADVANCEBLOCK, CDLBELTHOLD, CDLBREAKAWAY, CDLCLOSINGMARUBOZU, CDLCONCEALBABYSWALL, CDLCOUNTERATTACK

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
    "SMA", "EMA", "BBANDS",
    "AD", "ADOSC",
    "ADX", "ADXR", "APO", "AROON", "AROONOSC", "ATR", "BOP", "CCI",
    "BETA",
    "CDL2CROWS", "CDL3BLACKCROWS", "CDL3INSIDE", "CDL3OUTSIDE", "CDL3STARSINSOUTH", "CDL3WHITESOLDIERS", "CDLABANDONEDBABY", "CDLADVANCEBLOCK", "CDLBELTHOLD", "CDLBREAKAWAY", "CDLCLOSINGMARUBOZU", "CDLCONCEALBABYSWALL", "CDLCOUNTERATTACK",
    # Backend configuration
    "set_backend",
    "get_backend",
    "is_gpu_available",
    "get_backend_info"
]
