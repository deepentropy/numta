"""
talib-pure: Pure Python TA-Lib library with focus on performance
"""

from .overlap import SMA, EMA, BBANDS
from .volume_indicators import AD, ADOSC
from .momentum_indicators import ADX, ADXR, APO, AROON, AROONOSC, ATR, BOP, CCI
from .statistic_functions import BETA
from .pattern_recognition import CDL2CROWS, CDL3BLACKCROWS, CDL3INSIDE

__version__ = "0.1.0"
__all__ = ["SMA", "EMA", "BBANDS", "AD", "ADOSC", "ADX", "ADXR", "APO", "AROON", "AROONOSC", "ATR", "BETA", "BOP", "CCI", "CDL2CROWS", "CDL3BLACKCROWS", "CDL3INSIDE"]
