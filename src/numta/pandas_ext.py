"""
Pandas DataFrame extension accessor for numta technical analysis indicators.

This module provides a `.ta` accessor on pandas DataFrames for seamless
calculation and appending of technical indicators.

Usage:
    import pandas as pd
    import numta  # Auto-registers the .ta accessor

    df = pd.read_csv("prices.csv")
    
    # Calculate and return Series
    sma_series = df.ta.sma(timeperiod=20)
    
    # Calculate and append to DataFrame
    df.ta.sma(timeperiod=20, append=True)  # Adds column 'SMA_20'
"""

from typing import Optional, Union, List, Tuple
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


if HAS_PANDAS:
    
    @pd.api.extensions.register_dataframe_accessor("ta")
    class TAAccessor:
        """
        Pandas DataFrame accessor for technical analysis indicators.
        
        Provides access to all numta indicators through a `.ta` namespace
        on pandas DataFrames.
        """
        
        # OHLCV column name variations (case-insensitive matching)
        _OPEN_NAMES = ['open', 'o']
        _HIGH_NAMES = ['high', 'h']
        _LOW_NAMES = ['low', 'l']
        _CLOSE_NAMES = ['close', 'c', 'adj close', 'adj_close', 'adjclose']
        _VOLUME_NAMES = ['volume', 'v', 'vol']
        
        def __init__(self, pandas_obj: pd.DataFrame):
            self._obj = pandas_obj
            self._validate()
            self._detect_ohlcv_columns()
        
        def _validate(self):
            """Ensure we have a valid DataFrame."""
            if not isinstance(self._obj, pd.DataFrame):
                raise AttributeError("Can only use .ta accessor with a DataFrame")
        
        def _detect_ohlcv_columns(self):
            """Auto-detect OHLCV columns (case-insensitive)."""
            columns_lower = {col.lower(): col for col in self._obj.columns}
            
            self._open_col = None
            self._high_col = None
            self._low_col = None
            self._close_col = None
            self._volume_col = None
            
            for name in self._OPEN_NAMES:
                if name in columns_lower:
                    self._open_col = columns_lower[name]
                    break
            
            for name in self._HIGH_NAMES:
                if name in columns_lower:
                    self._high_col = columns_lower[name]
                    break
            
            for name in self._LOW_NAMES:
                if name in columns_lower:
                    self._low_col = columns_lower[name]
                    break
            
            for name in self._CLOSE_NAMES:
                if name in columns_lower:
                    self._close_col = columns_lower[name]
                    break
            
            for name in self._VOLUME_NAMES:
                if name in columns_lower:
                    self._volume_col = columns_lower[name]
                    break
        
        def _get_column(self, name: str, column: Optional[str] = None) -> np.ndarray:
            """Get column data as numpy array."""
            if column is not None:
                if column not in self._obj.columns:
                    raise ValueError(f"Column '{column}' not found in DataFrame")
                return self._obj[column].values.astype(np.float64)
            
            col_map = {
                'open': self._open_col,
                'high': self._high_col,
                'low': self._low_col,
                'close': self._close_col,
                'volume': self._volume_col
            }
            
            detected_col = col_map.get(name)
            if detected_col is None:
                raise ValueError(
                    f"Could not auto-detect '{name}' column. "
                    f"Please specify using the 'column' parameter."
                )
            
            return self._obj[detected_col].values.astype(np.float64)
        
        def _append_or_return(
            self,
            result: Union[np.ndarray, Tuple[np.ndarray, ...]],
            column_names: Union[str, List[str]],
            append: bool
        ) -> Optional[Union[pd.Series, pd.DataFrame]]:
            """Either append result to DataFrame or return as Series/DataFrame."""
            if isinstance(column_names, str):
                column_names = [column_names]
                result = (result,) if not isinstance(result, tuple) else result
            
            if not isinstance(result, tuple):
                result = (result,)
            
            if append:
                for name, data in zip(column_names, result):
                    self._obj[name] = data
                return None
            else:
                if len(column_names) == 1:
                    return pd.Series(result[0], index=self._obj.index, name=column_names[0])
                else:
                    return pd.DataFrame(
                        {name: data for name, data in zip(column_names, result)},
                        index=self._obj.index
                    )

        # =====================================================================
        # OVERLAP STUDIES
        # =====================================================================
        
        def sma(self, timeperiod: int = 30, column: Optional[str] = None, 
                append: bool = False) -> Optional[pd.Series]:
            """Simple Moving Average."""
            from . import SMA
            close = self._get_column('close', column)
            result = SMA(close, timeperiod=timeperiod)
            return self._append_or_return(result, f"SMA_{timeperiod}", append)
        
        def ema(self, timeperiod: int = 30, column: Optional[str] = None,
                append: bool = False) -> Optional[pd.Series]:
            """Exponential Moving Average."""
            from . import EMA
            close = self._get_column('close', column)
            result = EMA(close, timeperiod=timeperiod)
            return self._append_or_return(result, f"EMA_{timeperiod}", append)
        
        def bbands(self, timeperiod: int = 5, nbdevup: float = 2.0, 
                   nbdevdn: float = 2.0, matype: int = 0,
                   column: Optional[str] = None, append: bool = False) -> Optional[pd.DataFrame]:
            """Bollinger Bands."""
            from . import BBANDS
            close = self._get_column('close', column)
            upper, middle, lower = BBANDS(close, timeperiod=timeperiod, 
                                          nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)
            col_names = [f"BBU_{timeperiod}_{nbdevup}", f"BBM_{timeperiod}", f"BBL_{timeperiod}_{nbdevdn}"]
            return self._append_or_return((upper, middle, lower), col_names, append)
        
        def dema(self, timeperiod: int = 30, column: Optional[str] = None,
                 append: bool = False) -> Optional[pd.Series]:
            """Double Exponential Moving Average."""
            from . import DEMA
            close = self._get_column('close', column)
            result = DEMA(close, timeperiod=timeperiod)
            return self._append_or_return(result, f"DEMA_{timeperiod}", append)
        
        def kama(self, timeperiod: int = 30, column: Optional[str] = None,
                 append: bool = False) -> Optional[pd.Series]:
            """Kaufman Adaptive Moving Average."""
            from . import KAMA
            close = self._get_column('close', column)
            result = KAMA(close, timeperiod=timeperiod)
            return self._append_or_return(result, f"KAMA_{timeperiod}", append)
        
        def ma(self, timeperiod: int = 30, matype: int = 0,
               column: Optional[str] = None, append: bool = False) -> Optional[pd.Series]:
            """Moving Average."""
            from . import MA
            close = self._get_column('close', column)
            result = MA(close, timeperiod=timeperiod, matype=matype)
            return self._append_or_return(result, f"MA_{timeperiod}_{matype}", append)
        
        def mama(self, fastlimit: float = 0.5, slowlimit: float = 0.05,
                 column: Optional[str] = None, append: bool = False) -> Optional[pd.DataFrame]:
            """MESA Adaptive Moving Average."""
            from . import MAMA
            close = self._get_column('close', column)
            mama_result, fama_result = MAMA(close, fastlimit=fastlimit, slowlimit=slowlimit)
            col_names = [f"MAMA_{fastlimit}_{slowlimit}", f"FAMA_{fastlimit}_{slowlimit}"]
            return self._append_or_return((mama_result, fama_result), col_names, append)
        
        def sar(self, acceleration: float = 0.02, maximum: float = 0.2,
                append: bool = False) -> Optional[pd.Series]:
            """Parabolic SAR."""
            from . import SAR
            high = self._get_column('high')
            low = self._get_column('low')
            result = SAR(high, low, acceleration=acceleration, maximum=maximum)
            return self._append_or_return(result, f"SAR_{acceleration}_{maximum}", append)
        
        def sarext(self, startvalue: float = 0.0, offsetonreverse: float = 0.0,
                   accelerationinit_long: float = 0.02, accelerationlong: float = 0.02,
                   accelerationmax_long: float = 0.2, accelerationinit_short: float = 0.02,
                   accelerationshort: float = 0.02, accelerationmax_short: float = 0.2,
                   append: bool = False) -> Optional[pd.Series]:
            """Parabolic SAR - Extended."""
            from . import SAREXT
            high = self._get_column('high')
            low = self._get_column('low')
            result = SAREXT(high, low, startvalue=startvalue, offsetonreverse=offsetonreverse,
                           accelerationinit_long=accelerationinit_long,
                           accelerationlong=accelerationlong,
                           accelerationmax_long=accelerationmax_long,
                           accelerationinit_short=accelerationinit_short,
                           accelerationshort=accelerationshort,
                           accelerationmax_short=accelerationmax_short)
            return self._append_or_return(result, "SAREXT", append)
        
        def t3(self, timeperiod: int = 5, vfactor: float = 0.7,
               column: Optional[str] = None, append: bool = False) -> Optional[pd.Series]:
            """Triple Exponential Moving Average (T3)."""
            from . import T3
            close = self._get_column('close', column)
            result = T3(close, timeperiod=timeperiod, vfactor=vfactor)
            return self._append_or_return(result, f"T3_{timeperiod}_{vfactor}", append)
        
        def tema(self, timeperiod: int = 30, column: Optional[str] = None,
                 append: bool = False) -> Optional[pd.Series]:
            """Triple Exponential Moving Average."""
            from . import TEMA
            close = self._get_column('close', column)
            result = TEMA(close, timeperiod=timeperiod)
            return self._append_or_return(result, f"TEMA_{timeperiod}", append)
        
        def trima(self, timeperiod: int = 30, column: Optional[str] = None,
                  append: bool = False) -> Optional[pd.Series]:
            """Triangular Moving Average."""
            from . import TRIMA
            close = self._get_column('close', column)
            result = TRIMA(close, timeperiod=timeperiod)
            return self._append_or_return(result, f"TRIMA_{timeperiod}", append)
        
        def wma(self, timeperiod: int = 30, column: Optional[str] = None,
                append: bool = False) -> Optional[pd.Series]:
            """Weighted Moving Average."""
            from . import WMA
            close = self._get_column('close', column)
            result = WMA(close, timeperiod=timeperiod)
            return self._append_or_return(result, f"WMA_{timeperiod}", append)

        # =====================================================================
        # MOMENTUM INDICATORS
        # =====================================================================
        
        def adx(self, timeperiod: int = 14, append: bool = False) -> Optional[pd.Series]:
            """Average Directional Movement Index."""
            from . import ADX
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = ADX(high, low, close, timeperiod=timeperiod)
            return self._append_or_return(result, f"ADX_{timeperiod}", append)
        
        def adxr(self, timeperiod: int = 14, append: bool = False) -> Optional[pd.Series]:
            """Average Directional Movement Index Rating."""
            from . import ADXR
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = ADXR(high, low, close, timeperiod=timeperiod)
            return self._append_or_return(result, f"ADXR_{timeperiod}", append)
        
        def apo(self, fastperiod: int = 12, slowperiod: int = 26, matype: int = 0,
                column: Optional[str] = None, append: bool = False) -> Optional[pd.Series]:
            """Absolute Price Oscillator."""
            from . import APO
            close = self._get_column('close', column)
            result = APO(close, fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)
            return self._append_or_return(result, f"APO_{fastperiod}_{slowperiod}", append)
        
        def aroon(self, timeperiod: int = 14, append: bool = False) -> Optional[pd.DataFrame]:
            """Aroon."""
            from . import AROON
            high = self._get_column('high')
            low = self._get_column('low')
            aroondown, aroonup = AROON(high, low, timeperiod=timeperiod)
            col_names = [f"AROONDOWN_{timeperiod}", f"AROONUP_{timeperiod}"]
            return self._append_or_return((aroondown, aroonup), col_names, append)
        
        def aroonosc(self, timeperiod: int = 14, append: bool = False) -> Optional[pd.Series]:
            """Aroon Oscillator."""
            from . import AROONOSC
            high = self._get_column('high')
            low = self._get_column('low')
            result = AROONOSC(high, low, timeperiod=timeperiod)
            return self._append_or_return(result, f"AROONOSC_{timeperiod}", append)
        
        def atr(self, timeperiod: int = 14, append: bool = False) -> Optional[pd.Series]:
            """Average True Range."""
            from . import ATR
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = ATR(high, low, close, timeperiod=timeperiod)
            return self._append_or_return(result, f"ATR_{timeperiod}", append)
        
        def bop(self, append: bool = False) -> Optional[pd.Series]:
            """Balance of Power."""
            from . import BOP
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = BOP(open_, high, low, close)
            return self._append_or_return(result, "BOP", append)
        
        def cci(self, timeperiod: int = 14, append: bool = False) -> Optional[pd.Series]:
            """Commodity Channel Index."""
            from . import CCI
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CCI(high, low, close, timeperiod=timeperiod)
            return self._append_or_return(result, f"CCI_{timeperiod}", append)
        
        def cmo(self, timeperiod: int = 14, column: Optional[str] = None,
                append: bool = False) -> Optional[pd.Series]:
            """Chande Momentum Oscillator."""
            from . import CMO
            close = self._get_column('close', column)
            result = CMO(close, timeperiod=timeperiod)
            return self._append_or_return(result, f"CMO_{timeperiod}", append)
        
        def dx(self, timeperiod: int = 14, append: bool = False) -> Optional[pd.Series]:
            """Directional Movement Index."""
            from . import DX
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = DX(high, low, close, timeperiod=timeperiod)
            return self._append_or_return(result, f"DX_{timeperiod}", append)
        
        def macd(self, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9,
                 column: Optional[str] = None, append: bool = False) -> Optional[pd.DataFrame]:
            """Moving Average Convergence/Divergence."""
            from . import MACD
            close = self._get_column('close', column)
            macd_line, signal, hist = MACD(close, fastperiod=fastperiod, 
                                           slowperiod=slowperiod, signalperiod=signalperiod)
            col_names = [
                f"MACD_{fastperiod}_{slowperiod}_{signalperiod}",
                f"MACDSignal_{fastperiod}_{slowperiod}_{signalperiod}",
                f"MACDHist_{fastperiod}_{slowperiod}_{signalperiod}"
            ]
            return self._append_or_return((macd_line, signal, hist), col_names, append)
        
        def macdext(self, fastperiod: int = 12, fastmatype: int = 0,
                    slowperiod: int = 26, slowmatype: int = 0,
                    signalperiod: int = 9, signalmatype: int = 0,
                    column: Optional[str] = None, append: bool = False) -> Optional[pd.DataFrame]:
            """MACD with Controllable MA Type."""
            from . import MACDEXT
            close = self._get_column('close', column)
            macd_line, signal, hist = MACDEXT(close, fastperiod=fastperiod, fastmatype=fastmatype,
                                              slowperiod=slowperiod, slowmatype=slowmatype,
                                              signalperiod=signalperiod, signalmatype=signalmatype)
            col_names = [
                f"MACDEXT_{fastperiod}_{slowperiod}_{signalperiod}",
                f"MACDEXTSignal_{fastperiod}_{slowperiod}_{signalperiod}",
                f"MACDEXTHist_{fastperiod}_{slowperiod}_{signalperiod}"
            ]
            return self._append_or_return((macd_line, signal, hist), col_names, append)
        
        def macdfix(self, signalperiod: int = 9, column: Optional[str] = None,
                    append: bool = False) -> Optional[pd.DataFrame]:
            """MACD with fixed 12/26 periods."""
            from . import MACDFIX
            close = self._get_column('close', column)
            macd_line, signal, hist = MACDFIX(close, signalperiod=signalperiod)
            col_names = [
                f"MACDFIX_12_26_{signalperiod}",
                f"MACDFIXSignal_12_26_{signalperiod}",
                f"MACDFIXHist_12_26_{signalperiod}"
            ]
            return self._append_or_return((macd_line, signal, hist), col_names, append)
        
        def mfi(self, timeperiod: int = 14, append: bool = False) -> Optional[pd.Series]:
            """Money Flow Index."""
            from . import MFI
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            volume = self._get_column('volume')
            result = MFI(high, low, close, volume, timeperiod=timeperiod)
            return self._append_or_return(result, f"MFI_{timeperiod}", append)
        
        def minus_di(self, timeperiod: int = 14, append: bool = False) -> Optional[pd.Series]:
            """Minus Directional Indicator."""
            from . import MINUS_DI
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = MINUS_DI(high, low, close, timeperiod=timeperiod)
            return self._append_or_return(result, f"MINUS_DI_{timeperiod}", append)
        
        def minus_dm(self, timeperiod: int = 14, append: bool = False) -> Optional[pd.Series]:
            """Minus Directional Movement."""
            from . import MINUS_DM
            high = self._get_column('high')
            low = self._get_column('low')
            result = MINUS_DM(high, low, timeperiod=timeperiod)
            return self._append_or_return(result, f"MINUS_DM_{timeperiod}", append)
        
        def mom(self, timeperiod: int = 10, column: Optional[str] = None,
                append: bool = False) -> Optional[pd.Series]:
            """Momentum."""
            from . import MOM
            close = self._get_column('close', column)
            result = MOM(close, timeperiod=timeperiod)
            return self._append_or_return(result, f"MOM_{timeperiod}", append)
        
        def plus_di(self, timeperiod: int = 14, append: bool = False) -> Optional[pd.Series]:
            """Plus Directional Indicator."""
            from . import PLUS_DI
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = PLUS_DI(high, low, close, timeperiod=timeperiod)
            return self._append_or_return(result, f"PLUS_DI_{timeperiod}", append)
        
        def plus_dm(self, timeperiod: int = 14, append: bool = False) -> Optional[pd.Series]:
            """Plus Directional Movement."""
            from . import PLUS_DM
            high = self._get_column('high')
            low = self._get_column('low')
            result = PLUS_DM(high, low, timeperiod=timeperiod)
            return self._append_or_return(result, f"PLUS_DM_{timeperiod}", append)
        
        def ppo(self, fastperiod: int = 12, slowperiod: int = 26, matype: int = 0,
                column: Optional[str] = None, append: bool = False) -> Optional[pd.Series]:
            """Percentage Price Oscillator."""
            from . import PPO
            close = self._get_column('close', column)
            result = PPO(close, fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)
            return self._append_or_return(result, f"PPO_{fastperiod}_{slowperiod}", append)
        
        def roc(self, timeperiod: int = 10, column: Optional[str] = None,
                append: bool = False) -> Optional[pd.Series]:
            """Rate of Change."""
            from . import ROC
            close = self._get_column('close', column)
            result = ROC(close, timeperiod=timeperiod)
            return self._append_or_return(result, f"ROC_{timeperiod}", append)
        
        def rocp(self, timeperiod: int = 10, column: Optional[str] = None,
                 append: bool = False) -> Optional[pd.Series]:
            """Rate of Change Percentage."""
            from . import ROCP
            close = self._get_column('close', column)
            result = ROCP(close, timeperiod=timeperiod)
            return self._append_or_return(result, f"ROCP_{timeperiod}", append)
        
        def rocr(self, timeperiod: int = 10, column: Optional[str] = None,
                 append: bool = False) -> Optional[pd.Series]:
            """Rate of Change Ratio."""
            from . import ROCR
            close = self._get_column('close', column)
            result = ROCR(close, timeperiod=timeperiod)
            return self._append_or_return(result, f"ROCR_{timeperiod}", append)
        
        def rocr100(self, timeperiod: int = 10, column: Optional[str] = None,
                    append: bool = False) -> Optional[pd.Series]:
            """Rate of Change Ratio 100 Scale."""
            from . import ROCR100
            close = self._get_column('close', column)
            result = ROCR100(close, timeperiod=timeperiod)
            return self._append_or_return(result, f"ROCR100_{timeperiod}", append)
        
        def rsi(self, timeperiod: int = 14, column: Optional[str] = None,
                append: bool = False) -> Optional[pd.Series]:
            """Relative Strength Index."""
            from . import RSI
            close = self._get_column('close', column)
            result = RSI(close, timeperiod=timeperiod)
            return self._append_or_return(result, f"RSI_{timeperiod}", append)
        
        def stoch(self, fastk_period: int = 5, slowk_period: int = 3,
                  slowk_matype: int = 0, slowd_period: int = 3, slowd_matype: int = 0,
                  append: bool = False) -> Optional[pd.DataFrame]:
            """Stochastic."""
            from . import STOCH
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            slowk, slowd = STOCH(high, low, close, fastk_period=fastk_period,
                                 slowk_period=slowk_period, slowk_matype=slowk_matype,
                                 slowd_period=slowd_period, slowd_matype=slowd_matype)
            col_names = [
                f"STOCH_SLOWK_{fastk_period}_{slowk_period}_{slowd_period}",
                f"STOCH_SLOWD_{fastk_period}_{slowk_period}_{slowd_period}"
            ]
            return self._append_or_return((slowk, slowd), col_names, append)
        
        def stochf(self, fastk_period: int = 5, fastd_period: int = 3,
                   fastd_matype: int = 0, append: bool = False) -> Optional[pd.DataFrame]:
            """Stochastic Fast."""
            from . import STOCHF
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            fastk, fastd = STOCHF(high, low, close, fastk_period=fastk_period,
                                  fastd_period=fastd_period, fastd_matype=fastd_matype)
            col_names = [
                f"STOCHF_FASTK_{fastk_period}_{fastd_period}",
                f"STOCHF_FASTD_{fastk_period}_{fastd_period}"
            ]
            return self._append_or_return((fastk, fastd), col_names, append)
        
        def stochrsi(self, timeperiod: int = 14, fastk_period: int = 5,
                     fastd_period: int = 3, fastd_matype: int = 0,
                     column: Optional[str] = None, append: bool = False) -> Optional[pd.DataFrame]:
            """Stochastic RSI."""
            from . import STOCHRSI
            close = self._get_column('close', column)
            fastk, fastd = STOCHRSI(close, timeperiod=timeperiod, fastk_period=fastk_period,
                                    fastd_period=fastd_period, fastd_matype=fastd_matype)
            col_names = [
                f"STOCHRSI_FASTK_{timeperiod}_{fastk_period}_{fastd_period}",
                f"STOCHRSI_FASTD_{timeperiod}_{fastk_period}_{fastd_period}"
            ]
            return self._append_or_return((fastk, fastd), col_names, append)
        
        def trix(self, timeperiod: int = 30, column: Optional[str] = None,
                 append: bool = False) -> Optional[pd.Series]:
            """1-day Rate-Of-Change of Triple Smooth EMA."""
            from . import TRIX
            close = self._get_column('close', column)
            result = TRIX(close, timeperiod=timeperiod)
            return self._append_or_return(result, f"TRIX_{timeperiod}", append)
        
        def ultosc(self, timeperiod1: int = 7, timeperiod2: int = 14,
                   timeperiod3: int = 28, append: bool = False) -> Optional[pd.Series]:
            """Ultimate Oscillator."""
            from . import ULTOSC
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = ULTOSC(high, low, close, timeperiod1=timeperiod1,
                           timeperiod2=timeperiod2, timeperiod3=timeperiod3)
            return self._append_or_return(result, f"ULTOSC_{timeperiod1}_{timeperiod2}_{timeperiod3}", append)
        
        def willr(self, timeperiod: int = 14, append: bool = False) -> Optional[pd.Series]:
            """Williams Percent R."""
            from . import WILLR
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = WILLR(high, low, close, timeperiod=timeperiod)
            return self._append_or_return(result, f"WILLR_{timeperiod}", append)

        # =====================================================================
        # VOLUME INDICATORS
        # =====================================================================
        
        def ad(self, append: bool = False) -> Optional[pd.Series]:
            """Chaikin A/D Line."""
            from . import AD
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            volume = self._get_column('volume')
            result = AD(high, low, close, volume)
            return self._append_or_return(result, 'AD', append)
        
        def adosc(self, fastperiod: int = 3, slowperiod: int = 10,
                  append: bool = False) -> Optional[pd.Series]:
            """Chaikin A/D Oscillator."""
            from . import ADOSC
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            volume = self._get_column('volume')
            result = ADOSC(high, low, close, volume, fastperiod=fastperiod, slowperiod=slowperiod)
            return self._append_or_return(result, f'ADOSC_{fastperiod}_{slowperiod}', append)
        
        def obv(self, append: bool = False) -> Optional[pd.Series]:
            """On Balance Volume."""
            from . import OBV
            close = self._get_column('close')
            volume = self._get_column('volume')
            result = OBV(close, volume)
            return self._append_or_return(result, 'OBV', append)
        
        # =====================================================================
        # VOLATILITY INDICATORS
        # =====================================================================
        
        def natr(self, timeperiod: int = 14, append: bool = False) -> Optional[pd.Series]:
            """Normalized Average True Range."""
            from . import NATR
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = NATR(high, low, close, timeperiod=timeperiod)
            return self._append_or_return(result, f'NATR_{timeperiod}', append)
        
        def trange(self, append: bool = False) -> Optional[pd.Series]:
            """True Range."""
            from . import TRANGE
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = TRANGE(high, low, close)
            return self._append_or_return(result, 'TRANGE', append)
        
        # =====================================================================
        # CYCLE INDICATORS
        # =====================================================================
        
        def ht_dcperiod(self, column: Optional[str] = None,
                        append: bool = False) -> Optional[pd.Series]:
            """Hilbert Transform - Dominant Cycle Period."""
            from . import HT_DCPERIOD
            close = self._get_column('close', column)
            result = HT_DCPERIOD(close)
            return self._append_or_return(result, 'HT_DCPERIOD', append)
        
        def ht_dcphase(self, column: Optional[str] = None,
                       append: bool = False) -> Optional[pd.Series]:
            """Hilbert Transform - Dominant Cycle Phase."""
            from . import HT_DCPHASE
            close = self._get_column('close', column)
            result = HT_DCPHASE(close)
            return self._append_or_return(result, 'HT_DCPHASE', append)
        
        def ht_phasor(self, column: Optional[str] = None,
                      append: bool = False) -> Optional[pd.DataFrame]:
            """Hilbert Transform - Phasor Components."""
            from . import HT_PHASOR
            close = self._get_column('close', column)
            inphase, quadrature = HT_PHASOR(close)
            col_names = ['HT_PHASOR_INPHASE', 'HT_PHASOR_QUADRATURE']
            return self._append_or_return((inphase, quadrature), col_names, append)
        
        def ht_sine(self, column: Optional[str] = None,
                    append: bool = False) -> Optional[pd.DataFrame]:
            """Hilbert Transform - SineWave."""
            from . import HT_SINE
            close = self._get_column('close', column)
            sine, leadsine = HT_SINE(close)
            col_names = ['HT_SINE', 'HT_LEADSINE']
            return self._append_or_return((sine, leadsine), col_names, append)
        
        def ht_trendline(self, column: Optional[str] = None,
                         append: bool = False) -> Optional[pd.Series]:
            """Hilbert Transform - Instantaneous Trendline."""
            from . import HT_TRENDLINE
            close = self._get_column('close', column)
            result = HT_TRENDLINE(close)
            return self._append_or_return(result, 'HT_TRENDLINE', append)
        
        def ht_trendmode(self, column: Optional[str] = None,
                         append: bool = False) -> Optional[pd.Series]:
            """Hilbert Transform - Trend vs Cycle Mode."""
            from . import HT_TRENDMODE
            close = self._get_column('close', column)
            result = HT_TRENDMODE(close)
            return self._append_or_return(result, 'HT_TRENDMODE', append)
        
        # =====================================================================
        # STATISTIC FUNCTIONS
        # =====================================================================
        
        def beta(self, timeperiod: int = 5, high_col: Optional[str] = None,
                 low_col: Optional[str] = None, append: bool = False) -> Optional[pd.Series]:
            """Beta."""
            from . import BETA
            high = self._get_column('high', high_col)
            low = self._get_column('low', low_col)
            result = BETA(high, low, timeperiod=timeperiod)
            return self._append_or_return(result, f'BETA_{timeperiod}', append)
        
        def correl(self, timeperiod: int = 30, high_col: Optional[str] = None,
                   low_col: Optional[str] = None, append: bool = False) -> Optional[pd.Series]:
            """Pearson Correlation Coefficient."""
            from . import CORREL
            high = self._get_column('high', high_col)
            low = self._get_column('low', low_col)
            result = CORREL(high, low, timeperiod=timeperiod)
            return self._append_or_return(result, f'CORREL_{timeperiod}', append)
        
        def linearreg(self, timeperiod: int = 14, column: Optional[str] = None,
                      append: bool = False) -> Optional[pd.Series]:
            """Linear Regression."""
            from . import LINEARREG
            close = self._get_column('close', column)
            result = LINEARREG(close, timeperiod=timeperiod)
            return self._append_or_return(result, f'LINEARREG_{timeperiod}', append)
        
        def linearreg_angle(self, timeperiod: int = 14, column: Optional[str] = None,
                            append: bool = False) -> Optional[pd.Series]:
            """Linear Regression Angle."""
            from . import LINEARREG_ANGLE
            close = self._get_column('close', column)
            result = LINEARREG_ANGLE(close, timeperiod=timeperiod)
            return self._append_or_return(result, f'LINEARREG_ANGLE_{timeperiod}', append)
        
        def linearreg_intercept(self, timeperiod: int = 14, column: Optional[str] = None,
                                append: bool = False) -> Optional[pd.Series]:
            """Linear Regression Intercept."""
            from . import LINEARREG_INTERCEPT
            close = self._get_column('close', column)
            result = LINEARREG_INTERCEPT(close, timeperiod=timeperiod)
            return self._append_or_return(result, f'LINEARREG_INTERCEPT_{timeperiod}', append)
        
        def linearreg_slope(self, timeperiod: int = 14, column: Optional[str] = None,
                            append: bool = False) -> Optional[pd.Series]:
            """Linear Regression Slope."""
            from . import LINEARREG_SLOPE
            close = self._get_column('close', column)
            result = LINEARREG_SLOPE(close, timeperiod=timeperiod)
            return self._append_or_return(result, f'LINEARREG_SLOPE_{timeperiod}', append)
        
        def stddev(self, timeperiod: int = 5, nbdev: float = 1.0,
                   column: Optional[str] = None, append: bool = False) -> Optional[pd.Series]:
            """Standard Deviation."""
            from . import STDDEV
            close = self._get_column('close', column)
            result = STDDEV(close, timeperiod=timeperiod, nbdev=nbdev)
            return self._append_or_return(result, f'STDDEV_{timeperiod}', append)
        
        def tsf(self, timeperiod: int = 14, column: Optional[str] = None,
                append: bool = False) -> Optional[pd.Series]:
            """Time Series Forecast."""
            from . import TSF
            close = self._get_column('close', column)
            result = TSF(close, timeperiod=timeperiod)
            return self._append_or_return(result, f'TSF_{timeperiod}', append)
        
        def var(self, timeperiod: int = 5, nbdev: float = 1.0,
                column: Optional[str] = None, append: bool = False) -> Optional[pd.Series]:
            """Variance."""
            from . import VAR
            close = self._get_column('close', column)
            result = VAR(close, timeperiod=timeperiod, nbdev=nbdev)
            return self._append_or_return(result, f'VAR_{timeperiod}', append)
        
        # =====================================================================
        # MATH OPERATORS
        # =====================================================================
        
        def max(self, timeperiod: int = 30, column: Optional[str] = None,
                append: bool = False) -> Optional[pd.Series]:
            """Highest value over a specified period."""
            from . import MAX
            close = self._get_column('close', column)
            result = MAX(close, timeperiod=timeperiod)
            return self._append_or_return(result, f'MAX_{timeperiod}', append)
        
        def maxindex(self, timeperiod: int = 30, column: Optional[str] = None,
                     append: bool = False) -> Optional[pd.Series]:
            """Index of highest value over a specified period."""
            from . import MAXINDEX
            close = self._get_column('close', column)
            result = MAXINDEX(close, timeperiod=timeperiod)
            return self._append_or_return(result, f'MAXINDEX_{timeperiod}', append)
        
        def min(self, timeperiod: int = 30, column: Optional[str] = None,
                append: bool = False) -> Optional[pd.Series]:
            """Lowest value over a specified period."""
            from . import MIN
            close = self._get_column('close', column)
            result = MIN(close, timeperiod=timeperiod)
            return self._append_or_return(result, f'MIN_{timeperiod}', append)
        
        def minindex(self, timeperiod: int = 30, column: Optional[str] = None,
                     append: bool = False) -> Optional[pd.Series]:
            """Index of lowest value over a specified period."""
            from . import MININDEX
            close = self._get_column('close', column)
            result = MININDEX(close, timeperiod=timeperiod)
            return self._append_or_return(result, f'MININDEX_{timeperiod}', append)
        
        def minmax(self, timeperiod: int = 30, column: Optional[str] = None,
                   append: bool = False) -> Optional[pd.DataFrame]:
            """Lowest and highest values over a specified period."""
            from . import MINMAX
            close = self._get_column('close', column)
            min_val, max_val = MINMAX(close, timeperiod=timeperiod)
            col_names = [f'MINMAX_MIN_{timeperiod}', f'MINMAX_MAX_{timeperiod}']
            return self._append_or_return((min_val, max_val), col_names, append)
        
        def minmaxindex(self, timeperiod: int = 30, column: Optional[str] = None,
                        append: bool = False) -> Optional[pd.DataFrame]:
            """Indexes of lowest and highest values over a specified period."""
            from . import MINMAXINDEX
            close = self._get_column('close', column)
            minidx, maxidx = MINMAXINDEX(close, timeperiod=timeperiod)
            col_names = [f'MINMAXINDEX_MIN_{timeperiod}', f'MINMAXINDEX_MAX_{timeperiod}']
            return self._append_or_return((minidx, maxidx), col_names, append)
        
        def sum(self, timeperiod: int = 30, column: Optional[str] = None,
                append: bool = False) -> Optional[pd.Series]:
            """Summation."""
            from . import SUM
            close = self._get_column('close', column)
            result = SUM(close, timeperiod=timeperiod)
            return self._append_or_return(result, f'SUM_{timeperiod}', append)
        
        # =====================================================================
        # PRICE TRANSFORM
        # =====================================================================
        
        def medprice(self, append: bool = False) -> Optional[pd.Series]:
            """Median Price."""
            from . import MEDPRICE
            high = self._get_column('high')
            low = self._get_column('low')
            result = MEDPRICE(high, low)
            return self._append_or_return(result, 'MEDPRICE', append)
        
        def midpoint(self, timeperiod: int = 14, column: Optional[str] = None,
                     append: bool = False) -> Optional[pd.Series]:
            """MidPoint over period."""
            from . import MIDPOINT
            close = self._get_column('close', column)
            result = MIDPOINT(close, timeperiod=timeperiod)
            return self._append_or_return(result, f'MIDPOINT_{timeperiod}', append)
        
        def midprice(self, timeperiod: int = 14, append: bool = False) -> Optional[pd.Series]:
            """MidPoint Price over period."""
            from . import MIDPRICE
            high = self._get_column('high')
            low = self._get_column('low')
            result = MIDPRICE(high, low, timeperiod=timeperiod)
            return self._append_or_return(result, f'MIDPRICE_{timeperiod}', append)
        
        def typprice(self, append: bool = False) -> Optional[pd.Series]:
            """Typical Price."""
            from . import TYPPRICE
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = TYPPRICE(high, low, close)
            return self._append_or_return(result, 'TYPPRICE', append)
        
        def wclprice(self, append: bool = False) -> Optional[pd.Series]:
            """Weighted Close Price."""
            from . import WCLPRICE
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = WCLPRICE(high, low, close)
            return self._append_or_return(result, 'WCLPRICE', append)

        # =====================================================================
        # PATTERN RECOGNITION
        # =====================================================================

        def cdl2crows(self, append: bool = False) -> Optional[pd.Series]:
            """Two Crows pattern."""
            from . import CDL2CROWS
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDL2CROWS(open_, high, low, close)
            return self._append_or_return(result, 'CDL2CROWS', append)

        def cdl3blackcrows(self, append: bool = False) -> Optional[pd.Series]:
            """Three Black Crows pattern."""
            from . import CDL3BLACKCROWS
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDL3BLACKCROWS(open_, high, low, close)
            return self._append_or_return(result, 'CDL3BLACKCROWS', append)

        def cdl3inside(self, append: bool = False) -> Optional[pd.Series]:
            """Three Inside Up/Down pattern."""
            from . import CDL3INSIDE
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDL3INSIDE(open_, high, low, close)
            return self._append_or_return(result, 'CDL3INSIDE', append)

        def cdl3outside(self, append: bool = False) -> Optional[pd.Series]:
            """Three Outside Up/Down pattern."""
            from . import CDL3OUTSIDE
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDL3OUTSIDE(open_, high, low, close)
            return self._append_or_return(result, 'CDL3OUTSIDE', append)

        def cdl3starsinsouth(self, append: bool = False) -> Optional[pd.Series]:
            """Three Stars in the South pattern."""
            from . import CDL3STARSINSOUTH
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDL3STARSINSOUTH(open_, high, low, close)
            return self._append_or_return(result, 'CDL3STARSINSOUTH', append)

        def cdl3whitesoldiers(self, append: bool = False) -> Optional[pd.Series]:
            """Three White Soldiers pattern."""
            from . import CDL3WHITESOLDIERS
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDL3WHITESOLDIERS(open_, high, low, close)
            return self._append_or_return(result, 'CDL3WHITESOLDIERS', append)

        def cdlabandonedbaby(self, append: bool = False) -> Optional[pd.Series]:
            """Abandoned Baby pattern."""
            from . import CDLABANDONEDBABY
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLABANDONEDBABY(open_, high, low, close)
            return self._append_or_return(result, 'CDLABANDONEDBABY', append)

        def cdladvanceblock(self, append: bool = False) -> Optional[pd.Series]:
            """Advance Block pattern."""
            from . import CDLADVANCEBLOCK
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLADVANCEBLOCK(open_, high, low, close)
            return self._append_or_return(result, 'CDLADVANCEBLOCK', append)

        def cdlbelthold(self, append: bool = False) -> Optional[pd.Series]:
            """Belt Hold pattern."""
            from . import CDLBELTHOLD
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLBELTHOLD(open_, high, low, close)
            return self._append_or_return(result, 'CDLBELTHOLD', append)

        def cdlbreakaway(self, append: bool = False) -> Optional[pd.Series]:
            """Breakaway pattern."""
            from . import CDLBREAKAWAY
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLBREAKAWAY(open_, high, low, close)
            return self._append_or_return(result, 'CDLBREAKAWAY', append)

        def cdlclosingmarubozu(self, append: bool = False) -> Optional[pd.Series]:
            """Closing Marubozu pattern."""
            from . import CDLCLOSINGMARUBOZU
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLCLOSINGMARUBOZU(open_, high, low, close)
            return self._append_or_return(result, 'CDLCLOSINGMARUBOZU', append)

        def cdlconcealbabyswall(self, append: bool = False) -> Optional[pd.Series]:
            """Concealing Baby Swallow pattern."""
            from . import CDLCONCEALBABYSWALL
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLCONCEALBABYSWALL(open_, high, low, close)
            return self._append_or_return(result, 'CDLCONCEALBABYSWALL', append)

        def cdlcounterattack(self, append: bool = False) -> Optional[pd.Series]:
            """Counterattack pattern."""
            from . import CDLCOUNTERATTACK
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLCOUNTERATTACK(open_, high, low, close)
            return self._append_or_return(result, 'CDLCOUNTERATTACK', append)

        def cdldarkcloudcover(self, append: bool = False) -> Optional[pd.Series]:
            """Dark Cloud Cover pattern."""
            from . import CDLDARKCLOUDCOVER
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLDARKCLOUDCOVER(open_, high, low, close)
            return self._append_or_return(result, 'CDLDARKCLOUDCOVER', append)

        def cdldoji(self, append: bool = False) -> Optional[pd.Series]:
            """Doji pattern."""
            from . import CDLDOJI
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLDOJI(open_, high, low, close)
            return self._append_or_return(result, 'CDLDOJI', append)

        def cdldojistar(self, append: bool = False) -> Optional[pd.Series]:
            """Doji Star pattern."""
            from . import CDLDOJISTAR
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLDOJISTAR(open_, high, low, close)
            return self._append_or_return(result, 'CDLDOJISTAR', append)

        def cdldragonflydoji(self, append: bool = False) -> Optional[pd.Series]:
            """Dragonfly Doji pattern."""
            from . import CDLDRAGONFLYDOJI
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLDRAGONFLYDOJI(open_, high, low, close)
            return self._append_or_return(result, 'CDLDRAGONFLYDOJI', append)

        def cdlengulfing(self, append: bool = False) -> Optional[pd.Series]:
            """Engulfing pattern."""
            from . import CDLENGULFING
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLENGULFING(open_, high, low, close)
            return self._append_or_return(result, 'CDLENGULFING', append)

        def cdleveningdojistar(self, append: bool = False) -> Optional[pd.Series]:
            """Evening Doji Star pattern."""
            from . import CDLEVENINGDOJISTAR
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLEVENINGDOJISTAR(open_, high, low, close)
            return self._append_or_return(result, 'CDLEVENINGDOJISTAR', append)

        def cdleveningstar(self, append: bool = False) -> Optional[pd.Series]:
            """Evening Star pattern."""
            from . import CDLEVENINGSTAR
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLEVENINGSTAR(open_, high, low, close)
            return self._append_or_return(result, 'CDLEVENINGSTAR', append)

        def cdlgapsidesidewhite(self, append: bool = False) -> Optional[pd.Series]:
            """Gap Side-by-Side White Lines pattern."""
            from . import CDLGAPSIDESIDEWHITE
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLGAPSIDESIDEWHITE(open_, high, low, close)
            return self._append_or_return(result, 'CDLGAPSIDESIDEWHITE', append)

        def cdlgravestonedoji(self, append: bool = False) -> Optional[pd.Series]:
            """Gravestone Doji pattern."""
            from . import CDLGRAVESTONEDOJI
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLGRAVESTONEDOJI(open_, high, low, close)
            return self._append_or_return(result, 'CDLGRAVESTONEDOJI', append)

        def cdlhammer(self, append: bool = False) -> Optional[pd.Series]:
            """Hammer pattern."""
            from . import CDLHAMMER
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLHAMMER(open_, high, low, close)
            return self._append_or_return(result, 'CDLHAMMER', append)

        def cdlhangingman(self, append: bool = False) -> Optional[pd.Series]:
            """Hanging Man pattern."""
            from . import CDLHANGINGMAN
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLHANGINGMAN(open_, high, low, close)
            return self._append_or_return(result, 'CDLHANGINGMAN', append)

        def cdlharami(self, append: bool = False) -> Optional[pd.Series]:
            """Harami pattern."""
            from . import CDLHARAMI
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLHARAMI(open_, high, low, close)
            return self._append_or_return(result, 'CDLHARAMI', append)

        def cdlharamicross(self, append: bool = False) -> Optional[pd.Series]:
            """Harami Cross pattern."""
            from . import CDLHARAMICROSS
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLHARAMICROSS(open_, high, low, close)
            return self._append_or_return(result, 'CDLHARAMICROSS', append)

        def cdlhighwave(self, append: bool = False) -> Optional[pd.Series]:
            """High Wave pattern."""
            from . import CDLHIGHWAVE
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLHIGHWAVE(open_, high, low, close)
            return self._append_or_return(result, 'CDLHIGHWAVE', append)

        def cdlhikkake(self, append: bool = False) -> Optional[pd.Series]:
            """Hikkake pattern."""
            from . import CDLHIKKAKE
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLHIKKAKE(open_, high, low, close)
            return self._append_or_return(result, 'CDLHIKKAKE', append)

        def cdlhikkakemod(self, append: bool = False) -> Optional[pd.Series]:
            """Modified Hikkake pattern."""
            from . import CDLHIKKAKEMOD
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLHIKKAKEMOD(open_, high, low, close)
            return self._append_or_return(result, 'CDLHIKKAKEMOD', append)

        def cdlhomingpigeon(self, append: bool = False) -> Optional[pd.Series]:
            """Homing Pigeon pattern."""
            from . import CDLHOMINGPIGEON
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLHOMINGPIGEON(open_, high, low, close)
            return self._append_or_return(result, 'CDLHOMINGPIGEON', append)

        def cdlidentical3crows(self, append: bool = False) -> Optional[pd.Series]:
            """Identical Three Crows pattern."""
            from . import CDLIDENTICAL3CROWS
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLIDENTICAL3CROWS(open_, high, low, close)
            return self._append_or_return(result, 'CDLIDENTICAL3CROWS', append)

        def cdlinneck(self, append: bool = False) -> Optional[pd.Series]:
            """In-Neck pattern."""
            from . import CDLINNECK
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLINNECK(open_, high, low, close)
            return self._append_or_return(result, 'CDLINNECK', append)

        def cdlinvertedhammer(self, append: bool = False) -> Optional[pd.Series]:
            """Inverted Hammer pattern."""
            from . import CDLINVERTEDHAMMER
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLINVERTEDHAMMER(open_, high, low, close)
            return self._append_or_return(result, 'CDLINVERTEDHAMMER', append)

        def cdlkicking(self, append: bool = False) -> Optional[pd.Series]:
            """Kicking pattern."""
            from . import CDLKICKING
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLKICKING(open_, high, low, close)
            return self._append_or_return(result, 'CDLKICKING', append)

        def cdlkickingbylength(self, append: bool = False) -> Optional[pd.Series]:
            """Kicking By Length pattern."""
            from . import CDLKICKINGBYLENGTH
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLKICKINGBYLENGTH(open_, high, low, close)
            return self._append_or_return(result, 'CDLKICKINGBYLENGTH', append)

        def cdlladderbottom(self, append: bool = False) -> Optional[pd.Series]:
            """Ladder Bottom pattern."""
            from . import CDLLADDERBOTTOM
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLLADDERBOTTOM(open_, high, low, close)
            return self._append_or_return(result, 'CDLLADDERBOTTOM', append)

        def cdllongleggeddoji(self, append: bool = False) -> Optional[pd.Series]:
            """Long-Legged Doji pattern."""
            from . import CDLLONGLEGGEDDOJI
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLLONGLEGGEDDOJI(open_, high, low, close)
            return self._append_or_return(result, 'CDLLONGLEGGEDDOJI', append)

        def cdllongline(self, append: bool = False) -> Optional[pd.Series]:
            """Long Line pattern."""
            from . import CDLLONGLINE
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLLONGLINE(open_, high, low, close)
            return self._append_or_return(result, 'CDLLONGLINE', append)

        def cdlmarubozu(self, append: bool = False) -> Optional[pd.Series]:
            """Marubozu pattern."""
            from . import CDLMARUBOZU
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLMARUBOZU(open_, high, low, close)
            return self._append_or_return(result, 'CDLMARUBOZU', append)

        def cdlmatchinglow(self, append: bool = False) -> Optional[pd.Series]:
            """Matching Low pattern."""
            from . import CDLMATCHINGLOW
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLMATCHINGLOW(open_, high, low, close)
            return self._append_or_return(result, 'CDLMATCHINGLOW', append)

        def cdlmathold(self, append: bool = False) -> Optional[pd.Series]:
            """Mat Hold pattern."""
            from . import CDLMATHOLD
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLMATHOLD(open_, high, low, close)
            return self._append_or_return(result, 'CDLMATHOLD', append)

        def cdlmorningdojistar(self, append: bool = False) -> Optional[pd.Series]:
            """Morning Doji Star pattern."""
            from . import CDLMORNINGDOJISTAR
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLMORNINGDOJISTAR(open_, high, low, close)
            return self._append_or_return(result, 'CDLMORNINGDOJISTAR', append)

        def cdlmorningstar(self, append: bool = False) -> Optional[pd.Series]:
            """Morning Star pattern."""
            from . import CDLMORNINGSTAR
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLMORNINGSTAR(open_, high, low, close)
            return self._append_or_return(result, 'CDLMORNINGSTAR', append)

        def cdlonneck(self, append: bool = False) -> Optional[pd.Series]:
            """On-Neck pattern."""
            from . import CDLONNECK
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLONNECK(open_, high, low, close)
            return self._append_or_return(result, 'CDLONNECK', append)

        def cdlpiercing(self, append: bool = False) -> Optional[pd.Series]:
            """Piercing pattern."""
            from . import CDLPIERCING
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLPIERCING(open_, high, low, close)
            return self._append_or_return(result, 'CDLPIERCING', append)

        def cdlrickshawman(self, append: bool = False) -> Optional[pd.Series]:
            """Rickshaw Man pattern."""
            from . import CDLRICKSHAWMAN
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLRICKSHAWMAN(open_, high, low, close)
            return self._append_or_return(result, 'CDLRICKSHAWMAN', append)

        def cdlrisefall3methods(self, append: bool = False) -> Optional[pd.Series]:
            """Rising/Falling Three Methods pattern."""
            from . import CDLRISEFALL3METHODS
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLRISEFALL3METHODS(open_, high, low, close)
            return self._append_or_return(result, 'CDLRISEFALL3METHODS', append)

        def cdlseparatinglines(self, append: bool = False) -> Optional[pd.Series]:
            """Separating Lines pattern."""
            from . import CDLSEPARATINGLINES
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLSEPARATINGLINES(open_, high, low, close)
            return self._append_or_return(result, 'CDLSEPARATINGLINES', append)

        def cdlshootingstar(self, append: bool = False) -> Optional[pd.Series]:
            """Shooting Star pattern."""
            from . import CDLSHOOTINGSTAR
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLSHOOTINGSTAR(open_, high, low, close)
            return self._append_or_return(result, 'CDLSHOOTINGSTAR', append)

        def cdlshortline(self, append: bool = False) -> Optional[pd.Series]:
            """Short Line pattern."""
            from . import CDLSHORTLINE
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLSHORTLINE(open_, high, low, close)
            return self._append_or_return(result, 'CDLSHORTLINE', append)

        def cdlspinningtop(self, append: bool = False) -> Optional[pd.Series]:
            """Spinning Top pattern."""
            from . import CDLSPINNINGTOP
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLSPINNINGTOP(open_, high, low, close)
            return self._append_or_return(result, 'CDLSPINNINGTOP', append)

        def cdlstalledpattern(self, append: bool = False) -> Optional[pd.Series]:
            """Stalled Pattern pattern."""
            from . import CDLSTALLEDPATTERN
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLSTALLEDPATTERN(open_, high, low, close)
            return self._append_or_return(result, 'CDLSTALLEDPATTERN', append)

        def cdlsticksandwich(self, append: bool = False) -> Optional[pd.Series]:
            """Stick Sandwich pattern."""
            from . import CDLSTICKSANDWICH
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLSTICKSANDWICH(open_, high, low, close)
            return self._append_or_return(result, 'CDLSTICKSANDWICH', append)

        def cdltakuri(self, append: bool = False) -> Optional[pd.Series]:
            """Takuri pattern."""
            from . import CDLTAKURI
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLTAKURI(open_, high, low, close)
            return self._append_or_return(result, 'CDLTAKURI', append)

        def cdltasukigap(self, append: bool = False) -> Optional[pd.Series]:
            """Tasuki Gap pattern."""
            from . import CDLTASUKIGAP
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLTASUKIGAP(open_, high, low, close)
            return self._append_or_return(result, 'CDLTASUKIGAP', append)

        def cdlthrusting(self, append: bool = False) -> Optional[pd.Series]:
            """Thrusting pattern."""
            from . import CDLTHRUSTING
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLTHRUSTING(open_, high, low, close)
            return self._append_or_return(result, 'CDLTHRUSTING', append)

        def cdltristar(self, append: bool = False) -> Optional[pd.Series]:
            """Tristar pattern."""
            from . import CDLTRISTAR
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLTRISTAR(open_, high, low, close)
            return self._append_or_return(result, 'CDLTRISTAR', append)

        def cdlunique3river(self, append: bool = False) -> Optional[pd.Series]:
            """Unique 3 River pattern."""
            from . import CDLUNIQUE3RIVER
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLUNIQUE3RIVER(open_, high, low, close)
            return self._append_or_return(result, 'CDLUNIQUE3RIVER', append)

        def cdlupsidegap2crows(self, append: bool = False) -> Optional[pd.Series]:
            """Upside Gap Two Crows pattern."""
            from . import CDLUPSIDEGAP2CROWS
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLUPSIDEGAP2CROWS(open_, high, low, close)
            return self._append_or_return(result, 'CDLUPSIDEGAP2CROWS', append)

        def cdlxsidegap3methods(self, append: bool = False) -> Optional[pd.Series]:
            """Upside/Downside Gap 3 Methods pattern."""
            from . import CDLXSIDEGAP3METHODS
            open_ = self._get_column('open')
            high = self._get_column('high')
            low = self._get_column('low')
            close = self._get_column('close')
            result = CDLXSIDEGAP3METHODS(open_, high, low, close)
            return self._append_or_return(result, 'CDLXSIDEGAP3METHODS', append)
