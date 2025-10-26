# talib-pure Function Implementation Status

This table shows which talib-pure functions have Numba (CPU) and GPU (CuPy) implementations.

| Category | Function Name | Numba (CPU) | GPU (CuPy) | Notes |
|----------|---------------|-------------|------------|-------|
| Cycle Indicators | HT_DCPERIOD | ✓ | ✗ | - |
| Cycle Indicators | HT_DCPHASE | ✓ | ✗ | - |
| Cycle Indicators | HT_PHASOR | ✓ | ✗ | - |
| Cycle Indicators | HT_SINE | ✓ | ✗ | - |
| Cycle Indicators | HT_TRENDLINE | ✓ | ✗ | - |
| Cycle Indicators | HT_TRENDMODE | ✓ | ✗ | - |
| Math Operators | MAX | ✓ | ✗ | - |
| Math Operators | MAXINDEX | ✓ | ✗ | - |
| Math Operators | MIN | ✓ | ✗ | - |
| Math Operators | MININDEX | ✓ | ✗ | - |
| Math Operators | MINMAX | ✓ | ✗ | - |
| Math Operators | MINMAXINDEX | ✓ | ✗ | - |
| Math Operators | SUM | ✓ | ✗ | - |
| Momentum Indicators | ADX | ✓ | ✗ | - |
| Momentum Indicators | ADXR | ✓ | ✗ | - |
| Momentum Indicators | APO | ✓ | ✗ | - |
| Momentum Indicators | AROON | ✓ | ✗ | - |
| Momentum Indicators | AROONOSC | ✓ | ✗ | - |
| Momentum Indicators | ATR | ✓ | ✗ | - |
| Momentum Indicators | BOP | ✓ | ✗ | - |
| Momentum Indicators | CCI | ✓ | ✗ | - |
| Momentum Indicators | CMO | ✓ | ✓ | - |
| Momentum Indicators | DX | ✓ | ✓ | - |
| Momentum Indicators | MACD | ✓ | ✓ | - |
| Momentum Indicators | MACDEXT | ✓ | ✗ | - |
| Momentum Indicators | MACDFIX | ✓ | ✗ | - |
| Momentum Indicators | MFI | ✓ | ✗ | - |
| Momentum Indicators | MINUS_DI | ✓ | ✗ | - |
| Momentum Indicators | MINUS_DM | ✓ | ✗ | - |
| Momentum Indicators | MOM | ✓ | ✗ | - |
| Momentum Indicators | PLUS_DI | ✓ | ✗ | - |
| Momentum Indicators | PLUS_DM | ✓ | ✗ | - |
| Momentum Indicators | PPO | ✓ | ✗ | - |
| Momentum Indicators | ROC | ✓ | ✗ | - |
| Momentum Indicators | ROCP | ✓ | ✗ | - |
| Momentum Indicators | ROCR | ✓ | ✗ | - |
| Momentum Indicators | ROCR100 | ✓ | ✗ | - |
| Momentum Indicators | RSI | ✓ | ✓ | - |
| Momentum Indicators | STOCH | ✓ | ✗ | - |
| Momentum Indicators | STOCHF | ✓ | ✗ | - |
| Momentum Indicators | STOCHRSI | ✓ | ✗ | - |
| Momentum Indicators | TRIX | ✓ | ✗ | - |
| Momentum Indicators | ULTOSC | ✓ | ✗ | - |
| Momentum Indicators | WILLR | ✓ | ✗ | - |
| Overlap | BBANDS | ✓ | ✗ | - |
| Overlap | DEMA | ✓ | ✓ | - |
| Overlap | EMA | ✓ | ✓ | - |
| Overlap | KAMA | ✓ | ✓ | - |
| Overlap | MA | ✓ | ✓ | Routes to all MA types (SMA/EMA/WMA/DEMA/TEMA/TRIMA/KAMA/MAMA/T3) |
| Overlap | MAMA | ✓ | ✓ | Simplified adaptive implementation |
| Overlap | SAR | ✓ | ✓ | - |
| Overlap | SAREXT | ✓ | ✓ | - |
| Overlap | SMA | ✓ | ✓ | - |
| Overlap | T3 | ✓ | ✓ | Uses 6 EMAs with coefficients |
| Overlap | TEMA | ✓ | ✓ | Uses EMA internally |
| Overlap | TRIMA | ✓ | ✓ | Uses SMA internally (double smoothed) |
| Overlap | WMA | ✓ | ✓ | - |
| Pattern Recognition | CDL2CROWS | ✓ | ✗ | Bearish reversal pattern |
| Pattern Recognition | CDL3BLACKCROWS | ✓ | ✗ | Bearish reversal pattern |
| Pattern Recognition | CDL3INSIDE | ✓ | ✗ | Bullish/Bearish reversal pattern |
| Pattern Recognition | CDL3OUTSIDE | ✓ | ✗ | Bullish/Bearish reversal pattern |
| Pattern Recognition | CDL3STARSINSOUTH | ✗ | ✗ | Not yet implemented |
| Pattern Recognition | CDL3WHITESOLDIERS | ✗ | ✗ | Not yet implemented |
| Pattern Recognition | CDLABANDONEDBABY | ✗ | ✗ | Not yet implemented |
| Pattern Recognition | CDLADVANCEBLOCK | ✗ | ✗ | Not yet implemented |
| Pattern Recognition | CDLBELTHOLD | ✗ | ✗ | Not yet implemented |
| Pattern Recognition | CDLBREAKAWAY | ✗ | ✗ | Not yet implemented |
| Pattern Recognition | CDLCLOSINGMARUBOZU | ✗ | ✗ | Not yet implemented |
| Pattern Recognition | CDLCONCEALBABYSWALL | ✗ | ✗ | Not yet implemented |
| Pattern Recognition | CDLCOUNTERATTACK | ✗ | ✗ | Not yet implemented |
| Pattern Recognition | CDLDARKCLOUDCOVER | ✗ | ✗ | Not yet implemented |
| Pattern Recognition | CDLDOJI | ✗ | ✗ | Not yet implemented |
| Pattern Recognition | CDLDOJISTAR | ✗ | ✗ | Not yet implemented |
| Pattern Recognition | CDLDRAGONFLYDOJI | ✗ | ✗ | Not yet implemented |
| Pattern Recognition | CDLENGULFING | ✗ | ✗ | Not yet implemented |
| Pattern Recognition | CDLEVENINGDOJISTAR | ✗ | ✗ | Not yet implemented |
| Pattern Recognition | CDLEVENINGSTAR | ✗ | ✗ | Not yet implemented |
| Pattern Recognition | CDLGAPSIDESIDEWHITE | ✗ | ✗ | Not yet implemented |
| Pattern Recognition | CDLGRAVESTONEDOJI | ✗ | ✗ | Not yet implemented |
| Pattern Recognition | CDLHAMMER | ✗ | ✗ | Not yet implemented |
| Pattern Recognition | CDLHANGINGMAN | ✗ | ✗ | Not yet implemented |
| Pattern Recognition | CDLHARAMI | ✗ | ✗ | Not yet implemented |
| Pattern Recognition | CDLHARAMICROSS | ✗ | ✗ | Not yet implemented |
| Pattern Recognition | CDLHIGHWAVE | ✗ | ✗ | Not yet implemented |
| Pattern Recognition | CDLHIKKAKE | ✗ | ✗ | Not yet implemented |
| Pattern Recognition | CDLHIKKAKEMOD | ✗ | ✗ | Not yet implemented |
| Pattern Recognition | CDLHOMINGPIGEON | ✗ | ✗ | Not yet implemented |
| Pattern Recognition | CDLIDENTICAL3CROWS | ✗ | ✗ | Not yet implemented |
| Pattern Recognition | CDLINNECK | ✗ | ✗ | Not yet implemented |
| Pattern Recognition | CDLINVERTEDHAMMER | ✗ | ✗ | Not yet implemented |
| Pattern Recognition | CDLKICKING | ✗ | ✗ | Not yet implemented |
| Pattern Recognition | CDLKICKINGBYLENGTH | ✗ | ✗ | Not yet implemented |
| Pattern Recognition | CDLLADDERBOTTOM | ✗ | ✗ | Not yet implemented |
| Pattern Recognition | CDLLONGLEGGEDDOJI | ✗ | ✗ | Not yet implemented |
| Pattern Recognition | CDLLONGLINE | ✗ | ✗ | Not yet implemented |
| Pattern Recognition | CDLMARUBOZU | ✓ | ✓ | - |
| Pattern Recognition | CDLMATCHINGLOW | ✓ | ✓ | - |
| Pattern Recognition | CDLMATHOLD | ✓ | ✓ | - |
| Pattern Recognition | CDLMORNINGDOJISTAR | ✓ | ✓ | - |
| Pattern Recognition | CDLMORNINGSTAR | ✓ | ✓ | - |
| Pattern Recognition | CDLONNECK | ✓ | ✓ | - |
| Pattern Recognition | CDLPIERCING | ✓ | ✓ | - |
| Pattern Recognition | CDLRICKSHAWMAN | ✓ | ✓ | - |
| Pattern Recognition | CDLRISEFALL3METHODS | ✓ | ✓ | - |
| Pattern Recognition | CDLSEPARATINGLINES | ✓ | ✓ | - |
| Pattern Recognition | CDLSHOOTINGSTAR | ✓ | ✓ | - |
| Pattern Recognition | CDLSHORTLINE | ✓ | ✓ | - |
| Pattern Recognition | CDLSPINNINGTOP | ✓ | ✓ | - |
| Pattern Recognition | CDLSTALLEDPATTERN | ✓ | ✓ | - |
| Pattern Recognition | CDLSTICKSANDWICH | ✓ | ✓ | - |
| Pattern Recognition | CDLTAKURI | ✓ | ✓ | - |
| Pattern Recognition | CDLTASUKIGAP | ✓ | ✓ | - |
| Pattern Recognition | CDLTHRUSTING | ✓ | ✓ | - |
| Pattern Recognition | CDLTRISTAR | ✓ | ✓ | - |
| Pattern Recognition | CDLUNIQUE3RIVER | ✓ | ✓ | - |
| Pattern Recognition | CDLUPSIDEGAP2CROWS | ✓ | ✓ | - |
| Pattern Recognition | CDLXSIDEGAP3METHODS | ✓ | ✓ | - |
| Price Transform | MEDPRICE | ✓ | ✗ | - |
| Price Transform | MIDPOINT | ✓ | ✗ | - |
| Price Transform | MIDPRICE | ✓ | ✗ | - |
| Price Transform | TYPPRICE | ✓ | ✗ | - |
| Price Transform | WCLPRICE | ✓ | ✗ | - |
| Statistic Functions | BETA | ✗ | ✗ | Not yet implemented |
| Statistic Functions | CORREL | ✓ | ✗ | - |
| Statistic Functions | LINEARREG | ✓ | ✗ | - |
| Statistic Functions | LINEARREG_ANGLE | ✓ | ✗ | - |
| Statistic Functions | LINEARREG_INTERCEPT | ✓ | ✗ | - |
| Statistic Functions | LINEARREG_SLOPE | ✓ | ✗ | - |
| Statistics | STDDEV | ✓ | ✗ | - |
| Statistics | TSF | ✓ | ✗ | - |
| Statistics | VAR | ✓ | ✗ | - |
| Volatility Indicators | NATR | ✗ | ✗ | Uses ATR internally |
| Volatility Indicators | TRANGE | ✓ | ✗ | - |
| Volume Indicators | AD | ✓ | ✗ | - |
| Volume Indicators | ADOSC | ✓ | ✗ | - |
| Volume Indicators | OBV | ✓ | ✗ | - |

## Summary Statistics

### By Category

| Category | Total Functions | CPU Implemented | GPU Implemented |
|----------|----------------|-----------------|-----------------|
| Cycle Indicators | 6 | 6 (100%) | 0 (0%) |
| Math Operators | 7 | 7 (100%) | 0 (0%) |
| Momentum Indicators | 31 | 31 (100%) | 4 (12%) |
| Overlap | 13 | 13 (100%) | 12 (92%) |
| Pattern Recognition | 60 | 26 (43%) | 22 (36%) |
| Price Transform | 5 | 5 (100%) | 0 (0%) |
| Statistic Functions | 6 | 5 (83%) | 0 (0%) |
| Statistics | 3 | 3 (100%) | 0 (0%) |
| Volatility Indicators | 2 | 1 (50%) | 0 (0%) |
| Volume Indicators | 3 | 3 (100%) | 0 (0%) |

### Overall

- **Total Functions**: 136
- **CPU (Numba) Implementations**: 100 (74%)
- **GPU (CuPy) Implementations**: 38 (27%)
- **Not Yet Implemented**: 18
- **Fully Implemented**: 106

## Notes

- **✓** indicates the function has the corresponding implementation
- **✗** indicates the function does not have the corresponding implementation
- Some functions like `MA`, `TEMA`, `T3`, `TRIMA` use other functions internally rather than having direct Numba implementations
- Functions marked as "Not yet implemented" will raise `NotImplementedError` when called

---
*Generated automatically from source code analysis*
