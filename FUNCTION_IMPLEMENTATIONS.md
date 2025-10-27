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
| Pattern Recognition | CDL3STARSINSOUTH | ✓ | ✗ | Bullish reversal - 3 black candles with progressively smaller bodies |
| Pattern Recognition | CDL3WHITESOLDIERS | ✓ | ✗ | Bullish reversal - 3 consecutive white candles advancing progressively |
| Pattern Recognition | CDLABANDONEDBABY | ✓ | ✗ | Reversal pattern with isolated doji (gap before and after) |
| Pattern Recognition | CDLADVANCEBLOCK | ✓ | ✗ | Bearish warning - 3 white candles with decreasing bodies |
| Pattern Recognition | CDLBELTHOLD | ✓ | ✗ | Reversal - marubozu opening on extreme (white on low/black on high) |
| Pattern Recognition | CDLBREAKAWAY | ✓ | ✗ | 5-candle continuation pattern with gap closure |
| Pattern Recognition | CDLCLOSINGMARUBOZU | ✓ | ✗ | Single candle with no shadow at close (marubozu variant) |
| Pattern Recognition | CDLCONCEALBABYSWALL | ✓ | ✗ | Rare 4-candle bullish reversal with black marubozu pattern |
| Pattern Recognition | CDLCOUNTERATTACK | ✓ | ✗ | 2-candle reversal with matching closes |
| Pattern Recognition | CDLDARKCLOUDCOVER | ✓ | ✗ | Bearish reversal - black candle penetrating into white candle body |
| Pattern Recognition | CDLDOJI | ✓ | ✗ | Single candle with very small body (indecision) |
| Pattern Recognition | CDLDOJISTAR | ✓ | ✗ | 2-candle reversal with doji gapping away from trend |
| Pattern Recognition | CDLDRAGONFLYDOJI | ✓ | ✗ | Bullish doji with long lower shadow (T-shape) |
| Pattern Recognition | CDLENGULFING | ✓ | ✗ | 2-candle reversal where second body engulfs first completely |
| Pattern Recognition | CDLEVENINGDOJISTAR | ✓ | ✗ | Bearish 3-candle reversal with doji at top |
| Pattern Recognition | CDLEVENINGSTAR | ✓ | ✗ | Bearish 3-candle reversal with small star at top |
| Pattern Recognition | CDLGAPSIDESIDEWHITE | ✓ | ✗ | 3-candle continuation - two white candles side-by-side after gap |
| Pattern Recognition | CDLGRAVESTONEDOJI | ✓ | ✗ | Bearish doji with long upper shadow (inverted T-shape) |
| Pattern Recognition | CDLHAMMER | ✓ | ✗ | Bullish reversal - small body with long lower shadow |
| Pattern Recognition | CDLHANGINGMAN | ✓ | ✗ | Bearish reversal - visually identical to hammer but context differs |
| Pattern Recognition | CDLHARAMI | ✓ | ✗ | 2-candle reversal where second body contained within first |
| Pattern Recognition | CDLHARAMICROSS | ✓ | ✗ | Harami pattern with doji as second candle |
| Pattern Recognition | CDLHIGHWAVE | ✓ | ✗ | Doji with very long upper and lower shadows |
| Pattern Recognition | CDLHIKKAKE | ✓ | ✗ | 3-bar pattern with false inside day followed by breakout |
| Pattern Recognition | CDLHIKKAKEMOD | ✓ | ✗ | Modified Hikkake with delayed confirmation window |
| Pattern Recognition | CDLHOMINGPIGEON | ✓ | ✗ | Bullish reversal - 2 black candles, second contained in first body |
| Pattern Recognition | CDLIDENTICAL3CROWS | ✓ | ✗ | Bearish reversal - 3 black candles with similar closes |
| Pattern Recognition | CDLINNECK | ✓ | ✗ | Bearish continuation - white candle closes at prior low |
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
| Pattern Recognition | 60 | 54 (90%) | 22 (36%) |
| Price Transform | 5 | 5 (100%) | 0 (0%) |
| Statistic Functions | 6 | 5 (83%) | 0 (0%) |
| Statistics | 3 | 3 (100%) | 0 (0%) |
| Volatility Indicators | 2 | 1 (50%) | 0 (0%) |
| Volume Indicators | 3 | 3 (100%) | 0 (0%) |

### Overall

- **Total Functions**: 136
- **CPU (Numba) Implementations**: 128 (94%)
- **GPU (CuPy) Implementations**: 38 (27%)
- **Not Yet Implemented**: 8
- **Fully Implemented**: 128

## Notes

- **✓** indicates the function has the corresponding implementation
- **✗** indicates the function does not have the corresponding implementation
- Some functions like `MA`, `TEMA`, `T3`, `TRIMA` use other functions internally rather than having direct Numba implementations
- Functions marked as "Not yet implemented" will raise `NotImplementedError` when called

---
*Generated automatically from source code analysis*
