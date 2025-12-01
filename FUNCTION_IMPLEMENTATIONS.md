# numta Function Implementation Status

This document lists all implemented functions in numta.

## Technical Indicators

| Category | Function Name | Notes |
|----------|---------------|-------|
| Cycle Indicators | HT_DCPERIOD | Hilbert Transform - Dominant Cycle Period |
| Cycle Indicators | HT_DCPHASE | Hilbert Transform - Dominant Cycle Phase |
| Cycle Indicators | HT_PHASOR | Hilbert Transform - Phasor Components |
| Cycle Indicators | HT_SINE | Hilbert Transform - SineWave |
| Cycle Indicators | HT_TRENDLINE | Hilbert Transform - Instantaneous Trendline |
| Cycle Indicators | HT_TRENDMODE | Hilbert Transform - Trend vs Cycle Mode |
| Math Operators | MAX | Highest value over a specified period |
| Math Operators | MAXINDEX | Index of highest value over a specified period |
| Math Operators | MIN | Lowest value over a specified period |
| Math Operators | MININDEX | Index of lowest value over a specified period |
| Math Operators | MINMAX | Lowest and highest values over a specified period |
| Math Operators | MINMAXINDEX | Indexes of lowest and highest values |
| Math Operators | SUM | Summation |
| Momentum Indicators | ADX | Average Directional Movement Index |
| Momentum Indicators | ADXR | Average Directional Movement Index Rating |
| Momentum Indicators | APO | Absolute Price Oscillator |
| Momentum Indicators | AROON | Aroon |
| Momentum Indicators | AROONOSC | Aroon Oscillator |
| Momentum Indicators | ATR | Average True Range |
| Momentum Indicators | BOP | Balance Of Power |
| Momentum Indicators | CCI | Commodity Channel Index |
| Momentum Indicators | CMO | Chande Momentum Oscillator |
| Momentum Indicators | DX | Directional Movement Index |
| Momentum Indicators | MACD | Moving Average Convergence/Divergence |
| Momentum Indicators | MACDEXT | MACD with controllable MA type |
| Momentum Indicators | MACDFIX | Moving Average Convergence/Divergence Fix 12/26 |
| Momentum Indicators | MFI | Money Flow Index |
| Momentum Indicators | MINUS_DI | Minus Directional Indicator |
| Momentum Indicators | MINUS_DM | Minus Directional Movement |
| Momentum Indicators | MOM | Momentum |
| Momentum Indicators | PLUS_DI | Plus Directional Indicator |
| Momentum Indicators | PLUS_DM | Plus Directional Movement |
| Momentum Indicators | PPO | Percentage Price Oscillator |
| Momentum Indicators | ROC | Rate of Change |
| Momentum Indicators | ROCP | Rate of Change Percentage |
| Momentum Indicators | ROCR | Rate of Change Ratio |
| Momentum Indicators | ROCR100 | Rate of Change Ratio 100 scale |
| Momentum Indicators | RSI | Relative Strength Index |
| Momentum Indicators | STOCH | Stochastic |
| Momentum Indicators | STOCHF | Stochastic Fast |
| Momentum Indicators | STOCHRSI | Stochastic Relative Strength Index |
| Momentum Indicators | TRIX | 1-day Rate-Of-Change of Triple Smooth EMA |
| Momentum Indicators | ULTOSC | Ultimate Oscillator |
| Momentum Indicators | WILLR | Williams' %R |
| Overlap Studies | BBANDS | Bollinger Bands |
| Overlap Studies | DEMA | Double Exponential Moving Average |
| Overlap Studies | EMA | Exponential Moving Average |
| Overlap Studies | KAMA | Kaufman Adaptive Moving Average |
| Overlap Studies | MA | All Moving Average types |
| Overlap Studies | MAMA | MESA Adaptive Moving Average |
| Overlap Studies | SAR | Parabolic SAR |
| Overlap Studies | SAREXT | Parabolic SAR - Extended |
| Overlap Studies | SMA | Simple Moving Average |
| Overlap Studies | T3 | Triple Exponential Moving Average (T3) |
| Overlap Studies | TEMA | Triple Exponential Moving Average |
| Overlap Studies | TRIMA | Triangular Moving Average |
| Overlap Studies | WMA | Weighted Moving Average |
| Price Transform | MEDPRICE | Median Price |
| Price Transform | MIDPOINT | MidPoint over period |
| Price Transform | MIDPRICE | Midpoint Price over period |
| Price Transform | TYPPRICE | Typical Price |
| Price Transform | WCLPRICE | Weighted Close Price |
| Statistic Functions | BETA | Beta coefficient |
| Statistic Functions | CORREL | Pearson's Correlation Coefficient |
| Statistic Functions | LINEARREG | Linear Regression |
| Statistic Functions | LINEARREG_ANGLE | Linear Regression Angle |
| Statistic Functions | LINEARREG_INTERCEPT | Linear Regression Intercept |
| Statistic Functions | LINEARREG_SLOPE | Linear Regression Slope |
| Statistic Functions | STDDEV | Standard Deviation |
| Statistic Functions | TSF | Time Series Forecast |
| Statistic Functions | VAR | Variance |
| Volatility Indicators | NATR | Normalized Average True Range |
| Volatility Indicators | TRANGE | True Range |
| Volume Indicators | AD | Chaikin A/D Line |
| Volume Indicators | ADOSC | Chaikin A/D Oscillator |
| Volume Indicators | OBV | On Balance Volume |

## Candlestick Pattern Recognition (60+ patterns)

| Function | Description |
|----------|-------------|
| CDL2CROWS | Two Crows |
| CDL3BLACKCROWS | Three Black Crows |
| CDL3INSIDE | Three Inside Up/Down |
| CDL3OUTSIDE | Three Outside Up/Down |
| CDL3STARSINSOUTH | Three Stars In The South |
| CDL3WHITESOLDIERS | Three Advancing White Soldiers |
| CDLABANDONEDBABY | Abandoned Baby |
| CDLADVANCEBLOCK | Advance Block |
| CDLBELTHOLD | Belt-hold |
| CDLBREAKAWAY | Breakaway |
| CDLCLOSINGMARUBOZU | Closing Marubozu |
| CDLCONCEALBABYSWALL | Concealing Baby Swallow |
| CDLCOUNTERATTACK | Counterattack |
| CDLDARKCLOUDCOVER | Dark Cloud Cover |
| CDLDOJI | Doji |
| CDLDOJISTAR | Doji Star |
| CDLDRAGONFLYDOJI | Dragonfly Doji |
| CDLENGULFING | Engulfing Pattern |
| CDLEVENINGDOJISTAR | Evening Doji Star |
| CDLEVENINGSTAR | Evening Star |
| CDLGAPSIDESIDEWHITE | Up/Down-gap side-by-side white lines |
| CDLGRAVESTONEDOJI | Gravestone Doji |
| CDLHAMMER | Hammer |
| CDLHANGINGMAN | Hanging Man |
| CDLHARAMI | Harami Pattern |
| CDLHARAMICROSS | Harami Cross Pattern |
| CDLHIGHWAVE | High-Wave Candle |
| CDLHIKKAKE | Hikkake Pattern |
| CDLHIKKAKEMOD | Modified Hikkake Pattern |
| CDLHOMINGPIGEON | Homing Pigeon |
| CDLIDENTICAL3CROWS | Identical Three Crows |
| CDLINNECK | In-Neck Pattern |
| CDLINVERTEDHAMMER | Inverted Hammer |
| CDLKICKING | Kicking |
| CDLKICKINGBYLENGTH | Kicking by Length |
| CDLLADDERBOTTOM | Ladder Bottom |
| CDLLONGLEGGEDDOJI | Long Legged Doji |
| CDLLONGLINE | Long Line Candle |
| CDLMARUBOZU | Marubozu |
| CDLMATCHINGLOW | Matching Low |
| CDLMATHOLD | Mat Hold |
| CDLMORNINGDOJISTAR | Morning Doji Star |
| CDLMORNINGSTAR | Morning Star |
| CDLONNECK | On-Neck Pattern |
| CDLPIERCING | Piercing Pattern |
| CDLRICKSHAWMAN | Rickshaw Man |
| CDLRISEFALL3METHODS | Rising/Falling Three Methods |
| CDLSEPARATINGLINES | Separating Lines |
| CDLSHOOTINGSTAR | Shooting Star |
| CDLSHORTLINE | Short Line Candle |
| CDLSPINNINGTOP | Spinning Top |
| CDLSTALLEDPATTERN | Stalled Pattern |
| CDLSTICKSANDWICH | Stick Sandwich |
| CDLTAKURI | Takuri |
| CDLTASUKIGAP | Tasuki Gap |
| CDLTHRUSTING | Thrusting Pattern |
| CDLTRISTAR | Tristar Pattern |
| CDLUNIQUE3RIVER | Unique 3 River |
| CDLUPSIDEGAP2CROWS | Upside Gap Two Crows |
| CDLXSIDEGAP3METHODS | Upside/Downside Gap Three Methods |

## Chart Pattern Detection

| Function | Description |
|----------|-------------|
| find_swing_highs | Detect swing high points |
| find_swing_lows | Detect swing low points |
| find_swing_points | Detect all swing points |
| detect_head_shoulders | Head and Shoulders pattern |
| detect_inverse_head_shoulders | Inverse Head and Shoulders pattern |
| detect_double_top | Double Top pattern |
| detect_double_bottom | Double Bottom pattern |
| detect_triple_top | Triple Top pattern |
| detect_triple_bottom | Triple Bottom pattern |
| detect_triangle | Ascending, Descending, Symmetrical triangles |
| detect_wedge | Rising and Falling wedges |
| detect_flag | Bull Flag, Bear Flag, Pennant |
| detect_vcp | Volatility Contraction Pattern |

## Harmonic Pattern Detection

| Function | Description |
|----------|-------------|
| detect_gartley | Gartley pattern (61.8% XA retracement) |
| detect_butterfly | Butterfly pattern (78.6% XA retracement) |
| detect_bat | Bat pattern (38.2-50% XA retracement) |
| detect_crab | Crab pattern (161.8% XA extension) |
| detect_harmonic_patterns | Detect all harmonic patterns |
| fibonacci_retracement | Calculate Fibonacci retracement levels |
| fibonacci_extension | Calculate Fibonacci extension levels |

## Streaming Indicators

Real-time indicators that update efficiently with each new data point.

| Function | Description |
|----------|-------------|
| StreamingSMA | Simple Moving Average |
| StreamingEMA | Exponential Moving Average |
| StreamingBBANDS | Bollinger Bands |
| StreamingDEMA | Double Exponential Moving Average |
| StreamingTEMA | Triple Exponential Moving Average |
| StreamingWMA | Weighted Moving Average |
| StreamingRSI | Relative Strength Index |
| StreamingMACD | Moving Average Convergence/Divergence |
| StreamingSTOCH | Stochastic Oscillator |
| StreamingMOM | Momentum |
| StreamingROC | Rate of Change |
| StreamingATR | Average True Range |
| StreamingTRANGE | True Range |
| StreamingOBV | On Balance Volume |
| StreamingAD | Accumulation/Distribution |

## Summary Statistics

| Category | Count |
|----------|-------|
| Technical Indicators | 76 |
| Candlestick Patterns | 60 |
| Chart Pattern Functions | 14 |
| Harmonic Pattern Functions | 7 |
| Streaming Indicators | 15 |
| **Total** | **172** |

## Notes

- All technical indicators support NumPy arrays and Python lists
- Pattern recognition returns +100 (bullish), -100 (bearish), or 0 (no pattern)
- Chart patterns return confidence scores (0-1)
- Streaming indicators are optimized for real-time processing with O(1) updates
