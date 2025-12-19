# API Reference

numta provides a comprehensive set of technical analysis indicators organized by category.

## Indicator Categories

### [Overlap Studies](overlap.md)

Indicators that overlay directly on price charts:

- **SMA** - Simple Moving Average
- **EMA** - Exponential Moving Average
- **WMA** - Weighted Moving Average
- **DEMA** - Double Exponential Moving Average
- **TEMA** - Triple Exponential Moving Average
- **KAMA** - Kaufman Adaptive Moving Average
- **MAMA** - MESA Adaptive Moving Average
- **T3** - Triple Exponential T3
- **TRIMA** - Triangular Moving Average
- **BBANDS** - Bollinger Bands
- **SAR** - Parabolic SAR
- **SAREXT** - Parabolic SAR Extended
- **MA** - Generic Moving Average

### [Momentum Indicators](momentum.md)

Indicators measuring price momentum and strength:

- **RSI** - Relative Strength Index
- **MACD** - Moving Average Convergence/Divergence
- **STOCH** - Stochastic Oscillator
- **STOCHF** - Stochastic Fast
- **ADX** - Average Directional Movement Index
- **ADXR** - Average Directional Movement Rating
- **CCI** - Commodity Channel Index
- **MFI** - Money Flow Index
- **MOM** - Momentum
- **ROC** - Rate of Change
- **WILLR** - Williams %R
- **APO** - Absolute Price Oscillator
- **PPO** - Percentage Price Oscillator
- **AROON** - Aroon Indicator
- **AROONOSC** - Aroon Oscillator
- **BOP** - Balance of Power
- **DX** - Directional Movement Index
- **PLUS_DI** - Plus Directional Indicator
- **MINUS_DI** - Minus Directional Indicator
- **PLUS_DM** - Plus Directional Movement
- **MINUS_DM** - Minus Directional Movement
- **TRIX** - Triple Exponential Moving Average ROC
- **ULTOSC** - Ultimate Oscillator

### [Volume Indicators](volume.md)

Indicators based on trading volume:

- **OBV** - On Balance Volume
- **AD** - Chaikin Accumulation/Distribution Line
- **ADOSC** - Chaikin A/D Oscillator

### [Volatility Indicators](volatility.md)

Indicators measuring price volatility:

- **ATR** - Average True Range
- **NATR** - Normalized Average True Range
- **TRANGE** - True Range

### [Cycle Indicators](cycle.md)

Hilbert Transform based cycle analysis:

- **HT_DCPERIOD** - Dominant Cycle Period
- **HT_DCPHASE** - Dominant Cycle Phase
- **HT_PHASOR** - Phasor Components
- **HT_SINE** - SineWave
- **HT_TRENDLINE** - Instantaneous Trendline
- **HT_TRENDMODE** - Trend vs Cycle Mode

### [Statistical Functions](statistics.md)

Statistical analysis functions:

- **LINEARREG** - Linear Regression
- **STDDEV** - Standard Deviation
- **VAR** - Variance
- **CORREL** - Correlation
- **BETA** - Beta
- **TSF** - Time Series Forecast

### [Math Operators](math.md)

Mathematical operations:

- **MAX** - Highest Value
- **MIN** - Lowest Value
- **SUM** - Summation
- **MINMAX** - Min and Max Values

### [Price Transform](price_transform.md)

Price transformation functions:

- **MEDPRICE** - Median Price
- **TYPPRICE** - Typical Price
- **WCLPRICE** - Weighted Close Price
- **MIDPOINT** - Midpoint over Period
- **MIDPRICE** - Midpoint Price over Period

### [Pattern Recognition](patterns.md)

Candlestick and chart pattern detection:

- 60+ candlestick patterns (CDL*)
- Chart pattern detection

### [Streaming Indicators](streaming.md)

Real-time indicator updates:

- StreamingSMA, StreamingEMA
- StreamingRSI, StreamingMACD
- And more...

### [Pandas Extension](pandas.md)

DataFrame `.ta` accessor for easy integration.

## Common Parameters

Most indicators share these common parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `close` | array-like | Close prices array |
| `high` | array-like | High prices array |
| `low` | array-like | Low prices array |
| `open` | array-like | Open prices array |
| `volume` | array-like | Volume array |
| `timeperiod` | int | Number of periods for calculation |

## Return Values

- Single-output indicators return a NumPy array
- Multi-output indicators return a tuple of NumPy arrays
- NaN values are used for the lookback period

## TA-Lib Compatibility

All indicators are designed to be compatible with TA-Lib signatures for easy migration:

```python
# TA-Lib style
from numta import SMA, EMA, RSI
sma = SMA(close, timeperiod=20)
ema = EMA(close, timeperiod=12)
rsi = RSI(close, timeperiod=14)
```
