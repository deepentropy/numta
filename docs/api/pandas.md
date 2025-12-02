# Pandas Extension

numta provides a `.ta` accessor for Pandas DataFrames, making it easy to calculate technical indicators directly on your data.

## Setup

The accessor is automatically registered when you import numta:

```python
import pandas as pd
import numta  # Auto-registers the .ta accessor
```

## Basic Usage

### Creating a DataFrame

```python
import pandas as pd
import numpy as np
import numta

# Create sample OHLCV data
np.random.seed(42)
n = 100
df = pd.DataFrame({
    'open': np.random.uniform(100, 110, n),
    'high': np.random.uniform(110, 120, n),
    'low': np.random.uniform(90, 100, n),
    'close': np.random.uniform(100, 110, n),
    'volume': np.random.randint(1000, 10000, n)
})
```

### Calculating Indicators

#### Return as Series

```python
# Calculate indicators and return as Series
sma = df.ta.sma(timeperiod=20)
ema = df.ta.ema(timeperiod=12)
rsi = df.ta.rsi(timeperiod=14)
```

#### Append to DataFrame

```python
# Calculate and append to DataFrame
df.ta.sma(timeperiod=20, append=True)   # Adds 'SMA_20' column
df.ta.ema(timeperiod=12, append=True)   # Adds 'EMA_12' column
df.ta.rsi(timeperiod=14, append=True)   # Adds 'RSI_14' column
```

## Available Methods

### Moving Averages

| Method | Description |
|--------|-------------|
| `df.ta.sma(timeperiod)` | Simple Moving Average |
| `df.ta.ema(timeperiod)` | Exponential Moving Average |
| `df.ta.wma(timeperiod)` | Weighted Moving Average |
| `df.ta.dema(timeperiod)` | Double Exponential MA |
| `df.ta.tema(timeperiod)` | Triple Exponential MA |
| `df.ta.kama(timeperiod)` | Kaufman Adaptive MA |
| `df.ta.trima(timeperiod)` | Triangular MA |
| `df.ta.t3(timeperiod)` | T3 Moving Average |

### Momentum Indicators

| Method | Description |
|--------|-------------|
| `df.ta.rsi(timeperiod)` | Relative Strength Index |
| `df.ta.macd(fastperiod, slowperiod, signalperiod)` | MACD |
| `df.ta.stoch(fastk_period, slowk_period, slowd_period)` | Stochastic |
| `df.ta.adx(timeperiod)` | Average Directional Index |
| `df.ta.cci(timeperiod)` | Commodity Channel Index |
| `df.ta.mfi(timeperiod)` | Money Flow Index |
| `df.ta.mom(timeperiod)` | Momentum |
| `df.ta.roc(timeperiod)` | Rate of Change |
| `df.ta.willr(timeperiod)` | Williams %R |

### Volatility Indicators

| Method | Description |
|--------|-------------|
| `df.ta.atr(timeperiod)` | Average True Range |
| `df.ta.natr(timeperiod)` | Normalized ATR |
| `df.ta.trange()` | True Range |

### Bands

| Method | Description |
|--------|-------------|
| `df.ta.bbands(timeperiod, nbdevup, nbdevdn)` | Bollinger Bands |

### Volume Indicators

| Method | Description |
|--------|-------------|
| `df.ta.obv()` | On Balance Volume |
| `df.ta.ad()` | Accumulation/Distribution |
| `df.ta.adosc(fastperiod, slowperiod)` | A/D Oscillator |

### Trend Indicators

| Method | Description |
|--------|-------------|
| `df.ta.sar(acceleration, maximum)` | Parabolic SAR |

### Pattern Recognition

| Method | Description |
|--------|-------------|
| `df.ta.cdldoji()` | Doji Pattern |
| `df.ta.cdlengulfing()` | Engulfing Pattern |
| `df.ta.cdlhammer()` | Hammer Pattern |
| `df.ta.find_patterns()` | Find all patterns |
| `df.ta.find_harmonic_patterns()` | Find harmonic patterns |

## Multi-Output Indicators

Some indicators return multiple outputs:

```python
# MACD returns three columns
df.ta.macd(append=True)
# Adds: MACD_12_26_9, MACDsignal_12_26_9, MACDhist_12_26_9

# Bollinger Bands returns three columns
df.ta.bbands(append=True)
# Adds: BBU_5_2, BBM_5_2, BBL_5_2

# Stochastic returns two columns
df.ta.stoch(append=True)
# Adds: STOCHk_5_3_0_3_0, STOCHd_5_3_0_3_0
```

## Column Naming Convention

When using `append=True`, columns are named with the format:

```
{INDICATOR}_{PARAM1}_{PARAM2}...
```

Examples:
- `SMA_20` - SMA with timeperiod=20
- `EMA_12` - EMA with timeperiod=12
- `RSI_14` - RSI with timeperiod=14
- `MACD_12_26_9` - MACD with fast=12, slow=26, signal=9

## Custom Column Names

You can specify custom column names with the `column` parameter:

```python
df.ta.sma(timeperiod=20, append=True, column='my_sma')
# Adds 'my_sma' column instead of 'SMA_20'
```

## Working with Different Column Names

If your DataFrame uses different column names:

```python
df = pd.DataFrame({
    'price': prices,
    'hi': highs,
    'lo': lows,
    'vol': volumes
})

# Specify column mappings
df.ta.set_columns(close='price', high='hi', low='lo', volume='vol')

# Now indicators use the correct columns
df.ta.sma(timeperiod=20)
```

## Complete Example

```python
import pandas as pd
import numpy as np
import numta

# Create sample data
np.random.seed(42)
n = 200
df = pd.DataFrame({
    'open': np.random.uniform(100, 110, n),
    'high': np.random.uniform(110, 120, n),
    'low': np.random.uniform(90, 100, n),
    'close': np.random.uniform(100, 110, n),
    'volume': np.random.randint(1000, 10000, n)
})

# Add multiple indicators
df.ta.sma(timeperiod=20, append=True)
df.ta.sma(timeperiod=50, append=True)
df.ta.ema(timeperiod=12, append=True)
df.ta.rsi(timeperiod=14, append=True)
df.ta.macd(append=True)
df.ta.bbands(append=True)
df.ta.atr(timeperiod=14, append=True)

# View results
print(df.tail())

# Generate signals
df['signal'] = 0
df.loc[(df['SMA_20'] > df['SMA_50']) & (df['RSI_14'] < 30), 'signal'] = 1
df.loc[(df['SMA_20'] < df['SMA_50']) & (df['RSI_14'] > 70), 'signal'] = -1

# Filter signals
signals = df[df['signal'] != 0]
print(f"Number of signals: {len(signals)}")
```
