# Quick Start

This guide will help you get started with numta in just a few minutes.

## Basic Usage

### Import and Create Data

```python
import numpy as np
from numta import SMA, EMA, RSI, MACD, BBANDS

# Create sample OHLCV data
np.random.seed(42)
n = 100
close = np.cumsum(np.random.randn(n)) + 100
high = close + np.abs(np.random.randn(n))
low = close - np.abs(np.random.randn(n))
open_ = (high + low) / 2
volume = np.random.randint(1000, 10000, n).astype(float)
```

### Calculate Moving Averages

```python
# Simple Moving Average
sma_20 = SMA(close, timeperiod=20)

# Exponential Moving Average
ema_12 = EMA(close, timeperiod=12)
ema_26 = EMA(close, timeperiod=26)

print(f"SMA(20): {sma_20[-1]:.2f}")
print(f"EMA(12): {ema_12[-1]:.2f}")
```

### Calculate Momentum Indicators

```python
# Relative Strength Index
rsi = RSI(close, timeperiod=14)
print(f"RSI: {rsi[-1]:.2f}")

# MACD
macd, signal, hist = MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
print(f"MACD: {macd[-1]:.4f}, Signal: {signal[-1]:.4f}")
```

### Calculate Bollinger Bands

```python
# Bollinger Bands
upper, middle, lower = BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
print(f"Upper: {upper[-1]:.2f}, Middle: {middle[-1]:.2f}, Lower: {lower[-1]:.2f}")
```

## Using with Pandas

```python
import pandas as pd
import numta  # Auto-registers the .ta accessor

# Create DataFrame
df = pd.DataFrame({
    'open': open_,
    'high': high,
    'low': low,
    'close': close,
    'volume': volume
})

# Calculate indicators and append to DataFrame
df.ta.sma(timeperiod=20, append=True)   # Adds 'SMA_20' column
df.ta.ema(timeperiod=12, append=True)   # Adds 'EMA_12' column
df.ta.rsi(timeperiod=14, append=True)   # Adds 'RSI_14' column
df.ta.macd(append=True)                 # Adds MACD columns

# View results
print(df[['close', 'SMA_20', 'EMA_12', 'RSI_14']].tail())
```

## Streaming Mode

For real-time applications:

```python
from numta.streaming import StreamingSMA, StreamingRSI

# Initialize streaming indicators
sma = StreamingSMA(timeperiod=20)
rsi = StreamingRSI(timeperiod=14)

# Simulate streaming data
for price in close:
    sma_value = sma.update(price)
    rsi_value = rsi.update(price)
    
    if sma.ready and rsi.ready:
        print(f"Price: {price:.2f}, SMA: {sma_value:.2f}, RSI: {rsi_value:.2f}")
```

## Pattern Recognition

### Candlestick Patterns

```python
from numta import CDLDOJI, CDLENGULFING, CDLHAMMER

# Detect patterns
doji = CDLDOJI(open_, high, low, close)
engulfing = CDLENGULFING(open_, high, low, close)
hammer = CDLHAMMER(open_, high, low, close)

# Pattern values: +100 (bullish), -100 (bearish), 0 (no pattern)
pattern_days = np.where(doji != 0)[0]
print(f"Doji patterns found at indices: {pattern_days}")
```

## What's Next?

- Explore the [API Reference](../api/index.md) for all available indicators
- Check the [Examples](../examples/basic_usage.md) for more detailed use cases
- See [Performance](../performance.md) for optimization tips
