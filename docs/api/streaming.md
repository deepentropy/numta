# Streaming Indicators

Streaming indicators allow real-time calculation of technical indicators as new price data arrives, without recalculating the entire history.

## Overview

Streaming indicators are ideal for:

- Real-time trading systems
- Live market data processing
- Memory-efficient indicator calculation
- Event-driven architectures

## Available Streaming Indicators

### Moving Averages

#### StreamingSMA

```python
from numta.streaming import StreamingSMA

sma = StreamingSMA(timeperiod=20)

for price in price_stream:
    value = sma.update(price)
    if sma.ready:
        print(f"SMA: {value:.2f}")
```

#### StreamingEMA

```python
from numta.streaming import StreamingEMA

ema = StreamingEMA(timeperiod=12)

for price in price_stream:
    value = ema.update(price)
    if ema.ready:
        print(f"EMA: {value:.2f}")
```

### Momentum Indicators

#### StreamingRSI

```python
from numta.streaming import StreamingRSI

rsi = StreamingRSI(timeperiod=14)

for price in price_stream:
    value = rsi.update(price)
    if rsi.ready:
        print(f"RSI: {value:.2f}")
```

#### StreamingMACD

```python
from numta.streaming import StreamingMACD

macd = StreamingMACD(fastperiod=12, slowperiod=26, signalperiod=9)

for price in price_stream:
    macd_value, signal_value, hist_value = macd.update(price)
    if macd.ready:
        print(f"MACD: {macd_value:.4f}, Signal: {signal_value:.4f}")
```

### Volatility Indicators

#### StreamingATR

```python
from numta.streaming import StreamingATR

atr = StreamingATR(timeperiod=14)

for high, low, close in ohlc_stream:
    value = atr.update(high, low, close)
    if atr.ready:
        print(f"ATR: {value:.2f}")
```

### Bands

#### StreamingBBANDS

```python
from numta.streaming import StreamingBBANDS

bbands = StreamingBBANDS(timeperiod=20, nbdevup=2, nbdevdn=2)

for price in price_stream:
    upper, middle, lower = bbands.update(price)
    if bbands.ready:
        print(f"Upper: {upper:.2f}, Middle: {middle:.2f}, Lower: {lower:.2f}")
```

## Common Interface

All streaming indicators share a common interface:

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `ready` | bool | Whether enough data has been received |
| `value` | float | Current indicator value |
| `count` | int | Number of data points processed |

### Methods

| Method | Description |
|--------|-------------|
| `update(price)` | Process new price data and return current value |
| `reset()` | Reset the indicator state |

## Usage Example

```python
from numta.streaming import StreamingSMA, StreamingRSI, StreamingMACD

# Initialize streaming indicators
sma_20 = StreamingSMA(timeperiod=20)
sma_50 = StreamingSMA(timeperiod=50)
rsi = StreamingRSI(timeperiod=14)
macd = StreamingMACD()

# Simulate streaming data
import numpy as np
np.random.seed(42)
prices = np.cumsum(np.random.randn(100)) + 100

for price in prices:
    sma_20_value = sma_20.update(price)
    sma_50_value = sma_50.update(price)
    rsi_value = rsi.update(price)
    macd_value, signal_value, _ = macd.update(price)
    
    if sma_20.ready and sma_50.ready and rsi.ready:
        # Trading logic
        if sma_20_value > sma_50_value and rsi_value < 30:
            print(f"Potential buy signal at price {price:.2f}")
        elif sma_20_value < sma_50_value and rsi_value > 70:
            print(f"Potential sell signal at price {price:.2f}")
```

## Multi-Indicator Strategy

```python
from numta.streaming import StreamingSMA, StreamingRSI, StreamingATR

class TradingStrategy:
    def __init__(self):
        self.sma_fast = StreamingSMA(timeperiod=10)
        self.sma_slow = StreamingSMA(timeperiod=30)
        self.rsi = StreamingRSI(timeperiod=14)
        self.atr = StreamingATR(timeperiod=14)
        self.position = 0
    
    def on_price(self, high, low, close):
        # Update indicators
        fast = self.sma_fast.update(close)
        slow = self.sma_slow.update(close)
        rsi = self.rsi.update(close)
        atr = self.atr.update(high, low, close)
        
        # Check if all indicators are ready
        if not all([self.sma_fast.ready, self.sma_slow.ready, 
                    self.rsi.ready, self.atr.ready]):
            return None
        
        # Generate signal
        if self.position == 0:
            if fast > slow and rsi < 30:
                return {'action': 'BUY', 'stop_loss': close - 2 * atr}
            elif fast < slow and rsi > 70:
                return {'action': 'SELL', 'stop_loss': close + 2 * atr}
        
        return None

# Usage
strategy = TradingStrategy()
for high, low, close in ohlc_data:
    signal = strategy.on_price(high, low, close)
    if signal:
        print(f"Signal: {signal}")
```

## Performance Considerations

Streaming indicators are optimized for:

1. **Memory efficiency**: Only stores necessary historical data
2. **Speed**: O(1) updates for most indicators
3. **No recalculation**: Updates incrementally without full recalculation

For batch processing of historical data, use the regular indicator functions instead.
