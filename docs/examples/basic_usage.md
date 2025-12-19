# Examples

This page provides comprehensive examples of using numta for technical analysis.

## Basic Indicator Usage

### Moving Averages

```python
import numpy as np
from numta import SMA, EMA, WMA, DEMA, TEMA, KAMA

# Create sample price data
np.random.seed(42)
close = np.cumsum(np.random.randn(100)) + 100

# Calculate various moving averages
sma_20 = SMA(close, timeperiod=20)
ema_12 = EMA(close, timeperiod=12)
wma_20 = WMA(close, timeperiod=20)
dema_20 = DEMA(close, timeperiod=20)
tema_20 = TEMA(close, timeperiod=20)
kama_30 = KAMA(close, timeperiod=30)

# Compare values at the end
print(f"Close: {close[-1]:.2f}")
print(f"SMA(20): {sma_20[-1]:.2f}")
print(f"EMA(12): {ema_12[-1]:.2f}")
print(f"WMA(20): {wma_20[-1]:.2f}")
print(f"DEMA(20): {dema_20[-1]:.2f}")
print(f"TEMA(20): {tema_20[-1]:.2f}")
print(f"KAMA(30): {kama_30[-1]:.2f}")
```

### Momentum Indicators

```python
import numpy as np
from numta import RSI, MACD, STOCH, ADX, CCI

# Create sample OHLC data
np.random.seed(42)
n = 100
close = np.cumsum(np.random.randn(n)) + 100
high = close + np.abs(np.random.randn(n))
low = close - np.abs(np.random.randn(n))

# RSI
rsi = RSI(close, timeperiod=14)
print(f"RSI: {rsi[-1]:.2f}")

# Check overbought/oversold
if rsi[-1] > 70:
    print("RSI indicates overbought condition")
elif rsi[-1] < 30:
    print("RSI indicates oversold condition")

# MACD
macd, signal, hist = MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
print(f"MACD: {macd[-1]:.4f}, Signal: {signal[-1]:.4f}, Histogram: {hist[-1]:.4f}")

# Check MACD crossover
if macd[-1] > signal[-1] and macd[-2] <= signal[-2]:
    print("Bullish MACD crossover")
elif macd[-1] < signal[-1] and macd[-2] >= signal[-2]:
    print("Bearish MACD crossover")

# Stochastic
slowk, slowd = STOCH(high, low, close)
print(f"Stochastic %K: {slowk[-1]:.2f}, %D: {slowd[-1]:.2f}")

# ADX (trend strength)
adx = ADX(high, low, close, timeperiod=14)
print(f"ADX: {adx[-1]:.2f}")
if adx[-1] > 25:
    print("Strong trend detected")

# CCI
cci = CCI(high, low, close, timeperiod=14)
print(f"CCI: {cci[-1]:.2f}")
```

### Bollinger Bands

```python
import numpy as np
from numta import BBANDS

# Create sample data
np.random.seed(42)
close = np.cumsum(np.random.randn(100)) + 100

# Calculate Bollinger Bands
upper, middle, lower = BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)

# Check position relative to bands
current_price = close[-1]
if current_price > upper[-1]:
    print(f"Price ({current_price:.2f}) is above upper band ({upper[-1]:.2f}) - Overbought")
elif current_price < lower[-1]:
    print(f"Price ({current_price:.2f}) is below lower band ({lower[-1]:.2f}) - Oversold")
else:
    print(f"Price ({current_price:.2f}) is within bands")

# Calculate bandwidth (volatility measure)
bandwidth = (upper[-1] - lower[-1]) / middle[-1] * 100
print(f"Bandwidth: {bandwidth:.2f}%")
```

## Trading Strategy Example

### Golden Cross Strategy

```python
import numpy as np
from numta import SMA

def golden_cross_strategy(close):
    """
    Simple Golden Cross / Death Cross strategy.
    
    Returns:
        signals: 1 for buy, -1 for sell, 0 for hold
    """
    sma_50 = SMA(close, timeperiod=50)
    sma_200 = SMA(close, timeperiod=200)
    
    signals = np.zeros(len(close))
    
    for i in range(1, len(close)):
        # Skip if not enough data
        if np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            continue
            
        # Golden Cross (bullish)
        if sma_50[i] > sma_200[i] and sma_50[i-1] <= sma_200[i-1]:
            signals[i] = 1
        # Death Cross (bearish)
        elif sma_50[i] < sma_200[i] and sma_50[i-1] >= sma_200[i-1]:
            signals[i] = -1
    
    return signals

# Generate sample data
np.random.seed(42)
close = np.cumsum(np.random.randn(300)) + 100

# Run strategy
signals = golden_cross_strategy(close)

# Find signal indices
buy_signals = np.where(signals == 1)[0]
sell_signals = np.where(signals == -1)[0]

print(f"Buy signals at: {buy_signals}")
print(f"Sell signals at: {sell_signals}")
```

### RSI Mean Reversion Strategy

```python
import numpy as np
from numta import RSI, SMA

def rsi_mean_reversion(close, rsi_period=14, rsi_oversold=30, rsi_overbought=70):
    """
    RSI mean reversion strategy with trend filter.
    """
    rsi = RSI(close, timeperiod=rsi_period)
    sma_200 = SMA(close, timeperiod=200)
    
    signals = np.zeros(len(close))
    
    for i in range(200, len(close)):
        # Only trade in direction of trend
        uptrend = close[i] > sma_200[i]
        downtrend = close[i] < sma_200[i]
        
        # Buy signal: RSI crosses above oversold in uptrend
        if uptrend and rsi[i] > rsi_oversold and rsi[i-1] <= rsi_oversold:
            signals[i] = 1
        # Sell signal: RSI crosses below overbought in downtrend
        elif downtrend and rsi[i] < rsi_overbought and rsi[i-1] >= rsi_overbought:
            signals[i] = -1
    
    return signals

# Generate sample data
np.random.seed(42)
close = np.cumsum(np.random.randn(500)) + 100

# Run strategy
signals = rsi_mean_reversion(close)

# Count signals
print(f"Total buy signals: {np.sum(signals == 1)}")
print(f"Total sell signals: {np.sum(signals == -1)}")
```

## Pandas Integration Example

```python
import pandas as pd
import numpy as np
import numta

# Create DataFrame
np.random.seed(42)
n = 200
df = pd.DataFrame({
    'open': np.random.uniform(100, 110, n),
    'high': np.random.uniform(110, 120, n),
    'low': np.random.uniform(90, 100, n),
    'close': np.random.uniform(100, 110, n),
    'volume': np.random.randint(1000, 10000, n)
})

# Add indicators
df.ta.sma(timeperiod=20, append=True)
df.ta.sma(timeperiod=50, append=True)
df.ta.rsi(timeperiod=14, append=True)
df.ta.macd(append=True)
df.ta.bbands(timeperiod=20, append=True)
df.ta.atr(timeperiod=14, append=True)

# Generate signals
df['signal'] = 0

# Buy when SMA20 > SMA50, RSI < 30, price at lower Bollinger Band
buy_condition = (
    (df['SMA_20'] > df['SMA_50']) & 
    (df['RSI_14'] < 30) & 
    (df['close'] <= df['BBL_20_2.0'])
)
df.loc[buy_condition, 'signal'] = 1

# Sell when SMA20 < SMA50, RSI > 70, price at upper Bollinger Band
sell_condition = (
    (df['SMA_20'] < df['SMA_50']) & 
    (df['RSI_14'] > 70) & 
    (df['close'] >= df['BBU_20_2.0'])
)
df.loc[sell_condition, 'signal'] = -1

# Show signals
signals_df = df[df['signal'] != 0][['close', 'SMA_20', 'RSI_14', 'signal']]
print(signals_df)
```

## Streaming Example

```python
from numta.streaming import StreamingSMA, StreamingRSI, StreamingBBANDS

class LiveTradingBot:
    def __init__(self):
        self.sma_20 = StreamingSMA(timeperiod=20)
        self.sma_50 = StreamingSMA(timeperiod=50)
        self.rsi = StreamingRSI(timeperiod=14)
        self.bbands = StreamingBBANDS(timeperiod=20)
        self.position = 0  # 0 = flat, 1 = long, -1 = short
        
    def on_price(self, price):
        """Process new price and generate signals."""
        # Update indicators
        sma_20 = self.sma_20.update(price)
        sma_50 = self.sma_50.update(price)
        rsi = self.rsi.update(price)
        upper, middle, lower = self.bbands.update(price)
        
        # Check if indicators are ready
        if not all([self.sma_20.ready, self.sma_50.ready, 
                    self.rsi.ready, self.bbands.ready]):
            return None
        
        signal = None
        
        # Entry logic
        if self.position == 0:
            # Buy signal
            if sma_20 > sma_50 and rsi < 30 and price <= lower:
                self.position = 1
                signal = {'action': 'BUY', 'price': price, 'rsi': rsi}
            # Sell signal
            elif sma_20 < sma_50 and rsi > 70 and price >= upper:
                self.position = -1
                signal = {'action': 'SELL', 'price': price, 'rsi': rsi}
        
        # Exit logic
        elif self.position == 1 and (rsi > 70 or price >= upper):
            self.position = 0
            signal = {'action': 'CLOSE_LONG', 'price': price, 'rsi': rsi}
        elif self.position == -1 and (rsi < 30 or price <= lower):
            self.position = 0
            signal = {'action': 'CLOSE_SHORT', 'price': price, 'rsi': rsi}
        
        return signal

# Simulate streaming
import numpy as np
np.random.seed(42)

bot = LiveTradingBot()
prices = np.cumsum(np.random.randn(200)) + 100

for i, price in enumerate(prices):
    signal = bot.on_price(price)
    if signal:
        print(f"Bar {i}: {signal}")
```

## Pattern Recognition Example

```python
import numpy as np
from numta import CDLDOJI, CDLENGULFING, CDLHAMMER, CDLMORNINGSTAR

# Create sample OHLC data
np.random.seed(42)
n = 100
close = np.cumsum(np.random.randn(n)) + 100
open_ = close + np.random.randn(n) * 0.5
high = np.maximum(open_, close) + np.abs(np.random.randn(n))
low = np.minimum(open_, close) - np.abs(np.random.randn(n))

# Detect patterns
doji = CDLDOJI(open_, high, low, close)
engulfing = CDLENGULFING(open_, high, low, close)
hammer = CDLHAMMER(open_, high, low, close)

# Find pattern occurrences
doji_indices = np.where(doji != 0)[0]
engulfing_bullish = np.where(engulfing == 100)[0]
engulfing_bearish = np.where(engulfing == -100)[0]
hammer_indices = np.where(hammer != 0)[0]

print(f"Doji patterns: {len(doji_indices)}")
print(f"Bullish engulfing: {len(engulfing_bullish)}")
print(f"Bearish engulfing: {len(engulfing_bearish)}")
print(f"Hammer patterns: {len(hammer_indices)}")

# Combine patterns for stronger signals
combined_bullish = np.where(
    (engulfing == 100) | (hammer == 100)
)[0]
print(f"\nCombined bullish signals at: {combined_bullish}")
```
