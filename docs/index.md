# numta

**Pure Python Technical Analysis Library**

A modern, high-performance alternative to TA-Lib with zero C dependencies.

---

## Highlights

<div class="grid cards" markdown>

-   :material-language-python: **Pure Python**

    ---

    No C compiler required, works everywhere Python runs

-   :material-lightning-bolt: **Fast**

    ---

    5-10x speedup with optional Numba JIT compilation

-   :material-check-all: **Complete**

    ---

    130+ indicators, 60+ candlestick patterns, chart pattern detection

-   :material-new-box: **Modern**

    ---

    Pandas integration, streaming support, Jupyter visualization

</div>

## Quick Example

```python
import numpy as np
from numta import SMA, EMA, RSI

# Create sample price data
close_prices = np.random.uniform(100, 200, 100)

# Calculate indicators
sma = SMA(close_prices, timeperiod=20)
ema = EMA(close_prices, timeperiod=12)
rsi = RSI(close_prices, timeperiod=14)
```

## Installation

```bash
# Basic installation
pip install numta

# With Numba for 5-10x speedup
pip install "numta[numba]"

# With pandas integration
pip install "numta[pandas]"

# Full installation with all features
pip install "numta[full]"
```

## Features

### Technical Indicators

numta provides 130+ technical indicators across multiple categories:

| Category | Indicators |
|----------|-----------|
| **Overlap Studies** | SMA, EMA, DEMA, TEMA, WMA, BBANDS, KAMA, MAMA, T3, SAR |
| **Momentum** | RSI, MACD, STOCH, ADX, CCI, MFI, ROC, MOM, WILLR |
| **Volume** | OBV, AD, ADOSC |
| **Volatility** | ATR, NATR, TRANGE |
| **Cycle** | Hilbert Transform functions |
| **Statistical** | LINEARREG, STDDEV, VAR, CORREL, BETA |

### Pandas Integration

```python
import pandas as pd
import numta  # Auto-registers the .ta accessor

df = pd.DataFrame({
    'open': [...], 'high': [...], 'low': [...],
    'close': [...], 'volume': [...]
})

# Calculate and return as Series
sma = df.ta.sma(timeperiod=20)

# Append indicators to DataFrame
df.ta.sma(timeperiod=20, append=True)   # Adds 'SMA_20'
df.ta.rsi(timeperiod=14, append=True)   # Adds 'RSI_14'
df.ta.macd(append=True)                 # Adds MACD columns
```

### Streaming Indicators

```python
from numta.streaming import StreamingSMA, StreamingRSI, StreamingMACD

# Create streaming indicators
sma = StreamingSMA(timeperiod=20)
rsi = StreamingRSI(timeperiod=14)

# Process streaming data
for price in price_stream:
    sma_value = sma.update(price)
    rsi_value = rsi.update(price)
    
    if sma.ready and rsi.ready:
        print(f"SMA: {sma_value:.2f}, RSI: {rsi_value:.2f}")
```

### Pattern Recognition

```python
from numta import CDLDOJI, CDLENGULFING, CDLHAMMER

# Returns +100 (bullish), -100 (bearish), or 0 (no pattern)
doji = CDLDOJI(open_, high, low, close)
engulfing = CDLENGULFING(open_, high, low, close)

# Via pandas accessor
df.ta.cdldoji(append=True)
df.ta.cdlengulfing(append=True)
```

## Getting Started

Check out the [Installation Guide](getting-started/installation.md) and [Quick Start](getting-started/quickstart.md) to begin using numta.

## API Reference

Explore the complete [API Reference](api/index.md) for detailed documentation of all indicators.

## License

MIT License - see [LICENSE](https://github.com/deepentropy/numta/blob/main/LICENSE) for details.
