# numta

Pure Python technical analysis library. A modern, high-performance alternative to TA-Lib with zero C dependencies.

## Highlights

- **Pure Python**: No C compiler required, works everywhere Python runs
- **Fast**: 5-10x speedup with optional Numba JIT compilation
- **Complete**: 130+ indicators, 60+ candlestick patterns, chart pattern detection
- **Modern**: Pandas integration, streaming support, Jupyter visualization

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

# From source
git clone https://github.com/deepentropy/numta.git
cd numta
pip install -e .
```

## Quick Start

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

## Features

### Technical Indicators

numta provides 130+ technical indicators across multiple categories:

- **Overlap Studies**: SMA, EMA, DEMA, TEMA, WMA, BBANDS, KAMA, MAMA, T3, SAR
- **Momentum**: RSI, MACD, STOCH, ADX, CCI, MFI, ROC, MOM, WILLR
- **Volume**: OBV, AD, ADOSC
- **Volatility**: ATR, NATR, TRANGE
- **Cycle**: Hilbert Transform functions
- **Statistical**: LINEARREG, STDDEV, VAR, CORREL, BETA

See [FUNCTION_IMPLEMENTATIONS.md](FUNCTION_IMPLEMENTATIONS.md) for the complete list.

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

### Pattern Recognition

#### Candlestick Patterns

```python
from numta import CDLDOJI, CDLENGULFING, CDLHAMMER

# Returns +100 (bullish), -100 (bearish), or 0 (no pattern)
doji = CDLDOJI(open_, high, low, close)
engulfing = CDLENGULFING(open_, high, low, close)

# Via pandas accessor
df.ta.cdldoji(append=True)
df.ta.cdlengulfing(append=True)
```

#### Chart Patterns

```python
from numta import (
    detect_head_shoulders, detect_double_top,
    detect_triangle, detect_wedge, detect_flag
)

# Detect patterns with confidence scores
patterns = detect_head_shoulders(highs, lows, order=5)

# Via pandas accessor
patterns = df.ta.find_patterns(pattern_type='all')
harmonics = df.ta.find_harmonic_patterns()
```

### Streaming/Real-Time

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

### Visualization

```python
# Install visualization support
# pip install "numta[viz]"

from numta.viz import plot_ohlc, plot_with_indicators

# Basic candlestick chart
chart = plot_ohlc(df, volume=True)

# Chart with indicators
chart = plot_with_indicators(df, indicators={'SMA_20': sma_data})

# Plot detected patterns
chart = df.ta.plot(patterns=patterns)
```

## Documentation

- [Getting Started](notebooks/01_getting_started.ipynb)
- [Technical Indicators Guide](notebooks/02_technical_indicators.ipynb)
- [Candlestick Patterns](notebooks/03_candlestick_patterns.ipynb)
- [Chart Patterns](notebooks/04_chart_patterns.ipynb)
- [Harmonic Patterns](notebooks/05_harmonic_patterns.ipynb)
- [Streaming Indicators](notebooks/06_streaming_indicators.ipynb)
- [Visualization](notebooks/07_visualization.ipynb)
- [Performance Optimization](notebooks/08_performance_optimization.ipynb)

## Performance

numta uses optimized algorithms and optional Numba JIT compilation:

| Implementation | Speed vs Default | Requirements |
|----------------|------------------|--------------|
| numpy (default) | 1.0x (baseline) | None |
| cumsum | ~3x faster | None |
| numba | 5-10x faster | `pip install numba` |

```python
from numta import SMA_auto, SMA_cumsum

# Automatic backend selection
sma = SMA_auto(close_prices, timeperiod=30, backend='auto')

# Or choose specific backend
sma_fast = SMA_cumsum(close_prices, timeperiod=30)
```

## API Reference

See [FUNCTION_IMPLEMENTATIONS.md](FUNCTION_IMPLEMENTATIONS.md) for detailed implementation status of all indicators.

## Requirements

- Python >= 3.8
- NumPy >= 1.20.0

### Optional Dependencies

| Feature | Package | Installation |
|---------|---------|--------------|
| Performance | numba >= 0.56.0 | `pip install "numta[numba]"` |
| Pandas | pandas >= 1.3.0 | `pip install "numta[pandas]"` |
| Visualization | lwcharts >= 0.1.0 | `pip install "numta[viz]"` |
| All features | - | `pip install "numta[full]"` |

## Project Structure

```
numta/
├── src/numta/
│   ├── __init__.py
│   ├── api/                 # Indicator implementations
│   ├── cpu/                 # Numba-optimized versions
│   ├── patterns/            # Chart pattern detection
│   ├── streaming/           # Real-time indicators
│   ├── viz/                 # Visualization (lwcharts)
│   ├── pandas_ext.py        # DataFrame accessor
│   ├── backend.py           # Backend selection
│   ├── benchmark.py         # Performance tools
│   └── optimized.py         # Optimized implementations
├── notebooks/               # Example notebooks
├── tests/                   # Test suite
├── pyproject.toml
└── README.md
```

## Testing

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with verbose output
pytest -v
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Implement your changes with tests
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

This project implements technical analysis algorithms based on publicly available mathematical formulas. We acknowledge:

- **TA-Lib** - The original Technical Analysis Library
  - Website: https://ta-lib.org/
  - Python wrapper: https://github.com/TA-Lib/ta-lib-python
  - License: BSD 3-Clause

numta is an independent implementation and is not derived from TA-Lib's source code. All code is original work licensed under the MIT License.

## Support

For issues, questions, or contributions:
https://github.com/deepentropy/numta/issues

## Citation

```bibtex
@software{numta,
  title={numta: NumPy-based Technical Analysis Library},
  author={numta contributors},
  url={https://github.com/deepentropy/numta},
  year={2025}
}
```
