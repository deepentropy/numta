# numta

A pure Python implementation of TA-Lib (Technical Analysis Library) with a focus on performance and compatibility.

## Overview

`numta` (NumPy Technical Analysis) provides the same functionality as the popular TA-Lib library but implemented entirely in Python using NumPy and Numba for performance. This eliminates the need for complex C library dependencies while maintaining high performance through optimized NumPy operations and JIT compilation.

**Disclaimer**: This is an independent implementation inspired by TA-Lib. It is not affiliated with, endorsed by, or connected to the original TA-Lib project. The technical analysis algorithms implemented here are based on publicly available mathematical formulas and are compatible with TA-Lib's function signatures for ease of migration.

## Features

- **Pure Python implementation** (no C dependencies)
- **TA-Lib compatible** function signatures
- **Multiple performance backends:**
  - Default NumPy implementation
  - Optimized cumsum algorithm (3x faster, no dependencies)
  - Numba JIT compilation (5-10x faster)
- **Automatic backend selection** for optimal performance
- **Easy installation** via pip
- **Comprehensive test suite** comparing outputs with original TA-Lib
- **Built-in performance benchmarking** tools

## Installation

### Basic Installation

```bash
# From PyPI (when published)
pip install numta

# From source
git clone https://github.com/deepentropy/numta.git
cd numta
pip install -e .
```

### Performance Optimizations (Recommended)

```bash
# Install with Numba for 5-10x speedup
pip install "numta[numba]"

# Install development dependencies
pip install "numta[dev]"
```

## Quick Start

```python
import numpy as np
from numta import SMA

# Create sample price data
close_prices = np.random.uniform(100, 200, 100)

# Calculate Simple Moving Average with default period (30)
sma = SMA(close_prices)

# Calculate SMA with custom period
sma_20 = SMA(close_prices, timeperiod=20)

print(f"SMA values: {sma[-5:]}")  # Last 5 values
```

## Pandas Integration 🐼

`numta` provides a seamless pandas DataFrame integration through the `.ta` accessor, allowing you to calculate and append technical indicators directly to your DataFrames.

### Installation with Pandas Support

```bash
pip install "numta[pandas]"
```

### Basic Usage with DataFrames

```python
import pandas as pd
import numta  # Auto-registers the .ta accessor

# Load your OHLCV data
df = pd.DataFrame({
    'open': [100, 101, 102, 103, 104],
    'high': [105, 106, 107, 108, 109],
    'low': [99, 100, 101, 102, 103],
    'close': [104, 105, 106, 107, 108],
    'volume': [1000, 1100, 1200, 1300, 1400]
})

# Calculate and return Series (default behavior)
sma_series = df.ta.sma(timeperiod=20)

# Calculate and append to DataFrame
df.ta.sma(timeperiod=20, append=True)   # Adds column 'SMA_20'
df.ta.ema(timeperiod=12, append=True)   # Adds column 'EMA_12'
df.ta.rsi(timeperiod=14, append=True)   # Adds column 'RSI_14'
```

### Multi-Output Indicators

```python
# MACD returns 3 columns
df.ta.macd(append=True)
# Adds: 'MACD_12_26_9', 'MACDSignal_12_26_9', 'MACDHist_12_26_9'

# Bollinger Bands returns 3 columns
df.ta.bbands(timeperiod=20, append=True)
# Adds: 'BBU_20_2.0', 'BBM_20', 'BBL_20_2.0'

# Stochastic returns 2 columns
df.ta.stoch(append=True)
# Adds: 'STOCH_SLOWK_5_3_3', 'STOCH_SLOWD_5_3_3'
```

### Pattern Recognition

```python
# Candlestick patterns
df.ta.cdldoji(append=True)       # Adds 'CDLDOJI'
df.ta.cdlengulfing(append=True)  # Adds 'CDLENGULFING'
df.ta.cdlhammer(append=True)     # Adds 'CDLHAMMER'
```

### Chart Pattern Recognition 📊

Detect classic chart patterns with confidence scores:

```python
# Find all chart patterns
patterns = df.ta.find_patterns(pattern_type='all', order=5)

# Find specific pattern types
double_patterns = df.ta.find_patterns(pattern_type='double')
triangles = df.ta.find_patterns(pattern_type='triangle')
head_shoulders = df.ta.find_patterns(pattern_type='head_shoulders')

# Available pattern types:
# - 'head_shoulders': Head and Shoulders (regular and inverse)
# - 'double': Double Top/Bottom
# - 'triple': Triple Top/Bottom
# - 'triangle': Triangles (ascending, descending, symmetrical)
# - 'wedge': Wedges (rising, falling)
# - 'flag': Flags and Pennants
# - 'vcp': Volatility Contraction Pattern

# Access pattern details
for pattern in patterns:
    print(f"Type: {pattern.pattern_type}")
    print(f"Confidence: {pattern.confidence:.2f}")
    print(f"Start: {pattern.start_index}, End: {pattern.end_index}")
```

### Harmonic Pattern Detection 🔢

Detect harmonic patterns using Fibonacci ratios:

```python
# Find all harmonic patterns
harmonics = df.ta.find_harmonic_patterns()

# Find specific harmonic patterns
gartley_patterns = df.ta.find_harmonic_patterns(patterns=['gartley'])
bat_crab = df.ta.find_harmonic_patterns(patterns=['bat', 'crab'])

# Available patterns: 'gartley', 'butterfly', 'bat', 'crab'

# Access harmonic pattern details
for h in harmonics:
    print(f"Pattern: {h.pattern_type}")
    print(f"Direction: {h.direction}")  # 'bullish' or 'bearish'
    print(f"PRZ: {h.prz}")  # Potential Reversal Zone
    print(f"Confidence: {h.confidence:.2f}")
```

### Visualization (with lwcharts) 📈

Visualize charts with patterns in Jupyter notebooks:

```python
# Install visualization support
# pip install "numta[viz]"

# Plot candlestick chart with indicators
df.ta.plot(indicators={'SMA_20': df.ta.sma(20), 'EMA_10': df.ta.ema(10)})

# Plot with detected patterns
patterns = df.ta.find_patterns()
df.ta.plot(patterns=patterns)

# Alternative: use the viz module directly
from numta.viz import plot_chart, plot_ohlc, plot_line, plot_with_indicators

# Simple OHLC chart
chart = plot_ohlc(df, volume=True)

# Chart with indicators overlay
chart = plot_with_indicators(df, indicators={'SMA_20': sma_data}, volume=True)

# Pattern visualization
from numta.viz import plot_pattern, plot_harmonic
chart = plot_pattern(df, patterns, show_trendlines=True)
chart = plot_harmonic(df, harmonic_patterns, show_prz=True)
```

### Streaming/Real-Time Indicators ⚡

numta provides streaming versions of indicators for real-time data processing:

```python
from numta.streaming import (
    StreamingSMA, StreamingEMA, StreamingRSI, 
    StreamingMACD, StreamingATR
)

# Create streaming indicators
sma = StreamingSMA(timeperiod=20)
ema = StreamingEMA(timeperiod=12)
rsi = StreamingRSI(timeperiod=14)
macd = StreamingMACD(fastperiod=12, slowperiod=26, signalperiod=9)

# Process streaming price data
for price in price_stream:
    sma_value = sma.update(price)
    ema_value = ema.update(price)
    rsi_value = rsi.update(price)
    macd_result = macd.update(price)
    
    if sma.ready and rsi.ready:
        print(f"SMA: {sma_value:.2f}, RSI: {rsi_value:.2f}")

# For OHLCV data, use update_bar()
atr = StreamingATR(timeperiod=14)
for bar in ohlcv_stream:
    atr_value = atr.update_bar(
        open_=bar['open'],
        high=bar['high'],
        low=bar['low'],
        close=bar['close'],
        volume=bar['volume']
    )
```

**Available Streaming Indicators:**
- **Overlap**: StreamingSMA, StreamingEMA, StreamingBBANDS, StreamingDEMA, StreamingTEMA, StreamingWMA
- **Momentum**: StreamingRSI, StreamingMACD, StreamingSTOCH, StreamingMOM, StreamingROC
- **Volatility**: StreamingATR, StreamingTRANGE
- **Volume**: StreamingOBV, StreamingAD

### StreamingChart for Real-Time Visualization

```python
from numta.viz import StreamingChart
from numta.streaming import StreamingSMA

# Create streaming chart
chart = StreamingChart(max_points=500, title="Real-Time Chart")
sma = StreamingSMA(timeperiod=20)

# Add streaming data
for i, bar in enumerate(price_stream):
    chart.add_bar(i, bar['open'], bar['high'], bar['low'], bar['close'], bar['volume'])
    
    sma_value = sma.update(bar['close'])
    if sma_value is not None:
        chart.add_indicator('SMA_20', i, sma_value, color='blue')

# Display in Jupyter notebook
chart.show()
```

### Auto-Detection of OHLCV Columns

The accessor automatically detects standard column names (case-insensitive):
- Open: `open`, `Open`, `OPEN`, `o`
- High: `high`, `High`, `HIGH`, `h`
- Low: `low`, `Low`, `LOW`, `l`
- Close: `close`, `Close`, `CLOSE`, `c`, `adj_close`
- Volume: `volume`, `Volume`, `VOLUME`, `v`, `vol`

### Custom Column Specification

```python
# Use a custom column instead of auto-detected 'close'
df.ta.sma(column='adj_close', timeperiod=20, append=True)
```

### Column Naming Convention

All columns follow TA-Lib/pandas-ta conventions:
- Single output: `{INDICATOR}_{param1}_{param2}` (e.g., `SMA_20`, `RSI_14`)
- Multiple outputs: Named appropriately (e.g., `MACD_12_26_9`, `MACDSignal_12_26_9`)
- Patterns: `{PATTERN_NAME}` (e.g., `CDLDOJI`, `CDLENGULFING`)

### Migration from pandas-ta

If you're migrating from pandas-ta (abandoned since 2019), numta provides a compatible API:

```python
# pandas-ta style (works with numta)
df.ta.sma(timeperiod=20, append=True)
df.ta.rsi(timeperiod=14, append=True)
df.ta.macd(append=True)
df.ta.bbands(append=True)
```

## Performance Optimization 🚀

numta can match or exceed TA-Lib's performance using optional optimization backends:

```python
from numta import SMA_auto

# Automatic backend selection (recommended)
sma = SMA_auto(close_prices, timeperiod=30, backend='auto')

# Or choose specific backend
from numta import SMA_cumsum, SMA_numba

sma_fast = SMA_cumsum(close_prices, timeperiod=30)    # 3x faster, no deps
sma_faster = SMA_numba(close_prices, timeperiod=30)   # 5-10x faster
```

### Performance Comparison

| Implementation      | Speed vs Original | Requirements        |
|---------------------|-------------------|---------------------|
| **numpy (default)** | 1.0x (baseline)   | None                |
| **cumsum**          | **3.14x faster**  | None                |
| **numba**           | **5-10x faster**  | `pip install numba` |

**Benchmark Results (10,000 points):**
- Original (numpy): 0.154ms
- Cumsum: 0.049ms (3.14x faster)
- **Numba: 0.028ms (5.52x faster)** ⭐

## Implemented Indicators

This library implements a comprehensive set of technical analysis indicators across multiple categories:

### Overlap Studies
SMA, EMA, DEMA, TEMA, TRIMA, WMA, KAMA, MAMA, T3, BBANDS, MA, SAR, SAREXT

### Momentum Indicators
RSI, MACD, MACDEXT, MACDFIX, STOCH, STOCHF, STOCHRSI, ADX, ADXR, APO, AROON, AROONOSC, ATR, BOP, CCI, CMO, DX, MFI, MINUS_DI, MINUS_DM, MOM, PLUS_DI, PLUS_DM, PPO, ROC, ROCP, ROCR, ROCR100, TRIX, ULTOSC, WILLR

### Volume Indicators
AD, ADOSC, OBV

### Volatility Indicators
NATR, TRANGE

### Cycle Indicators
HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, HT_SINE, HT_TRENDLINE, HT_TRENDMODE

### Statistical Functions
BETA, CORREL, LINEARREG, LINEARREG_ANGLE, LINEARREG_INTERCEPT, LINEARREG_SLOPE, STDDEV, TSF, VAR

### Math Operators
MAX, MAXINDEX, MIN, MININDEX, MINMAX, MINMAXINDEX, SUM

### Price Transform
MEDPRICE, MIDPOINT, MIDPRICE, TYPPRICE, WCLPRICE

### Candlestick Pattern Recognition
60+ candlestick patterns including: Doji, Hammer, Engulfing, Morning Star, Evening Star, Three White Soldiers, and many more.

### Chart Pattern Recognition
- **Head and Shoulders**: Regular and Inverse patterns
- **Double Patterns**: Double Top, Double Bottom
- **Triple Patterns**: Triple Top, Triple Bottom
- **Triangles**: Ascending, Descending, Symmetrical
- **Wedges**: Rising, Falling
- **Flags**: Bull Flag, Bear Flag, Pennant
- **VCP**: Volatility Contraction Pattern

### Harmonic Pattern Recognition
- **Gartley**: 61.8% XA retracement, 78.6% completion
- **Butterfly**: 78.6% XA retracement, 127-161.8% extension
- **Bat**: 38.2-50% XA retracement, 88.6% completion
- **Crab**: 38.2-61.8% XA retracement, 161.8% extension

**See [FUNCTION_IMPLEMENTATIONS.md](FUNCTION_IMPLEMENTATIONS.md) for detailed implementation status.**

## Usage Examples

### Basic Usage

```python
import numpy as np
from numta import SMA

# Generate sample data
close = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)

# Calculate 3-period SMA
result = SMA(close, timeperiod=3)

# Output: [nan nan  2.  3.  4.  5.  6.  7.  8.  9.]
# First (timeperiod-1) values are NaN due to lookback period
```

### Working with Real Market Data

```python
import numpy as np
from numta import SMA

# Example with stock prices
close_prices = np.array([
    150.0, 151.5, 149.0, 153.0, 155.0,
    154.0, 156.5, 158.0, 157.0, 159.5
])

# Calculate 5-period SMA
sma_5 = SMA(close_prices, timeperiod=5)

for i, (price, sma) in enumerate(zip(close_prices, sma_5)):
    if not np.isnan(sma):
        print(f"Day {i+1}: Price={price:.2f}, SMA(5)={sma:.2f}")
```

## Performance Benchmarking

`numta` includes a powerful benchmarking class to compare performance with the original TA-Lib:

```python
import numpy as np
from numta import SMA as SMA_numta
from numta.benchmark import PerformanceMeasurement
import talib

# Create test data
data = np.random.uniform(100, 200, 10000)

# Setup benchmark
bench = PerformanceMeasurement()
bench.add_function("numta SMA", SMA_numta, data, timeperiod=30)
bench.add_function("TA-Lib SMA", talib.SMA, data, timeperiod=30)

# Run benchmark
results = bench.run(iterations=1000)
bench.print_results(results)
```

### Running the Benchmark Script

```bash
python examples/benchmark_sma.py
```

### Benchmark Features

The `PerformanceMeasurement` class provides:

- Measure execution time with configurable iterations and warmup
- Compare multiple functions side-by-side
- Calculate speedup ratios
- Test across different data sizes
- Statistical analysis (mean, median, std dev, min, max)

## Testing

The library includes comprehensive tests that compare outputs with the original TA-Lib:

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Optional: Install TA-Lib for comparison tests (requires TA-Lib C library)
pip install -e ".[comparison]"

# Run specific test file
pytest tests/test_sma.py

# Run with verbose output
pytest -v

# Run benchmark tests
pytest tests/test_benchmark.py
```

### Test Coverage

- **Comparison Tests**: Verify that outputs match TA-Lib exactly
- **Edge Cases**: Handle empty arrays, insufficient data, etc.
- **Input Validation**: Test error handling for invalid inputs
- **Data Types**: Support for both numpy arrays and Python lists

## API Compatibility

`numta` maintains full API compatibility with TA-Lib:

| Feature             | TA-Lib | numta |
|---------------------|--------|-------|
| Function signatures | ✓      | ✓     |
| Return values       | ✓      | ✓     |
| NaN handling        | ✓      | ✓     |
| NumPy array support | ✓      | ✓     |
| List support        | ✓      | ✓     |
| Default parameters  | ✓      | ✓     |

## Development

### Project Structure

```
numta/
├── src/
│   └── numta/
│       ├── __init__.py           # Main package exports
│       ├── backend.py            # Backend selection logic
│       ├── benchmark.py          # Performance measurement tools
│       ├── optimized.py          # Optimized implementations
│       ├── api/                  # Public API layer
│       │   ├── overlap.py        # Overlap studies (SMA, EMA, etc.)
│       │   ├── momentum_indicators.py
│       │   ├── volume_indicators.py
│       │   └── ...
│       └── cpu/                  # CPU/Numba implementations
│           ├── overlap.py        # Numba-optimized overlap studies
│           ├── math_operators.py
│           └── ...
├── tests/                        # Comprehensive test suite
│   ├── test_sma.py
│   ├── test_ema.py
│   └── test_benchmark.py
├── examples/                     # Usage examples
│   ├── benchmark_sma.py
│   └── benchmark_optimized.py
├── development/                  # Development tools
│   ├── accuracy_*.py             # Accuracy comparison scripts
│   ├── benchmark_*.py            # Benchmark scripts
│   └── ACCURACY.md               # Accuracy test results
├── PERFORMANCE.md                # Performance analysis
├── FUNCTION_IMPLEMENTATIONS.md   # Implementation details
├── pyproject.toml
├── LICENSE
└── README.md
```

### Adding New Indicators

To add a new indicator:

1. Implement the function in the appropriate API module (e.g., `api/overlap.py` for overlap studies)
2. Optionally add optimized Numba implementation in the corresponding `cpu/` module
3. Ensure the signature matches TA-Lib exactly
4. Add comparison tests in `tests/`
5. Export the function in `__init__.py`
6. Add documentation and examples

Example:

```python
def EMA(close: Union[np.ndarray, list], timeperiod: int = 30) -> np.ndarray:
    """
    Exponential Moving Average

    Parameters
    ----------
    close : array-like
        Close prices
    timeperiod : int, optional
        Period for EMA (default: 30)

    Returns
    -------
    np.ndarray
        EMA values
    """
    # Implementation here
    pass
```

### Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Implement your changes with tests
4. Ensure all tests pass
5. Submit a pull request

## Performance

`numta` uses NumPy's optimized functions and Numba JIT compilation to achieve performance competitive with the C-based TA-Lib. Benchmark results:

- **SMA**: Comparable performance to TA-Lib for large datasets
- **Lookback handling**: Efficient NaN placement
- **Memory usage**: Optimized array operations

Run `python examples/benchmark_sma.py` to see detailed benchmarks on your system.

## Requirements

- Python >= 3.8
- NumPy >= 1.20.0

### Optional Dependencies

- **Testing**: pytest >= 7.0.0, pytest-benchmark >= 4.0.0
- **Performance**: numba >= 0.56.0 (for JIT compilation, 5-10x speedup)
- **Comparison**: TA-Lib >= 0.4.0 (only for development/comparison scripts, requires C library)

## License

MIT License - see LICENSE file for details

## Acknowledgments

This project implements technical analysis algorithms that are publicly available mathematical formulas. We acknowledge and credit:

- **TA-Lib** - The original Technical Analysis Library (Copyright (c) 1999-2024, Mario Fortier)
  - Website: https://ta-lib.org/
  - Python wrapper: https://github.com/TA-Lib/ta-lib-python
  - License: BSD 3-Clause

`numta` is an independent clean-room implementation and is not derived from TA-Lib's source code. All code in this repository is original work licensed under the MIT License (see LICENSE file).

## Roadmap

### Completed ✅
- [x] Core overlap studies (SMA, EMA, DEMA, TEMA, WMA, KAMA, etc.)
- [x] Momentum indicators (RSI, MACD, STOCH, ADX, etc.)
- [x] Volume indicators (OBV, AD, ADOSC)
- [x] Volatility indicators (NATR, TRANGE)
- [x] Pattern recognition (60+ candlestick patterns)
- [x] Cycle indicators (Hilbert Transform functions)
- [x] Statistical functions
- [x] Math operators
- [x] Price transforms
- [x] Comprehensive test framework
- [x] Performance benchmarking tools
- [x] Multiple backend support (NumPy, Numba)
- [x] Pandas DataFrame extension accessor (`.ta`)
- [x] Streaming/real-time indicators
- [x] Jupyter notebook visualization with lwcharts

### In Progress 🚧
- [ ] Additional performance optimizations
- [ ] Extended documentation and examples
- [ ] More comprehensive benchmarks

### Future Plans 📋
- [ ] Integration with popular data providers
- [ ] Additional optimization backends

## Support

For issues, questions, or contributions, please visit:
https://github.com/deepentropy/numta/issues

## Citation

If you use this library in your research or project, please cite:

```
@software{numta,
  title={numta: NumPy-based Technical Analysis Library},
  author={numta contributors},
  url={https://github.com/deepentropy/numta},
  year={2025}
}
```
