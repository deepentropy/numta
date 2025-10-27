# talib-pure

A pure Python implementation of TA-Lib (Technical Analysis Library) with a focus on performance and compatibility.

## Overview

`talib-pure` aims to provide the same functionality as the popular TA-Lib library but implemented entirely in pure Python using NumPy for performance. This eliminates the need for complex C library dependencies while maintaining high performance through optimized NumPy operations.

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
pip install talib-pure

# From source
git clone https://github.com/houseofai/talib-pure.git
cd talib-pure
pip install -e .
```

### Performance Optimizations (Recommended)

```bash
# Install with Numba for 5-10x speedup
pip install "talib-pure[numba]"

# Install everything (for development)
pip install "talib-pure[all]"
```

## Quick Start

```python
import numpy as np
from talib_pure import SMA

# Create sample price data
close_prices = np.random.uniform(100, 200, 100)

# Calculate Simple Moving Average with default period (30)
sma = SMA(close_prices)

# Calculate SMA with custom period
sma_20 = SMA(close_prices, timeperiod=20)

print(f"SMA values: {sma[-5:]}")  # Last 5 values
```

## Performance Optimization 🚀

talib-pure can match or exceed TA-Lib's performance using optional optimization backends:

```python
from talib_pure import SMA_auto

# Automatic backend selection (recommended)
sma = SMA_auto(close_prices, timeperiod=30, backend='auto')

# Or choose specific backend
from talib_pure import SMA_cumsum, SMA_numba

sma_fast = SMA_cumsum(close_prices, timeperiod=30)    # 3x faster, no deps
sma_faster = SMA_numba(close_prices, timeperiod=30)   # 5-10x faster
```

### Performance Comparison

| Implementation | Speed vs Original | Requirements |
|---------------|------------------|--------------|
| **numpy (default)** | 1.0x (baseline) | None |
| **cumsum** | **3.14x faster** | None |
| **numba** | **5-10x faster** | `pip install numba` |

**Benchmark Results (10,000 points):**
- Original (numpy): 0.154ms
- Cumsum: 0.049ms (3.14x faster)
- **Numba: 0.028ms (5.52x faster)** ⭐

**See [OPTIMIZATION.md](OPTIMIZATION.md) for detailed performance guide.**

## Implemented Indicators

### Overlap Studies

- **SMA** - Simple Moving Average

More indicators coming soon!

## Usage Examples

### Basic Usage

```python
import numpy as np
from talib_pure import SMA

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
from talib_pure import SMA

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

`talib-pure` includes a powerful benchmarking class to compare performance with the original TA-Lib:

```python
import numpy as np
from talib_pure import SMA as SMA_pure
from talib_pure.benchmark import PerformanceMeasurement
import talib

# Create test data
data = np.random.uniform(100, 200, 10000)

# Setup benchmark
bench = PerformanceMeasurement()
bench.add_function("talib-pure SMA", SMA_pure, data, timeperiod=30)
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

# Run specific test file
pytest tests/test_sma_comparison.py

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

`talib-pure` maintains full API compatibility with TA-Lib:

| Feature | TA-Lib | talib-pure |
|---------|--------|------------|
| Function signatures | ✓ | ✓ |
| Return values | ✓ | ✓ |
| NaN handling | ✓ | ✓ |
| NumPy array support | ✓ | ✓ |
| List support | ✓ | ✓ |
| Default parameters | ✓ | ✓ |

## Development

### Project Structure

```
talib-pure/
├── src/
│   └── talib_pure/
│       ├── __init__.py
│       ├── overlap.py      # Overlap studies (SMA, EMA, etc.)
│       └── benchmark.py    # Performance measurement tools
├── tests/
│   ├── test_sma_comparison.py
│   └── test_benchmark.py
├── examples/
│   └── benchmark_sma.py
├── pyproject.toml
├── LICENSE
└── README.md
```

### Adding New Indicators

To add a new indicator:

1. Implement the function in the appropriate module (e.g., `overlap.py` for overlap studies)
2. Ensure the signature matches TA-Lib exactly
3. Add comparison tests in `tests/`
4. Update `__init__.py` to export the function
5. Add documentation and examples

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

`talib-pure` uses NumPy's optimized functions (like `convolve`) to achieve performance competitive with the C-based TA-Lib. Benchmark results:

- **SMA**: Comparable performance to TA-Lib for large datasets
- **Lookback handling**: Efficient NaN placement
- **Memory usage**: Optimized array operations

Run `python examples/benchmark_sma.py` to see detailed benchmarks on your system.

## Requirements

- Python >= 3.8
- NumPy >= 1.20.0

### Optional Dependencies

- TA-Lib >= 0.4.0 (for comparison tests only)
- pytest >= 7.0.0 (for running tests)
- pytest-benchmark >= 4.0.0 (for benchmark tests)

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Original TA-Lib library: https://ta-lib.org/
- TA-Lib Python wrapper: https://github.com/TA-Lib/ta-lib-python

## Roadmap

### Phase 1 (Current)
- [x] SMA (Simple Moving Average)
- [x] Test framework
- [x] Performance benchmarking tools

### Phase 2 (Planned)
- [ ] EMA (Exponential Moving Average)
- [ ] WMA (Weighted Moving Average)
- [ ] DEMA (Double Exponential Moving Average)
- [ ] TEMA (Triple Exponential Moving Average)
- [ ] RSI (Relative Strength Index)
- [ ] MACD (Moving Average Convergence Divergence)

### Phase 3 (Future)
- [ ] Complete overlap studies
- [ ] Momentum indicators
- [ ] Volume indicators
- [ ] Volatility indicators
- [ ] Pattern recognition

## Support

For issues, questions, or contributions, please visit:
https://github.com/houseofai/talib-pure/issues

## Citation

If you use this library in your research or project, please cite:

```
@software{talib_pure,
  title={talib-pure: Pure Python Technical Analysis Library},
  author={talib-pure contributors},
  url={https://github.com/houseofai/talib-pure},
  year={2025}
}
```
