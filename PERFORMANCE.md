# Performance Comparison

This document presents performance comparisons between **talib-pure** (Numba/CPU implementation) and the **original TA-Lib** library.

## Contents

- [Cycle Indicators](#cycle-indicators)
- [Math Operators](#math-operators)

---

# Cycle Indicators

## Test Environment

- **Python Version**: 3.11
- **NumPy Version**: 2.3.4
- **Numba Version**: 0.62.1
- **TA-Lib Version**: 0.6.8
- **Platform**: Linux
- **Test Method**: Average execution time over multiple iterations (100 iterations for small/medium datasets, 10 for large datasets)

## Summary

The following table shows the speedup factor (talib-pure vs TA-Lib) across different dataset sizes:

| Function | 1K bars | 10K bars | 100K bars | Average |
|----------|---------|----------|-----------|---------|
| **HT_DCPERIOD** | 0.79x | 0.32x | 0.38x | **0.50x** |
| **HT_DCPHASE** | 7.60x | 2.01x | 2.85x | **4.15x** |
| **HT_PHASOR** | 3.21x | 0.52x | 0.87x | **1.53x** |
| **HT_SINE** | 3.20x | 1.55x | 1.85x | **2.20x** |
| **HT_TRENDLINE** | 26.08x | 46.59x | 36.64x | **36.44x** |
| **HT_TRENDMODE** | 15.23x | 15.25x | 7.61x | **12.70x** |

**Note**: Values greater than 1.0x indicate talib-pure is faster; values less than 1.0x indicate TA-Lib is faster.

## Key Findings

### High Performance Functions (10x+ speedup)
- **HT_TRENDLINE**: 26-46x faster - Exceptional performance due to simple weighted moving average calculation optimized by Numba
- **HT_TRENDMODE**: 7-15x faster - Consistently excellent performance across all dataset sizes

### Good Performance Functions (2-10x speedup)
- **HT_DCPHASE**: 2-7.6x faster - Strong performance, especially on smaller datasets
- **HT_SINE**: 1.5-3.2x faster - Consistent moderate speedup across all sizes

### Slower Functions (< 1.0x)
- **HT_DCPERIOD**: 0.32-0.79x - The original TA-Lib implementation is faster for this function
- **HT_PHASOR**: 0.52-3.21x - Mixed performance depending on dataset size

## Detailed Results

### Dataset: 1,000 bars

| Function | TA-Lib (ms) | talib-pure (ms) | Speedup |
|----------|-------------|-----------------|---------|
| HT_DCPERIOD | 0.0578 | 0.0734 | **0.79x** |
| HT_DCPHASE | 0.2571 | 0.0338 | **7.60x** |
| HT_PHASOR | 0.0575 | 0.0179 | **3.21x** |
| HT_SINE | 0.3059 | 0.0956 | **3.20x** |
| HT_TRENDLINE | 0.0580 | 0.0022 | **26.08x** |
| HT_TRENDMODE | 0.3118 | 0.0205 | **15.23x** |

### Dataset: 10,000 bars

| Function | TA-Lib (ms) | talib-pure (ms) | Speedup |
|----------|-------------|-----------------|---------|
| HT_DCPERIOD | 0.5839 | 1.8229 | **0.32x** |
| HT_DCPHASE | 2.7919 | 1.3868 | **2.01x** |
| HT_PHASOR | 0.5810 | 1.1187 | **0.52x** |
| HT_SINE | 3.2005 | 2.0708 | **1.55x** |
| HT_TRENDLINE | 0.6517 | 0.0140 | **46.59x** |
| HT_TRENDMODE | 3.2921 | 0.2159 | **15.25x** |

### Dataset: 100,000 bars

| Function | TA-Lib (ms) | talib-pure (ms) | Speedup |
|----------|-------------|-----------------|---------|
| HT_DCPERIOD | 5.9591 | 15.8113 | **0.38x** |
| HT_DCPHASE | 28.9483 | 10.1458 | **2.85x** |
| HT_PHASOR | 6.7063 | 7.6714 | **0.87x** |
| HT_SINE | 35.2159 | 19.0322 | **1.85x** |
| HT_TRENDLINE | 6.8711 | 0.1875 | **36.64x** |
| HT_TRENDMODE | 33.3462 | 4.3796 | **7.61x** |

## Analysis

### Why Some Functions Are Faster

1. **HT_TRENDLINE**: This is the simplest cycle indicator, using only a weighted moving average. Numba's JIT compilation excels at optimizing such straightforward numerical operations.

2. **HT_TRENDMODE**: While more complex, the Numba-optimized implementation efficiently handles the trend/cycle mode detection logic with minimal overhead.

3. **HT_DCPHASE & HT_SINE**: These functions benefit from Numba's efficient handling of trigonometric operations and array computations.

### Why Some Functions Are Slower

1. **HT_DCPERIOD**: This function involves complex dominant cycle period calculations with multiple conditional branches. The original TA-Lib's C implementation may handle these specific branching patterns more efficiently.

2. **HT_PHASOR**: The performance varies by dataset size, suggesting that for medium-sized datasets (10K bars), the overhead of Numba's array operations may outweigh the benefits of JIT compilation.

### Performance Characteristics

- **Small datasets (1K bars)**: talib-pure generally shows excellent performance due to efficient Numba JIT compilation and minimal overhead
- **Medium datasets (10K bars)**: Mixed results, with some functions showing reduced speedup as JIT compilation overhead becomes more apparent
- **Large datasets (100K bars)**: Performance stabilizes, with most functions showing consistent speedup patterns

## Implementation Details

All cycle indicators in talib-pure are implemented using:
- **Numba JIT compilation** with `@jit(nopython=True, cache=True)` decorator
- **Array-based computations** for efficient vectorization
- **Identical algorithms** to TA-Lib for compatibility
- **32-bar lookback period** (first 32 values are NaN) following TA-Lib standards

## Recommendations

### When to Use talib-pure
- **HT_TRENDLINE** and **HT_TRENDMODE**: Always prefer talib-pure (10x+ faster)
- **HT_DCPHASE** and **HT_SINE**: Use talib-pure for 1.5-7.6x speedup
- When you need a pure Python implementation without C dependencies
- For deployment environments where installing TA-Lib's C library is challenging

### When to Use Original TA-Lib
- **HT_DCPERIOD**: The original TA-Lib is 2-3x faster
- When you need battle-tested implementations with years of production use
- If you're already using TA-Lib for other indicators

### Best of Both Worlds
Consider using a hybrid approach:
```python
# Use talib-pure for fast functions
from talib_pure import HT_TRENDLINE, HT_TRENDMODE, HT_DCPHASE, HT_SINE

# Use original TA-Lib for slower functions
import talib
ht_dcperiod = talib.HT_DCPERIOD
```

## Future Improvements

Potential optimizations for slower functions:
1. **HT_DCPERIOD**: Investigate alternative branching strategies or lookups for period detection
2. **HT_PHASOR**: Optimize array operations for medium-sized datasets
3. **GPU Support**: Implement CUDA/CuPy versions for 10-50x additional speedup on large datasets

## Reproducing These Results

To run the benchmarks yourself:

```bash
# Install dependencies
pip install -e ".[dev]"

# Run comparison benchmark
python benchmark_comparison.py

# Or run individual benchmarks
python tests/benchmark_ht_dcperiod.py
python tests/benchmark_ht_dcphase.py
python tests/benchmark_ht_phasor.py
python tests/benchmark_ht_sine.py
python tests/benchmark_ht_trendline.py
python tests/benchmark_ht_trendmode.py
```

## Conclusion

The talib-pure Numba/CPU implementation demonstrates **strong overall performance** for Cycle Indicators, with 4 out of 6 functions showing significant speedups (1.5x to 46x). While HT_DCPERIOD and HT_PHASOR are slower in certain scenarios, the majority of cycle indicators benefit substantially from the Numba-optimized implementation.

For typical use cases involving HT_TRENDLINE, HT_TRENDMODE, HT_DCPHASE, and HT_SINE, talib-pure offers compelling performance advantages while maintaining full API compatibility with the original TA-Lib library.

---

# Math Operators

## Test Environment

- **Python Version**: 3.11
- **NumPy Version**: 2.3.4
- **Numba Version**: 0.62.1
- **TA-Lib Version**: 0.6.8
- **Platform**: Linux
- **Test Method**: Average execution time over multiple iterations (100 iterations for small/medium datasets, 10 for large datasets)
- **Time Period**: 30 bars (default parameter)

## Summary

The following table shows the speedup factor (talib-pure vs TA-Lib) across different dataset sizes:

| Function | 1K bars | 10K bars | 100K bars | Average |
|----------|---------|----------|-----------|---------|
| **MAX** | 0.20x | 0.15x | 0.24x | **0.20x** |
| **MIN** | 0.17x | 0.15x | 0.25x | **0.19x** |
| **MINMAX** | 0.20x | 0.18x | 0.40x | **0.26x** |
| **MAXINDEX** | 0.13x | 0.13x | 0.14x | **0.13x** |
| **MININDEX** | 0.13x | 0.11x | 0.14x | **0.13x** |
| **MINMAXINDEX** | 0.14x | 0.11x | 0.14x | **0.13x** |
| **SUM** | 0.80x | 1.02x | 1.24x | **1.02x** |

**Note**: Values greater than 1.0x indicate talib-pure is faster; values less than 1.0x indicate TA-Lib is faster.

## Key Findings

### Faster Function (1.0x+ speedup)
- **SUM**: 0.8-1.24x faster - The only Math Operator where talib-pure matches or exceeds TA-Lib performance, thanks to optimized rolling window implementation

### Slower Functions (< 1.0x)
- **MAX/MIN/MINMAX**: 0.15-0.40x - Original TA-Lib is 2.5-7x faster
- **MAXINDEX/MININDEX/MINMAXINDEX**: 0.11-0.14x - Original TA-Lib is 7-9x faster

## Detailed Results

### Dataset: 1,000 bars

| Function | TA-Lib (ms) | talib-pure (ms) | Speedup |
|----------|-------------|-----------------|---------|
| MAX | 0.0041 | 0.0206 | **0.20x** |
| MIN | 0.0035 | 0.0204 | **0.17x** |
| MINMAX | 0.0047 | 0.0237 | **0.20x** |
| MAXINDEX | 0.0039 | 0.0303 | **0.13x** |
| MININDEX | 0.0040 | 0.0305 | **0.13x** |
| MINMAXINDEX | 0.0062 | 0.0455 | **0.14x** |
| SUM | 0.0028 | 0.0035 | **0.80x** |

### Dataset: 10,000 bars

| Function | TA-Lib (ms) | talib-pure (ms) | Speedup |
|----------|-------------|-----------------|---------|
| MAX | 0.0329 | 0.2191 | **0.15x** |
| MIN | 0.0334 | 0.2194 | **0.15x** |
| MINMAX | 0.0435 | 0.2353 | **0.18x** |
| MAXINDEX | 0.0383 | 0.3058 | **0.13x** |
| MININDEX | 0.0379 | 0.3403 | **0.11x** |
| MINMAXINDEX | 0.0521 | 0.4545 | **0.11x** |
| SUM | 0.0255 | 0.0250 | **1.02x** |

### Dataset: 100,000 bars

| Function | TA-Lib (ms) | talib-pure (ms) | Speedup |
|----------|-------------|-----------------|---------|
| MAX | 0.5125 | 2.1407 | **0.24x** |
| MIN | 0.5107 | 2.0619 | **0.25x** |
| MINMAX | 1.1511 | 2.8434 | **0.40x** |
| MAXINDEX | 0.4467 | 3.1797 | **0.14x** |
| MININDEX | 0.4389 | 3.1948 | **0.14x** |
| MINMAXINDEX | 0.7230 | 5.1243 | **0.14x** |
| SUM | 0.3793 | 0.3050 | **1.24x** |

## Analysis

### Why SUM Is Competitive

**SUM** is the only Math Operator where talib-pure approaches or exceeds TA-Lib performance:

1. **Optimized Algorithm**: Uses incremental rolling window calculation (add new value, subtract old value) rather than recalculating the entire sum each time
2. **O(n) Complexity**: Linear time complexity vs O(n*timeperiod) for naive implementations
3. **Numba JIT Benefits**: Simple arithmetic operations that Numba optimizes extremely well
4. **Minimal Overhead**: No complex logic or branching, just addition and subtraction

### Why Most Math Operators Are Slower

The majority of Math Operators (MAX, MIN, and their variants) are significantly slower in talib-pure:

1. **Highly Optimized C Code**: TA-Lib's C implementation uses highly optimized algorithms with minimal overhead for simple operations like finding min/max values

2. **JIT Compilation Overhead**: For simple operations, Numba's JIT compilation overhead doesn't provide enough benefit to offset the performance of native C code

3. **Window Iteration**: Functions like MAX/MIN iterate through rolling windows of size `timeperiod`. While Numba optimizes this, the original TA-Lib's C implementation with direct memory access is faster

4. **INDEX Functions**: MAXINDEX/MININDEX need to track both the value and position of extremes, adding complexity that compounds with the window iteration overhead

### Performance Characteristics

- **Small datasets (1K bars)**: talib-pure shows relatively better performance (0.13-0.80x) due to smaller overhead
- **Medium datasets (10K bars)**: Performance gap widens (0.11-1.02x) as TA-Lib's optimizations become more apparent
- **Large datasets (100K bars)**: Performance stabilizes (0.14-1.24x), with SUM becoming noticeably faster in talib-pure

## Implementation Details

All Math Operators in talib-pure are implemented using:
- **Numba JIT compilation** with `@jit(nopython=True, cache=True)` decorator
- **Rolling window operations** for most functions (O(n*timeperiod) complexity)
- **Optimized SUM** using incremental calculation (O(n) complexity)
- **Identical algorithms** to TA-Lib for compatibility
- **Lookback period** of `(timeperiod - 1)` bars

## Recommendations

### When to Use talib-pure
- **SUM**: Slightly faster on large datasets (1.02x average), safe to use
- When you need a pure Python implementation without C dependencies
- For deployment environments where installing TA-Lib's C library is challenging
- When accuracy matters more than raw performance (see ACCURACY.md)

### When to Use Original TA-Lib
- **MAX, MIN, MINMAX**: TA-Lib is 2.5-7x faster - prefer original implementation
- **MAXINDEX, MININDEX, MINMAXINDEX**: TA-Lib is 7-9x faster - prefer original implementation (also see accuracy concerns in ACCURACY.md)
- For production systems where performance is critical
- When every millisecond counts in high-frequency trading applications

### Hybrid Approach

For Math Operators, the recommendation is clear:

```python
# Use talib-pure for SUM (comparable performance)
from talib_pure import SUM

# Use original TA-Lib for all other Math Operators (much faster)
import talib
MAX = talib.MAX
MIN = talib.MIN
MINMAX = talib.MINMAX
MAXINDEX = talib.MAXINDEX
MININDEX = talib.MININDEX
MINMAXINDEX = talib.MINMAXINDEX
```

## Future Improvements

Potential optimizations for Math Operators:

1. **MAX/MIN Functions**: Investigate using more efficient data structures (e.g., deque with monotonic stack) to reduce window search time
2. **INDEX Functions**: Optimize position tracking algorithm
3. **GPU Support**: Implement CUDA/CuPy versions which could provide massive speedups for large datasets
4. **Specialized Algorithms**: Consider using segment trees or other advanced data structures for range queries

## Reproducing These Results

To run the benchmarks yourself:

```bash
# Install dependencies
pip install -e ".[dev]"

# Run Math Operators comparison benchmark
python benchmark_math_operators.py

# Or run individual benchmarks
python tests/benchmark_max.py
python tests/benchmark_min.py
python tests/benchmark_minmax.py
python tests/benchmark_maxindex.py
python tests/benchmark_minindex.py
python tests/benchmark_minmaxindex.py
python tests/benchmark_sum.py
```

## Conclusion

The talib-pure Numba/CPU implementation shows **mixed performance** for Math Operators. While **SUM** is competitive and even slightly faster on large datasets (1.02x average), all other Math Operators are **significantly slower** than the original TA-Lib (5-9x slower).

**Key Takeaway**: For Math Operators, the original TA-Lib's highly optimized C implementation provides superior performance for most operations. The exception is **SUM**, where talib-pure's optimized rolling window algorithm makes it competitive.

**Recommendation**: Use the original TA-Lib for Math Operators unless you specifically need a pure Python implementation or are primarily using SUM. The performance penalty for MAX/MIN/INDEX functions is too significant to ignore in performance-critical applications.
