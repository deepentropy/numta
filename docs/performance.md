# Performance

numta is designed for high performance through optimized algorithms and optional Numba JIT compilation.

## Performance Comparison

numta uses optimized algorithms and optional Numba JIT compilation to achieve significant speedups:

| Implementation | Speed vs Default | Requirements |
|----------------|------------------|--------------|
| NumPy (default) | 1.0x (baseline) | None |
| Cumsum optimized | ~3x faster | None |
| Numba JIT | 5-10x faster | `pip install numba` |

## Backend Selection

### Automatic Backend

numta automatically selects the best available backend:

```python
from numta import SMA_auto

# Auto-selects fastest available backend
sma = SMA_auto(close, timeperiod=30, backend='auto')
```

### Manual Backend Selection

```python
from numta import SMA_cumsum, SMA

# Use cumsum-optimized implementation
sma_fast = SMA_cumsum(close, timeperiod=30)

# Standard implementation (uses Numba if available)
sma_standard = SMA(close, timeperiod=30)
```

## Numba Acceleration

Numba JIT compilation provides 5-10x performance improvement:

```bash
# Install Numba
pip install "numta[numba]"
```

### Verifying Numba is Active

```python
from numta.backend import get_backend

backend = get_backend()
print(f"Current backend: {backend}")
# Output: 'numba' if Numba is available
```

### Disabling Numba (for debugging)

```python
import os
os.environ['NUMBA_DISABLE_JIT'] = '1'

import numta  # Will use pure NumPy implementation
```

## Benchmarks

### Simple Moving Average (SMA)

Benchmark on 1 million data points:

| Implementation | Time (ms) | Speedup |
|----------------|-----------|---------|
| NumPy naive | 850 | 1.0x |
| NumPy cumsum | 45 | 19x |
| Numba | 12 | 71x |

### RSI

Benchmark on 1 million data points:

| Implementation | Time (ms) | Speedup |
|----------------|-----------|---------|
| NumPy | 320 | 1.0x |
| Numba | 35 | 9x |

### MACD

Benchmark on 1 million data points:

| Implementation | Time (ms) | Speedup |
|----------------|-----------|---------|
| NumPy | 180 | 1.0x |
| Numba | 28 | 6x |

## Running Benchmarks

Run the included benchmark suite:

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run benchmarks
pytest tests/benchmark/ -v
```

### Custom Benchmark

```python
import numpy as np
import time
from numta import SMA, EMA, RSI, MACD

# Generate test data
np.random.seed(42)
close = np.random.randn(1_000_000) + 100

# Benchmark SMA
start = time.perf_counter()
for _ in range(10):
    sma = SMA(close, timeperiod=20)
elapsed = (time.perf_counter() - start) / 10
print(f"SMA (1M points): {elapsed*1000:.2f} ms")

# Benchmark RSI
start = time.perf_counter()
for _ in range(10):
    rsi = RSI(close, timeperiod=14)
elapsed = (time.perf_counter() - start) / 10
print(f"RSI (1M points): {elapsed*1000:.2f} ms")

# Benchmark MACD
start = time.perf_counter()
for _ in range(10):
    macd, signal, hist = MACD(close)
elapsed = (time.perf_counter() - start) / 10
print(f"MACD (1M points): {elapsed*1000:.2f} ms")
```

## Memory Efficiency

### Streaming Mode for Large Datasets

For very large datasets or real-time applications, use streaming indicators:

```python
from numta.streaming import StreamingSMA

sma = StreamingSMA(timeperiod=20)

# Process data in chunks
for chunk in data_stream:
    for price in chunk:
        value = sma.update(price)
        # Process value...
```

### In-Place Operations

Many numta functions support pre-allocated output arrays:

```python
import numpy as np
from numta.cpu.overlap import _sma_numba

close = np.random.randn(1000)
output = np.empty(1000)

# Fill pre-allocated array (avoids allocation)
_sma_numba(close, 20, output)
```

## Best Practices

### 1. Use Contiguous Arrays

```python
import numpy as np

# Good: Contiguous array
close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

# May be slower: Non-contiguous array (e.g., column from 2D array)
close_col = some_2d_array[:, 0]

# Fix: Make contiguous
close_contiguous = np.ascontiguousarray(close_col)
```

### 2. Batch Processing

Calculate multiple indicators in a single pass over the data:

```python
from numta import SMA, EMA, RSI

# Calculate all at once (memory locality)
sma = SMA(close, 20)
ema = EMA(close, 12)
rsi = RSI(close, 14)
```

### 3. Avoid Repeated Calculations

```python
# Bad: Recalculates SMA each time
for i in range(len(df)):
    sma = SMA(df['close'][:i+1], 20)
    
# Good: Calculate once
sma = SMA(df['close'].values, 20)
```

### 4. Use Appropriate Data Types

```python
# Good: float64 for best precision
close = np.array(prices, dtype=np.float64)

# May cause issues: integer arrays
close_int = np.array(prices, dtype=np.int32)  # Will be converted
```

## Comparison with TA-Lib

numta provides comparable performance to TA-Lib while being pure Python:

| Library | Installation | Performance | Notes |
|---------|--------------|-------------|-------|
| TA-Lib (C) | Requires C compiler | Fastest | Hard to install |
| numta (Numba) | Pure Python | Very fast | Easy install |
| numta (NumPy) | Pure Python | Fast | No dependencies |
| pandas-ta | Pure Python | Moderate | More features |

For most applications, numta with Numba provides sufficient performance without the complexity of installing TA-Lib's C dependencies.
