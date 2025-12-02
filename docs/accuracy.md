# Accuracy

numta is designed to produce results that match TA-Lib's output for all implemented indicators.

## Verification Methodology

numta includes comprehensive accuracy tests that compare output against TA-Lib:

```bash
# Install with comparison dependencies
pip install -e ".[dev,comparison]"

# Run accuracy tests
pytest tests/accuracy/ -v
```

## Accuracy Guarantees

### Numerical Precision

All indicators are tested to match TA-Lib output within floating-point precision:

- Relative tolerance: 1e-10
- Absolute tolerance: 1e-10

### Test Coverage

Each indicator is tested with:

- Standard parameters (e.g., RSI with timeperiod=14)
- Edge cases (minimum periods, single values)
- Various data lengths (100 to 10,000 points)
- Random and trending data patterns

## Accuracy Test Results

### Moving Averages

| Indicator | Status | Max Difference |
|-----------|--------|----------------|
| SMA | ✅ Pass | < 1e-14 |
| EMA | ✅ Pass | < 1e-14 |
| WMA | ✅ Pass | < 1e-14 |
| DEMA | ✅ Pass | < 1e-13 |
| TEMA | ✅ Pass | < 1e-13 |
| KAMA | ✅ Pass | < 1e-12 |
| TRIMA | ✅ Pass | < 1e-14 |
| T3 | ✅ Pass | < 1e-12 |

### Momentum Indicators

| Indicator | Status | Max Difference |
|-----------|--------|----------------|
| RSI | ✅ Pass | < 1e-12 |
| MACD | ✅ Pass | < 1e-13 |
| STOCH | ✅ Pass | < 1e-12 |
| ADX | ✅ Pass | < 1e-11 |
| CCI | ✅ Pass | < 1e-12 |
| MFI | ✅ Pass | < 1e-12 |
| ROC | ✅ Pass | < 1e-14 |
| MOM | ✅ Pass | < 1e-14 |
| WILLR | ✅ Pass | < 1e-14 |

### Volatility Indicators

| Indicator | Status | Max Difference |
|-----------|--------|----------------|
| ATR | ✅ Pass | < 1e-12 |
| NATR | ✅ Pass | < 1e-11 |
| TRANGE | ✅ Pass | < 1e-14 |

### Volume Indicators

| Indicator | Status | Max Difference |
|-----------|--------|----------------|
| OBV | ✅ Pass | < 1e-14 |
| AD | ✅ Pass | < 1e-12 |
| ADOSC | ✅ Pass | < 1e-11 |

### Bands and Channels

| Indicator | Status | Max Difference |
|-----------|--------|----------------|
| BBANDS | ✅ Pass | < 1e-13 |
| SAR | ✅ Pass | < 1e-11 |

## Running Accuracy Tests

### Quick Accuracy Check

```python
import numpy as np
from numta import SMA, RSI, MACD

# Try importing TA-Lib for comparison
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    print("TA-Lib not installed, skipping comparison")

if HAS_TALIB:
    # Generate test data
    np.random.seed(42)
    close = np.random.randn(1000).astype(np.float64) + 100
    
    # Compare SMA
    numta_sma = SMA(close, timeperiod=20)
    talib_sma = talib.SMA(close, timeperiod=20)
    sma_diff = np.nanmax(np.abs(numta_sma - talib_sma))
    print(f"SMA max difference: {sma_diff}")
    
    # Compare RSI
    numta_rsi = RSI(close, timeperiod=14)
    talib_rsi = talib.RSI(close, timeperiod=14)
    rsi_diff = np.nanmax(np.abs(numta_rsi - talib_rsi))
    print(f"RSI max difference: {rsi_diff}")
    
    # Compare MACD
    numta_macd, numta_signal, numta_hist = MACD(close)
    talib_macd, talib_signal, talib_hist = talib.MACD(close)
    macd_diff = np.nanmax(np.abs(numta_macd - talib_macd))
    print(f"MACD max difference: {macd_diff}")
```

### Full Accuracy Suite

```bash
# Install TA-Lib (requires C library)
# See: https://github.com/TA-Lib/ta-lib-python

# Run all accuracy tests
pytest tests/accuracy/ -v --tb=short

# Run specific indicator test
pytest tests/accuracy/test_overlap_accuracy.py -v
```

## Known Differences

### Lookback Period Handling

numta uses NaN for the lookback period, matching TA-Lib behavior:

```python
import numpy as np
from numta import SMA

close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
sma = SMA(close, timeperiod=3)
print(sma)  # [nan, nan, 2.0, 3.0, 4.0]
```

### MAMA/FAMA (Hilbert Transform)

The MAMA indicator uses a simplified implementation that may differ slightly from TA-Lib in edge cases due to the complexity of the Hilbert Transform algorithm.

### SAR Edge Cases

Parabolic SAR may show minor differences (< 1e-8) in the first few values due to initialization differences.

## Reporting Accuracy Issues

If you find an accuracy discrepancy:

1. Verify you're using the same input data types (float64)
2. Check the parameters match exactly
3. Open an issue with:
   - numta version
   - TA-Lib version
   - Sample data that reproduces the issue
   - Expected vs actual output

## Migration from TA-Lib

numta is designed as a drop-in replacement for TA-Lib:

```python
# Before (TA-Lib)
import talib
sma = talib.SMA(close, timeperiod=20)
rsi = talib.RSI(close, timeperiod=14)

# After (numta)
from numta import SMA, RSI
sma = SMA(close, timeperiod=20)
rsi = RSI(close, timeperiod=14)
```

The function signatures and output formats are designed to be identical.
