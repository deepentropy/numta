# Installation

numta can be installed using pip with various optional dependencies.

## Basic Installation

```bash
pip install numta
```

This installs the core library with NumPy support only.

## Optional Dependencies

### Numba Acceleration

For 5-10x performance speedup using Numba JIT compilation:

```bash
pip install "numta[numba]"
```

### Pandas Integration

To use the `.ta` DataFrame accessor:

```bash
pip install "numta[pandas]"
```

### Full Installation

For all features including visualization:

```bash
pip install "numta[full]"
```

## From Source

To install the development version:

```bash
git clone https://github.com/deepentropy/numta.git
cd numta
pip install -e .
```

For development with test dependencies:

```bash
pip install -e ".[dev]"
```

## Requirements

- Python >= 3.8
- NumPy >= 1.20.0

### Optional Requirements

| Feature | Package | Installation |
|---------|---------|--------------|
| Performance | numba >= 0.56.0 | `pip install "numta[numba]"` |
| Pandas | pandas >= 1.3.0 | `pip install "numta[pandas]"` |
| Visualization | lwcharts >= 0.1.0 | `pip install "numta[viz]"` |
| All features | - | `pip install "numta[full]"` |

## Verifying Installation

```python
import numta

# Check version
print(numta.__version__)

# Quick test
import numpy as np
close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
sma = numta.SMA(close, timeperiod=3)
print(sma)  # [nan, nan, 2., 3., 4.]
```

## Troubleshooting

### Numba Installation Issues

If you encounter issues with Numba:

1. Ensure you have a compatible Python version (3.8-3.12)
2. Try installing LLVM libraries first:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install llvm

   # macOS
   brew install llvm
   ```

### Missing Indicators

If an indicator is not found, ensure you've imported it correctly:

```python
# Import specific indicators
from numta import SMA, EMA, RSI

# Or import all
import numta
sma = numta.SMA(close, timeperiod=20)
```
