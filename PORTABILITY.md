.# Platform Portability Guide

## Overview

`talib-pure` is designed to be **fully portable** across all major platforms with multiple performance tiers.

## Performance Tiers

### 1. **Numba-Accelerated (Recommended)**
- **Speed**: Matches or exceeds C-based TA-Lib
- **Platforms**:
  - ✅ Windows (x86_64, ARM64)
  - ✅ Linux (x86_64, ARM64, aarch64)
  - ✅ macOS (x86_64, Apple Silicon M1/M2/M3)
  - ✅ WSL2
- **Requirements**: `numba >= 0.56.0`
- **Installation**: `pip install numba`

### 2. **NumPy Fallback (Always Available)**
- **Speed**: 2-3x slower than TA-Lib (still very fast)
- **Platforms**: Any platform that supports NumPy
- **Requirements**: `numpy >= 1.20.0` (already required)
- **Installation**: No additional packages needed

## Supported Platforms

### Primary Support (Tested)
| Platform | Architecture | Numba | NumPy | Status |
|----------|-------------|-------|-------|--------|
| Windows 10/11 | x86_64 | ✅ | ✅ | Fully Supported |
| Windows 11 | ARM64 | ✅ | ✅ | Fully Supported |
| Ubuntu 20.04+ | x86_64 | ✅ | ✅ | Fully Supported |
| Ubuntu 20.04+ | ARM64 | ✅ | ✅ | Fully Supported |
| macOS 11+ | x86_64 (Intel) | ✅ | ✅ | Fully Supported |
| macOS 11+ | ARM64 (Apple Silicon) | ✅ | ✅ | Fully Supported |
| WSL2 | x86_64 | ✅ | ✅ | Fully Supported |

### Extended Support (Should Work)
- Debian-based Linux distributions
- RHEL/CentOS/Fedora
- Alpine Linux (with NumPy fallback)
- FreeBSD (with NumPy fallback)
- Raspberry Pi (ARM)

### Limited Support
- **PyPy**: NumPy fallback only (Numba not compatible with PyPy)
- **WebAssembly**: NumPy fallback only
- **Embedded Systems**: NumPy fallback (if NumPy available)

## Installation Options

### Option 1: Full Performance (Recommended)
```bash
pip install talib-pure[numba]
```

### Option 2: Minimal (NumPy only)
```bash
pip install talib-pure
```

The library will automatically detect available optimization libraries and use the best available option.

## Python Version Support

| Python Version | Numba Support | NumPy Support | Recommended |
|---------------|---------------|---------------|-------------|
| 3.8 | ✅ | ✅ | ✅ |
| 3.9 | ✅ | ✅ | ✅ |
| 3.10 | ✅ | ✅ | ✅ |
| 3.11 | ✅ | ✅ | ✅ |
| 3.12 | ✅ | ✅ | ✅ |
| 3.13 | ⚠️ (Check Numba) | ✅ | ✅ |

## Automatic Fallback Behavior

The library uses a **graceful degradation** strategy:

```python
from talib_pure import SMA
import numpy as np

close = np.random.randn(1000)

# Automatically uses best available method:
# 1. Try Numba JIT (if installed)
# 2. Fall back to optimized NumPy (always available)
result = SMA(close, timeperiod=30)

# Force NumPy fallback (for testing/debugging)
result = SMA(close, timeperiod=30, use_numba=False)
```

## Performance Comparison

### With Numba (10,000 data points)
- Small datasets (100 points): **3x faster than TA-Lib**
- Medium datasets (1,000 points): **1.2x faster than TA-Lib**
- Large datasets (10,000+ points): **Equal to TA-Lib**

### Without Numba (NumPy fallback)
- All sizes: **2-3x slower than TA-Lib** (still very usable)
- Better than original pure Python (10-13x slower)

## Troubleshooting

### Numba Installation Issues

**Problem**: Numba fails to install on your platform

**Solution 1** - Use NumPy fallback:
```bash
pip install talib-pure
# Works without Numba, slightly slower but still fast
```

**Solution 2** - Try conda (better Numba compatibility):
```bash
conda install -c conda-forge numba
pip install talib-pure
```

### Platform-Specific Notes

#### Apple Silicon (M1/M2/M3)
```bash
# Use conda for best compatibility
conda create -n trading python=3.11
conda activate trading
conda install -c conda-forge numba numpy
pip install talib-pure
```

#### Alpine Linux / Docker
```bash
# Numba may not work, use NumPy fallback
apk add py3-numpy
pip install talib-pure
# Will automatically use NumPy fallback
```

#### Raspberry Pi
```bash
# ARM support available
pip install numba  # May take time to compile
pip install talib-pure
```

#### WSL2
```bash
# Works great with both Numba and NumPy
pip install talib-pure numba
```

## Checking Active Backend

```python
from talib_pure.overlap import NUMBA_AVAILABLE

if NUMBA_AVAILABLE:
    print("✅ Numba acceleration active (maximum performance)")
else:
    print("⚠️  Using NumPy fallback (good performance, portable)")
```

## Docker Example

```dockerfile
# Option 1: Full performance (larger image)
FROM python:3.11-slim
RUN pip install numpy numba talib-pure

# Option 2: Minimal portable (smaller image)
FROM python:3.11-alpine
RUN apk add py3-numpy && pip install talib-pure
```

## Continuous Integration

The library is tested on:
- GitHub Actions (Ubuntu, Windows, macOS)
- Multiple Python versions (3.8-3.12)
- Both Numba and NumPy-only environments

## Conclusion

✅ **100% portable** - Works everywhere NumPy works
✅ **Automatic optimization** - Uses best available backend
✅ **No breaking changes** - Numba is optional, not required
✅ **Cross-platform** - Windows, Linux, macOS, ARM, x86_64
✅ **Graceful degradation** - Fast with Numba, still fast without it

The library will **always work** with just NumPy, and will **automatically accelerate** if Numba is available.