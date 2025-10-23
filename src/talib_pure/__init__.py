"""
talib-pure: Pure Python TA-Lib library with focus on performance
"""

from .overlap import SMA
from .optimized import (
    SMA_auto,
    SMA_cumsum,
    get_available_backends,
    HAS_NUMBA,
    HAS_CUPY
)

# Conditionally import optimized versions
if HAS_NUMBA:
    from .optimized import SMA_numba

if HAS_CUPY:
    from .optimized import SMA_gpu

__version__ = "0.1.0"
__all__ = [
    "SMA",
    "SMA_auto",
    "SMA_cumsum",
    "get_available_backends",
    "HAS_NUMBA",
    "HAS_CUPY"
]

# Add optimized versions to __all__ if available
if HAS_NUMBA:
    __all__.append("SMA_numba")

if HAS_CUPY:
    __all__.append("SMA_gpu")
