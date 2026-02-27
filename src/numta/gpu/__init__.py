"""
GPU batch processing backend for numta using Numba CUDA.

Provides batch versions of all indicators that process thousands of tickers
simultaneously on the GPU (one CUDA thread per ticker).
"""

HAS_CUDA = False
_cuda = None

try:
    from numba import cuda
    if cuda.is_available():
        HAS_CUDA = True
        _cuda = cuda
except (ImportError, Exception):
    pass


def gpu_info():
    """Return GPU device information dict, or None if CUDA unavailable."""
    if not HAS_CUDA:
        return None
    device = _cuda.get_current_device()
    return {
        "name": device.name.decode() if isinstance(device.name, bytes) else device.name,
        "compute_capability": device.compute_capability,
        "max_threads_per_block": device.MAX_THREADS_PER_BLOCK,
        "max_grid_dim": device.MAX_GRID_DIM_X,
        "multiprocessors": device.MULTIPROCESSOR_COUNT,
    }
