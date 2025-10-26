"""
Backend configuration for talib-pure

Supports both CPU (Numba) and GPU (CuPy/CUDA) backends for accelerated computation.
"""

import os
from typing import Dict, Optional

# Current backend selection
_current_backend: str = "cpu"
_gpu_available: Optional[bool] = None


def is_gpu_available() -> bool:
    """
    Check if GPU (CuPy/CUDA) is available

    Returns
    -------
    bool
        True if CuPy and CUDA are available, False otherwise

    Notes
    -----
    This function caches the result after the first check for performance.
    """
    global _gpu_available

    if _gpu_available is not None:
        return _gpu_available

    try:
        import cupy as cp
        # Try a simple operation to verify CUDA is working
        _ = cp.array([1.0])
        _gpu_available = True
    except (ImportError, Exception):
        _gpu_available = False

    return _gpu_available


def set_backend(backend: str) -> None:
    """
    Set the computation backend

    Parameters
    ----------
    backend : str
        Backend to use: "cpu" or "gpu"

    Raises
    ------
    ValueError
        If backend is not "cpu" or "gpu"
    RuntimeError
        If "gpu" is selected but GPU is not available

    Examples
    --------
    >>> from talib_pure import set_backend
    >>> set_backend("cpu")  # Use CPU with Numba
    >>> set_backend("gpu")  # Use GPU with CuPy (if available)
    """
    global _current_backend

    backend = backend.lower()

    if backend not in ("cpu", "gpu"):
        raise ValueError(f"Invalid backend: {backend}. Must be 'cpu' or 'gpu'")

    if backend == "gpu" and not is_gpu_available():
        raise RuntimeError(
            "GPU backend requested but CuPy/CUDA is not available. "
            "Install CuPy with: pip install cupy-cuda11x (or appropriate CUDA version)"
        )

    _current_backend = backend


def get_backend() -> str:
    """
    Get the current computation backend

    Returns
    -------
    str
        Current backend: "cpu" or "gpu"

    Examples
    --------
    >>> from talib_pure import get_backend
    >>> backend = get_backend()
    >>> print(f"Current backend: {backend}")
    """
    return _current_backend


def get_backend_info() -> Dict[str, any]:
    """
    Get information about available backends and current configuration

    Returns
    -------
    dict
        Dictionary with backend information:
        - 'current': Current backend ("cpu" or "gpu")
        - 'gpu_available': Whether GPU/CuPy is available
        - 'numba_available': Whether Numba is available
        - 'cuda_version': CUDA version if GPU is available, None otherwise
        - 'cupy_version': CuPy version if available, None otherwise

    Examples
    --------
    >>> from talib_pure import get_backend_info
    >>> info = get_backend_info()
    >>> print(f"Current: {info['current']}")
    >>> print(f"GPU Available: {info['gpu_available']}")
    """
    info = {
        'current': _current_backend,
        'gpu_available': is_gpu_available(),
        'numba_available': False,
        'cuda_version': None,
        'cupy_version': None,
    }

    # Check Numba availability
    try:
        import numba
        info['numba_available'] = True
        info['numba_version'] = numba.__version__
    except ImportError:
        pass

    # Get GPU info if available
    if is_gpu_available():
        try:
            import cupy as cp
            info['cupy_version'] = cp.__version__
            info['cuda_version'] = cp.cuda.runtime.runtimeGetVersion()
            info['device_name'] = cp.cuda.Device().name
            info['device_compute_capability'] = cp.cuda.Device().compute_capability
        except Exception:
            pass

    return info


# Auto-detect GPU on import if environment variable is set
_auto_gpu = os.environ.get('TALIB_PURE_AUTO_GPU', '').lower() in ('1', 'true', 'yes')
if _auto_gpu and is_gpu_available():
    _current_backend = "gpu"
