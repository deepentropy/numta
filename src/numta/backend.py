"""
Backend configuration for numta

Uses CPU (Numba JIT) for accelerated computation.
GPU (Numba CUDA) available for batch processing via *_batch() functions.
"""

from typing import Dict

_current_backend: str = "cpu"
_VALID_BACKENDS = {"cpu", "gpu"}


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
        If "gpu" is requested but CUDA is not available

    Examples
    --------
    >>> from numta import set_backend
    >>> set_backend("cpu")  # Use CPU with Numba
    >>> set_backend("gpu")  # Use GPU (requires CUDA)
    """
    global _current_backend

    backend = backend.lower()

    if backend not in _VALID_BACKENDS:
        raise ValueError(f"Invalid backend: {backend}. Supported: {sorted(_VALID_BACKENDS)}")

    if backend == "gpu":
        from .gpu import HAS_CUDA
        if not HAS_CUDA:
            raise RuntimeError(
                "CUDA is not available. Install numba-cuda and ensure a CUDA-capable GPU is present."
            )

    _current_backend = backend


def get_backend() -> str:
    """
    Get the current computation backend

    Returns
    -------
    str
        Current backend ("cpu" or "gpu")

    Examples
    --------
    >>> from numta import get_backend
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
        - 'current': Current backend
        - 'numba_available': Whether Numba is available
        - 'numba_version': Numba version if available
        - 'cuda_available': Whether CUDA is available
        - 'gpu_info': GPU device info dict (if CUDA available)

    Examples
    --------
    >>> from numta import get_backend_info
    >>> info = get_backend_info()
    >>> print(f"Current: {info['current']}")
    >>> print(f"CUDA Available: {info['cuda_available']}")
    """
    info = {
        'current': _current_backend,
        'numba_available': False,
        'cuda_available': False,
    }

    # Check Numba availability
    try:
        import numba
        info['numba_available'] = True
        info['numba_version'] = numba.__version__
    except ImportError:
        pass

    # Check CUDA availability
    from .gpu import HAS_CUDA, gpu_info
    info['cuda_available'] = HAS_CUDA
    if HAS_CUDA:
        info['gpu_info'] = gpu_info()

    return info
