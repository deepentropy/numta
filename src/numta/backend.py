"""
Backend configuration for talib-pure

Uses CPU (Numba JIT) for accelerated computation.
"""

from typing import Dict

# Current backend selection (CPU only)
_current_backend: str = "cpu"


def set_backend(backend: str) -> None:
    """
    Set the computation backend

    Parameters
    ----------
    backend : str
        Backend to use: only "cpu" is supported

    Raises
    ------
    ValueError
        If backend is not "cpu"

    Examples
    --------
    >>> from talib_pure import set_backend
    >>> set_backend("cpu")  # Use CPU with Numba
    """
    global _current_backend

    backend = backend.lower()

    if backend != "cpu":
        raise ValueError(f"Invalid backend: {backend}. Only 'cpu' backend is supported")

    _current_backend = backend


def get_backend() -> str:
    """
    Get the current computation backend

    Returns
    -------
    str
        Current backend (always "cpu")

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
        - 'current': Current backend (always "cpu")
        - 'numba_available': Whether Numba is available
        - 'numba_version': Numba version if available

    Examples
    --------
    >>> from talib_pure import get_backend_info
    >>> info = get_backend_info()
    >>> print(f"Current: {info['current']}")
    >>> print(f"Numba Available: {info['numba_available']}")
    """
    info = {
        'current': _current_backend,
        'numba_available': False,
    }

    # Check Numba availability
    try:
        import numba
        info['numba_available'] = True
        info['numba_version'] = numba.__version__
    except ImportError:
        pass

    return info
