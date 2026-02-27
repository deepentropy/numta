"""
GPU memory utilities for batch processing.

Handles host-to-device transfers, output allocation, and grid computation.
"""

import numpy as np
from numba import cuda


def to_device_2d(data, dtype=np.float64):
    """
    Transfer a 2D numpy array to GPU device memory.

    Parameters
    ----------
    data : np.ndarray
        2D array of shape (num_tickers, num_bars)
    dtype : np.dtype
        Data type (default: float64)

    Returns
    -------
    numba.cuda.devicearray.DeviceNDArray
        Device array
    """
    data = np.ascontiguousarray(data, dtype=dtype)
    return cuda.to_device(data)


def allocate_output_2d(num_tickers, num_bars, dtype=np.float64):
    """
    Allocate a 2D output array on GPU device memory.

    Returns
    -------
    numba.cuda.devicearray.DeviceNDArray
        Device array filled with NaN (float) or zeros (int)
    """
    if np.issubdtype(dtype, np.integer):
        host = np.zeros((num_tickers, num_bars), dtype=dtype)
    else:
        host = np.full((num_tickers, num_bars), np.nan, dtype=dtype)
    return cuda.to_device(host)


def compute_grid_1d(num_tickers, threads_per_block=256):
    """
    Compute CUDA grid dimensions for 1D launch (one thread per ticker).

    Returns
    -------
    tuple
        (blocks_per_grid, threads_per_block)
    """
    blocks = (num_tickers + threads_per_block - 1) // threads_per_block
    return blocks, threads_per_block
