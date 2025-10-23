"""
Test suite for optimized SMA implementations
"""

import numpy as np
import pytest
from talib_pure import SMA, SMA_cumsum, SMA_auto, HAS_NUMBA, HAS_CUPY, get_available_backends

if HAS_NUMBA:
    from talib_pure import SMA_numba

if HAS_CUPY:
    from talib_pure import SMA_gpu


def test_get_available_backends():
    """Test backend availability check"""
    backends = get_available_backends()

    assert "numpy" in backends
    assert "cumsum" in backends
    assert "numba" in backends
    assert "gpu" in backends

    assert backends["numpy"]["available"] is True
    assert backends["cumsum"]["available"] is True


def test_sma_cumsum_vs_original():
    """Test cumsum implementation matches original"""
    close = np.random.uniform(100, 200, 1000)

    original = SMA(close, timeperiod=30)
    cumsum = SMA_cumsum(close, timeperiod=30)

    np.testing.assert_array_almost_equal(
        original, cumsum, decimal=10,
        err_msg="SMA_cumsum differs from original SMA"
    )


def test_sma_cumsum_various_timeperiods():
    """Test cumsum with various timeperiods"""
    close = np.random.uniform(100, 200, 500)

    for timeperiod in [5, 10, 20, 50, 100]:
        original = SMA(close, timeperiod=timeperiod)
        cumsum = SMA_cumsum(close, timeperiod=timeperiod)

        np.testing.assert_array_almost_equal(
            original, cumsum, decimal=10,
            err_msg=f"SMA_cumsum differs at timeperiod={timeperiod}"
        )


def test_sma_cumsum_edge_cases():
    """Test cumsum edge cases"""
    # Empty array
    assert len(SMA_cumsum(np.array([]), timeperiod=5)) == 0

    # Insufficient data
    close = np.array([1, 2, 3], dtype=np.float64)
    result = SMA_cumsum(close, timeperiod=5)
    assert all(np.isnan(result))

    # Single element
    close = np.array([100.0], dtype=np.float64)
    result = SMA_cumsum(close, timeperiod=1)
    assert result[0] == 100.0


@pytest.mark.skipif(not HAS_NUMBA, reason="Numba not installed")
def test_sma_numba_vs_original():
    """Test Numba implementation matches original"""
    close = np.random.uniform(100, 200, 1000)

    original = SMA(close, timeperiod=30)
    numba = SMA_numba(close, timeperiod=30)

    np.testing.assert_array_almost_equal(
        original, numba, decimal=10,
        err_msg="SMA_numba differs from original SMA"
    )


@pytest.mark.skipif(not HAS_NUMBA, reason="Numba not installed")
def test_sma_numba_various_timeperiods():
    """Test Numba with various timeperiods"""
    close = np.random.uniform(100, 200, 500)

    for timeperiod in [5, 10, 20, 50, 100]:
        original = SMA(close, timeperiod=timeperiod)
        numba = SMA_numba(close, timeperiod=timeperiod)

        np.testing.assert_array_almost_equal(
            original, numba, decimal=10,
            err_msg=f"SMA_numba differs at timeperiod={timeperiod}"
        )


@pytest.mark.skipif(not HAS_NUMBA, reason="Numba not installed")
def test_sma_numba_edge_cases():
    """Test Numba edge cases"""
    # Empty array
    assert len(SMA_numba(np.array([]), timeperiod=5)) == 0

    # Insufficient data
    close = np.array([1, 2, 3], dtype=np.float64)
    result = SMA_numba(close, timeperiod=5)
    assert all(np.isnan(result))

    # Single element
    close = np.array([100.0], dtype=np.float64)
    result = SMA_numba(close, timeperiod=1)
    np.testing.assert_almost_equal(result[0], 100.0, decimal=10)


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
def test_sma_gpu_vs_original():
    """Test GPU implementation matches original"""
    close = np.random.uniform(100, 200, 1000)

    original = SMA(close, timeperiod=30)
    gpu = SMA_gpu(close, timeperiod=30)

    np.testing.assert_array_almost_equal(
        original, gpu, decimal=10,
        err_msg="SMA_gpu differs from original SMA"
    )


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
def test_sma_gpu_various_timeperiods():
    """Test GPU with various timeperiods"""
    close = np.random.uniform(100, 200, 500)

    for timeperiod in [5, 10, 20, 50, 100]:
        original = SMA(close, timeperiod=timeperiod)
        gpu = SMA_gpu(close, timeperiod=timeperiod)

        np.testing.assert_array_almost_equal(
            original, gpu, decimal=10,
            err_msg=f"SMA_gpu differs at timeperiod={timeperiod}"
        )


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
def test_sma_gpu_edge_cases():
    """Test GPU edge cases"""
    # Empty array
    assert len(SMA_gpu(np.array([]), timeperiod=5)) == 0

    # Insufficient data
    close = np.array([1, 2, 3], dtype=np.float64)
    result = SMA_gpu(close, timeperiod=5)
    assert all(np.isnan(result))

    # Single element
    close = np.array([100.0], dtype=np.float64)
    result = SMA_gpu(close, timeperiod=1)
    np.testing.assert_almost_equal(result[0], 100.0, decimal=10)


def test_sma_auto_numpy_backend():
    """Test auto with numpy backend"""
    close = np.random.uniform(100, 200, 100)

    result_auto = SMA_auto(close, timeperiod=30, backend="numpy")
    result_original = SMA(close, timeperiod=30)

    np.testing.assert_array_almost_equal(result_auto, result_original, decimal=10)


def test_sma_auto_cumsum_backend():
    """Test auto with cumsum backend"""
    close = np.random.uniform(100, 200, 100)

    result_auto = SMA_auto(close, timeperiod=30, backend="cumsum")
    result_original = SMA(close, timeperiod=30)

    np.testing.assert_array_almost_equal(result_auto, result_original, decimal=10)


@pytest.mark.skipif(not HAS_NUMBA, reason="Numba not installed")
def test_sma_auto_numba_backend():
    """Test auto with numba backend"""
    close = np.random.uniform(100, 200, 100)

    result_auto = SMA_auto(close, timeperiod=30, backend="numba")
    result_original = SMA(close, timeperiod=30)

    np.testing.assert_array_almost_equal(result_auto, result_original, decimal=10)


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
def test_sma_auto_gpu_backend():
    """Test auto with GPU backend"""
    close = np.random.uniform(100, 200, 100)

    result_auto = SMA_auto(close, timeperiod=30, backend="gpu")
    result_original = SMA(close, timeperiod=30)

    np.testing.assert_array_almost_equal(result_auto, result_original, decimal=10)


def test_sma_auto_auto_backend():
    """Test auto with automatic backend selection"""
    close = np.random.uniform(100, 200, 100)

    result_auto = SMA_auto(close, timeperiod=30, backend="auto")
    result_original = SMA(close, timeperiod=30)

    np.testing.assert_array_almost_equal(result_auto, result_original, decimal=10)


def test_sma_auto_invalid_backend():
    """Test auto with invalid backend raises error"""
    close = np.random.uniform(100, 200, 100)

    with pytest.raises(ValueError, match="Unknown backend"):
        SMA_auto(close, timeperiod=30, backend="invalid")


def test_sma_auto_unavailable_backend():
    """Test that requesting unavailable backend raises ImportError"""
    close = np.random.uniform(100, 200, 100)

    if not HAS_NUMBA:
        with pytest.raises(ImportError, match="Numba is not installed"):
            SMA_auto(close, timeperiod=30, backend="numba")

    if not HAS_CUPY:
        with pytest.raises(ImportError, match="CuPy is not installed"):
            SMA_auto(close, timeperiod=30, backend="gpu")


def test_sma_cumsum_invalid_timeperiod():
    """Test cumsum raises error for invalid timeperiod"""
    close = np.array([1, 2, 3, 4, 5], dtype=np.float64)

    with pytest.raises(ValueError):
        SMA_cumsum(close, timeperiod=0)

    with pytest.raises(ValueError):
        SMA_cumsum(close, timeperiod=-1)


@pytest.mark.skipif(not HAS_NUMBA, reason="Numba not installed")
def test_sma_numba_invalid_timeperiod():
    """Test Numba raises error for invalid timeperiod"""
    close = np.array([1, 2, 3, 4, 5], dtype=np.float64)

    with pytest.raises(ValueError):
        SMA_numba(close, timeperiod=0)

    with pytest.raises(ValueError):
        SMA_numba(close, timeperiod=-1)


def test_all_implementations_agree():
    """Test that all available implementations produce identical results"""
    close = np.random.uniform(100, 200, 500)
    timeperiod = 30

    # Get original result
    original = SMA(close, timeperiod=timeperiod)

    # Test cumsum
    cumsum = SMA_cumsum(close, timeperiod=timeperiod)
    np.testing.assert_array_almost_equal(original, cumsum, decimal=10)

    # Test numba if available
    if HAS_NUMBA:
        numba = SMA_numba(close, timeperiod=timeperiod)
        np.testing.assert_array_almost_equal(original, numba, decimal=10)

    # Test GPU if available
    if HAS_CUPY:
        gpu = SMA_gpu(close, timeperiod=timeperiod)
        np.testing.assert_array_almost_equal(original, gpu, decimal=10)
