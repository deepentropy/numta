"""
Test suite for GPU implementations of Volatility and Volume Indicators

This module tests CuPy GPU implementations against CPU implementations
to ensure correctness.
"""

import numpy as np
import pytest

# Try to import CuPy
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

# Import CPU implementations
from talib_pure.cpu.volatility_indicators import _trange_numba
from talib_pure.cpu.volume_indicators import _ad_numba, _adosc_numba, _obv_numba

# Import GPU implementations
if CUPY_AVAILABLE:
    from talib_pure.gpu.volatility_indicators import _trange_cupy, _natr_cupy
    from talib_pure.gpu.volume_indicators import _ad_cupy, _adosc_cupy, _obv_cupy


# Skip all tests if CuPy is not available
pytestmark = pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")


class TestGPUVolatilityIndicators:
    """Test GPU implementations of Volatility Indicators"""

    def test_trange_gpu_vs_cpu(self):
        """Test TRANGE GPU implementation against CPU"""
        np.random.seed(42)
        n = 1000
        close = np.random.randn(n).cumsum() + 100
        high = close + np.random.rand(n) * 2
        low = close - np.random.rand(n) * 2

        # CPU implementation
        cpu_output = np.empty(n, dtype=np.float64)
        _trange_numba(high, low, close, cpu_output)

        # GPU implementation
        gpu_output = _trange_cupy(high, low, close)

        # Compare results
        np.testing.assert_allclose(gpu_output, cpu_output, rtol=1e-10, atol=1e-12)

    def test_trange_gpu_small_dataset(self):
        """Test TRANGE GPU with small dataset"""
        high = np.array([110, 112, 114, 113, 115], dtype=np.float64)
        low = np.array([100, 102, 104, 103, 105], dtype=np.float64)
        close = np.array([105, 107, 109, 108, 110], dtype=np.float64)

        # CPU implementation
        cpu_output = np.empty(5, dtype=np.float64)
        _trange_numba(high, low, close, cpu_output)

        # GPU implementation
        gpu_output = _trange_cupy(high, low, close)

        # Compare results
        np.testing.assert_allclose(gpu_output, cpu_output, rtol=1e-10, atol=1e-12)

        # Verify first value is just high - low
        assert abs(gpu_output[0] - 10.0) < 0.01

    def test_trange_gpu_with_gaps(self):
        """Test TRANGE GPU with price gaps"""
        high = np.array([110, 105, 115], dtype=np.float64)
        low = np.array([100, 95, 105], dtype=np.float64)
        close = np.array([105, 100, 110], dtype=np.float64)

        # CPU implementation
        cpu_output = np.empty(3, dtype=np.float64)
        _trange_numba(high, low, close, cpu_output)

        # GPU implementation
        gpu_output = _trange_cupy(high, low, close)

        # Compare results
        np.testing.assert_allclose(gpu_output, cpu_output, rtol=1e-10, atol=1e-12)

    def test_natr_gpu_basic(self):
        """Test NATR GPU implementation"""
        np.random.seed(42)
        n = 100
        close = np.random.randn(n).cumsum() + 100
        high = close + np.random.rand(n) * 2
        low = close - np.random.rand(n) * 2
        timeperiod = 14

        # GPU implementation
        gpu_output = _natr_cupy(high, low, close, timeperiod)

        # Verify output shape
        assert len(gpu_output) == n

        # Verify NaN for lookback period
        assert np.all(np.isnan(gpu_output[:timeperiod - 1]))

        # Verify values are positive percentages after lookback
        valid_values = gpu_output[timeperiod - 1:]
        valid_values = valid_values[~np.isnan(valid_values)]
        assert np.all(valid_values >= 0)

    def test_natr_gpu_small_dataset(self):
        """Test NATR GPU with small dataset"""
        high = np.array([110, 112, 111, 113, 115, 114, 116, 118, 117, 119,
                         121, 120, 122, 124, 123], dtype=np.float64)
        low = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
                        111, 110, 112, 114, 113], dtype=np.float64)
        close = np.array([105, 107, 106, 108, 110, 109, 111, 113, 112, 114,
                          116, 115, 117, 119, 118], dtype=np.float64)
        timeperiod = 14

        gpu_output = _natr_cupy(high, low, close, timeperiod)

        # Verify NaN for lookback period
        assert np.all(np.isnan(gpu_output[:timeperiod - 1]))

        # Verify valid values exist after lookback
        assert not np.isnan(gpu_output[timeperiod - 1])


class TestGPUVolumeIndicators:
    """Test GPU implementations of Volume Indicators"""

    def test_ad_gpu_vs_cpu(self):
        """Test AD (Accumulation/Distribution) GPU implementation against CPU"""
        np.random.seed(42)
        n = 1000
        close = np.random.randn(n).cumsum() + 100
        high = close + np.random.rand(n) * 2
        low = close - np.random.rand(n) * 2
        volume = np.random.rand(n) * 10000 + 1000

        # CPU implementation
        cpu_output = np.empty(n, dtype=np.float64)
        _ad_numba(high, low, close, volume, cpu_output)

        # GPU implementation
        gpu_output = _ad_cupy(high, low, close, volume)

        # Compare results
        np.testing.assert_allclose(gpu_output, cpu_output, rtol=1e-10, atol=1e-8)

    def test_ad_gpu_small_dataset(self):
        """Test AD GPU with small dataset"""
        high = np.array([10, 11, 12, 11, 13], dtype=np.float64)
        low = np.array([9, 10, 10, 9, 11], dtype=np.float64)
        close = np.array([9.5, 10.5, 11, 10, 12], dtype=np.float64)
        volume = np.array([1000, 1100, 1200, 900, 1300], dtype=np.float64)

        # CPU implementation
        cpu_output = np.empty(5, dtype=np.float64)
        _ad_numba(high, low, close, volume, cpu_output)

        # GPU implementation
        gpu_output = _ad_cupy(high, low, close, volume)

        # Compare results
        np.testing.assert_allclose(gpu_output, cpu_output, rtol=1e-10, atol=1e-8)

    def test_ad_gpu_with_equal_high_low(self):
        """Test AD GPU when high equals low (division by zero case)"""
        high = np.array([10, 10, 12], dtype=np.float64)
        low = np.array([10, 10, 10], dtype=np.float64)
        close = np.array([10, 10, 11], dtype=np.float64)
        volume = np.array([1000, 1000, 1000], dtype=np.float64)

        # CPU implementation
        cpu_output = np.empty(3, dtype=np.float64)
        _ad_numba(high, low, close, volume, cpu_output)

        # GPU implementation
        gpu_output = _ad_cupy(high, low, close, volume)

        # Compare results
        np.testing.assert_allclose(gpu_output, cpu_output, rtol=1e-10, atol=1e-8)

    def test_adosc_gpu_vs_cpu(self):
        """Test ADOSC (Chaikin A/D Oscillator) GPU implementation against CPU"""
        np.random.seed(42)
        n = 100
        close = np.random.randn(n).cumsum() + 100
        high = close + np.random.rand(n) * 2
        low = close - np.random.rand(n) * 2
        volume = np.random.rand(n) * 10000 + 1000
        fastperiod = 3
        slowperiod = 10

        # CPU implementation
        cpu_output = np.empty(n, dtype=np.float64)
        _adosc_numba(high, low, close, volume, fastperiod, slowperiod, cpu_output)

        # GPU implementation
        gpu_output = _adosc_cupy(high, low, close, volume, fastperiod, slowperiod)

        # Compare results
        np.testing.assert_allclose(gpu_output, cpu_output, rtol=1e-8, atol=1e-6)

    def test_adosc_gpu_small_dataset(self):
        """Test ADOSC GPU with small dataset"""
        high = np.array([10, 11, 12, 11, 13, 14, 15, 13, 16, 17, 18], dtype=np.float64)
        low = np.array([9, 10, 10, 9, 11, 12, 13, 11, 14, 15, 16], dtype=np.float64)
        close = np.array([9.5, 10.5, 11, 10, 12, 13, 14, 12, 15, 16, 17], dtype=np.float64)
        volume = np.array([1000, 1100, 1200, 900, 1300, 1400, 1500, 1000, 1600, 1700, 1800], dtype=np.float64)
        fastperiod = 3
        slowperiod = 10

        # CPU implementation
        cpu_output = np.empty(11, dtype=np.float64)
        _adosc_numba(high, low, close, volume, fastperiod, slowperiod, cpu_output)

        # GPU implementation
        gpu_output = _adosc_cupy(high, low, close, volume, fastperiod, slowperiod)

        # Compare results
        np.testing.assert_allclose(gpu_output, cpu_output, rtol=1e-8, atol=1e-6)

        # Verify NaN for lookback period
        assert np.all(np.isnan(gpu_output[:slowperiod - 1]))

    def test_obv_gpu_vs_cpu(self):
        """Test OBV (On Balance Volume) GPU implementation against CPU"""
        np.random.seed(42)
        n = 1000
        close = np.random.randn(n).cumsum() + 100
        volume = np.random.rand(n) * 10000 + 1000

        # CPU implementation
        cpu_output = np.empty(n, dtype=np.float64)
        _obv_numba(close, volume, cpu_output)

        # GPU implementation
        gpu_output = _obv_cupy(close, volume)

        # Compare results
        np.testing.assert_allclose(gpu_output, cpu_output, rtol=1e-10, atol=1e-8)

    def test_obv_gpu_small_dataset(self):
        """Test OBV GPU with small dataset"""
        close = np.array([100, 102, 101, 103, 105, 104, 106], dtype=np.float64)
        volume = np.array([1000, 1500, 1200, 1800, 2000, 1100, 1900], dtype=np.float64)

        # CPU implementation
        cpu_output = np.empty(7, dtype=np.float64)
        _obv_numba(close, volume, cpu_output)

        # GPU implementation
        gpu_output = _obv_cupy(close, volume)

        # Compare results
        np.testing.assert_allclose(gpu_output, cpu_output, rtol=1e-10, atol=1e-8)

        # Verify first value matches TA-Lib convention (first volume)
        assert abs(gpu_output[0] - volume[0]) < 0.01

    def test_obv_gpu_price_changes(self):
        """Test OBV GPU with various price change scenarios"""
        # Price goes up, down, stays same
        close = np.array([100, 102, 101, 101, 103], dtype=np.float64)
        volume = np.array([1000, 1000, 1000, 1000, 1000], dtype=np.float64)

        # CPU implementation
        cpu_output = np.empty(5, dtype=np.float64)
        _obv_numba(close, volume, cpu_output)

        # GPU implementation
        gpu_output = _obv_cupy(close, volume)

        # Compare results
        np.testing.assert_allclose(gpu_output, cpu_output, rtol=1e-10, atol=1e-8)

        # Verify logic:
        # [0]: 1000 (first volume)
        # [1]: 1000 + 1000 = 2000 (price up)
        # [2]: 2000 - 1000 = 1000 (price down)
        # [3]: 1000 (price same)
        # [4]: 1000 + 1000 = 2000 (price up)
        assert abs(gpu_output[0] - 1000) < 0.01
        assert abs(gpu_output[1] - 2000) < 0.01
        assert abs(gpu_output[2] - 1000) < 0.01
        assert abs(gpu_output[3] - 1000) < 0.01
        assert abs(gpu_output[4] - 2000) < 0.01


class TestGPUPerformance:
    """Test GPU performance characteristics"""

    def test_gpu_large_dataset_trange(self):
        """Test TRANGE GPU with large dataset"""
        np.random.seed(42)
        n = 100000
        close = np.random.randn(n).cumsum() + 100
        high = close + np.random.rand(n) * 2
        low = close - np.random.rand(n) * 2

        # GPU implementation should handle large datasets
        gpu_output = _trange_cupy(high, low, close)

        assert len(gpu_output) == n
        assert not np.any(np.isnan(gpu_output))

    def test_gpu_large_dataset_ad(self):
        """Test AD GPU with large dataset"""
        np.random.seed(42)
        n = 100000
        close = np.random.randn(n).cumsum() + 100
        high = close + np.random.rand(n) * 2
        low = close - np.random.rand(n) * 2
        volume = np.random.rand(n) * 10000 + 1000

        # GPU implementation should handle large datasets
        gpu_output = _ad_cupy(high, low, close, volume)

        assert len(gpu_output) == n
        assert not np.any(np.isnan(gpu_output))

    def test_gpu_large_dataset_obv(self):
        """Test OBV GPU with large dataset"""
        np.random.seed(42)
        n = 100000
        close = np.random.randn(n).cumsum() + 100
        volume = np.random.rand(n) * 10000 + 1000

        # GPU implementation should handle large datasets
        gpu_output = _obv_cupy(close, volume)

        assert len(gpu_output) == n
        assert not np.any(np.isnan(gpu_output))
