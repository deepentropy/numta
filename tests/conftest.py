"""
Shared pytest configuration and fixtures for numta test suite.
"""

import pytest
import numpy as np
from typing import Optional, Tuple

# =====================================================================
# Conditional Imports
# =====================================================================

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

try:
    from numba import cuda
    cuda.detect()
    HAS_CUDA = True
except Exception:
    HAS_CUDA = False

try:
    import pandas_ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False


# =====================================================================
# Custom Pytest Markers
# =====================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "numba: marks tests requiring numba (skip if not available)"
    )
    config.addinivalue_line(
        "markers", "pandas: marks tests requiring pandas (skip if not available)"
    )
    config.addinivalue_line(
        "markers", "talib: marks tests requiring TA-Lib (skip if not available)"
    )
    config.addinivalue_line(
        "markers", "pandas_ta: marks tests requiring pandas-ta (skip if not available)"
    )
    config.addinivalue_line(
        "markers", "cuda: marks tests requiring CUDA GPU (skip if not available)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as benchmark tests"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically skip tests based on marker and availability."""
    skip_numba = pytest.mark.skip(reason="numba not installed")
    skip_pandas = pytest.mark.skip(reason="pandas not installed")
    skip_talib = pytest.mark.skip(reason="TA-Lib not installed")
    skip_pandas_ta = pytest.mark.skip(reason="pandas-ta not installed")
    skip_cuda = pytest.mark.skip(reason="CUDA GPU not available")

    for item in items:
        if "numba" in item.keywords and not HAS_NUMBA:
            item.add_marker(skip_numba)
        if "pandas" in item.keywords and not HAS_PANDAS:
            item.add_marker(skip_pandas)
        if "talib" in item.keywords and not HAS_TALIB:
            item.add_marker(skip_talib)
        if "pandas_ta" in item.keywords and not HAS_PANDAS_TA:
            item.add_marker(skip_pandas_ta)
        if "cuda" in item.keywords and not HAS_CUDA:
            item.add_marker(skip_cuda)


# =====================================================================
# Data Generation Fixtures
# =====================================================================

RANDOM_SEED = 42


@pytest.fixture
def sample_ohlcv_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate sample OHLCV data as numpy arrays.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (open, high, low, close, volume) arrays with 100 data points
    """
    np.random.seed(RANDOM_SEED)
    n = 100
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.5)
    low = close - np.abs(np.random.randn(n) * 0.5)
    open_ = close + np.random.randn(n) * 0.3
    volume = np.random.randint(1000, 10000, n).astype(np.float64)
    
    return open_, high, low, close, volume


@pytest.fixture
def sample_close_data() -> np.ndarray:
    """
    Generate sample close price data.
    
    Returns
    -------
    np.ndarray
        Close price array with 100 data points
    """
    np.random.seed(RANDOM_SEED)
    return 100 + np.cumsum(np.random.randn(100) * 0.5)


@pytest.fixture
def sample_ohlcv_dataframe(sample_ohlcv_data):
    """
    Generate sample OHLCV data as pandas DataFrame.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with open, high, low, close, volume columns
    """
    if not HAS_PANDAS:
        pytest.skip("pandas not installed")
    
    open_, high, low, close, volume = sample_ohlcv_data
    return pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })


@pytest.fixture
def sample_ohlcv_dataframe_with_datetime(sample_ohlcv_data):
    """
    Generate sample OHLCV DataFrame with DatetimeIndex.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex
    """
    if not HAS_PANDAS:
        pytest.skip("pandas not installed")
    
    open_, high, low, close, volume = sample_ohlcv_data
    index = pd.date_range('2020-01-01', periods=len(close), freq='D')
    return pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=index)


@pytest.fixture
def large_sample_ohlcv_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate larger sample OHLCV data for performance testing.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (open, high, low, close, volume) arrays with 10000 data points
    """
    np.random.seed(RANDOM_SEED)
    n = 10000
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.5)
    low = close - np.abs(np.random.randn(n) * 0.5)
    open_ = close + np.random.randn(n) * 0.3
    volume = np.random.randint(1000, 10000, n).astype(np.float64)
    
    return open_, high, low, close, volume


# =====================================================================
# Edge Case Fixtures
# =====================================================================

@pytest.fixture
def edge_case_data():
    """
    Generate edge case test data.
    
    Returns
    -------
    dict
        Dictionary with various edge case data:
        - 'empty': empty array
        - 'single': single value array
        - 'two': two value array
        - 'constant': array with constant values
        - 'with_nan': array with NaN values
        - 'with_inf': array with inf values
        - 'all_nan': array with all NaN values
        - 'negative': array with negative values
        - 'zeros': array with zeros
    """
    return {
        'empty': np.array([], dtype=np.float64),
        'single': np.array([100.0], dtype=np.float64),
        'two': np.array([100.0, 101.0], dtype=np.float64),
        'constant': np.full(50, 100.0, dtype=np.float64),
        'with_nan': np.array([100.0, np.nan, 101.0, 102.0, np.nan, 103.0], dtype=np.float64),
        'with_inf': np.array([100.0, np.inf, 101.0, -np.inf, 102.0], dtype=np.float64),
        'all_nan': np.full(10, np.nan, dtype=np.float64),
        'negative': np.array([-100.0, -99.0, -98.0, -97.0, -96.0], dtype=np.float64),
        'zeros': np.zeros(10, dtype=np.float64),
    }


@pytest.fixture
def edge_case_ohlcv_data():
    """
    Generate edge case OHLCV test data.
    
    Returns
    -------
    dict
        Dictionary with OHLCV edge case data:
        - 'empty': tuple of empty arrays
        - 'single': tuple of single value arrays
        - 'constant': tuple of constant value arrays
    """
    empty = np.array([], dtype=np.float64)
    single = np.array([100.0], dtype=np.float64)
    constant = np.full(50, 100.0, dtype=np.float64)
    
    return {
        'empty': (empty, empty, empty, empty, empty),
        'single': (single, single + 1, single - 1, single, single * 10),
        'constant': (constant, constant + 1, constant - 1, constant, constant * 10),
    }


# =====================================================================
# Data Type Generators
# =====================================================================

@pytest.fixture
def data_generators():
    """
    Provide data generators for different data types.
    
    Returns
    -------
    dict
        Dictionary of data generators:
        - 'random': random walk data
        - 'trending': upward trending data with noise
        - 'cyclical': sine wave with noise
        - 'volatile': high volatility data
    """
    def generate_random(size: int = 1000, seed: int = 42) -> np.ndarray:
        """Generate random walk data."""
        np.random.seed(seed)
        return 100 + np.cumsum(np.random.randn(size) * 0.5)
    
    def generate_trending(size: int = 1000, seed: int = 42) -> np.ndarray:
        """Generate trending data with noise."""
        np.random.seed(seed)
        trend = np.linspace(100, 120, size)
        noise = np.random.randn(size) * 0.5
        return trend + noise
    
    def generate_cyclical(size: int = 1000, seed: int = 42) -> np.ndarray:
        """Generate cyclical data with noise."""
        np.random.seed(seed)
        x = np.linspace(0, 10 * np.pi, size)
        cycle = 10 * np.sin(x) + 100
        noise = np.random.randn(size) * 0.3
        return cycle + noise
    
    def generate_volatile(size: int = 1000, seed: int = 42) -> np.ndarray:
        """Generate high volatility data."""
        np.random.seed(seed)
        return 100 + np.cumsum(np.random.randn(size) * 2.0)
    
    return {
        'random': generate_random,
        'trending': generate_trending,
        'cyclical': generate_cyclical,
        'volatile': generate_volatile,
    }


# =====================================================================
# Helper Functions
# =====================================================================

def create_ohlcv_from_close(close: np.ndarray, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create OHLCV data from close prices.
    
    Parameters
    ----------
    close : np.ndarray
        Close price array
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    Tuple
        (open, high, low, close, volume) arrays
    """
    np.random.seed(seed)
    n = len(close)
    high = close + np.abs(np.random.randn(n) * 0.5)
    low = close - np.abs(np.random.randn(n) * 0.5)
    open_ = close + np.random.randn(n) * 0.3
    volume = np.random.randint(1000, 10000, n).astype(np.float64)
    return open_, high, low, close, volume


def arrays_almost_equal(a: np.ndarray, b: np.ndarray, decimal: int = 10) -> bool:
    """
    Check if two arrays are almost equal, ignoring NaN positions.
    
    Parameters
    ----------
    a, b : np.ndarray
        Arrays to compare
    decimal : int
        Number of decimal places for comparison
    
    Returns
    -------
    bool
        True if arrays are almost equal
    """
    # Handle different lengths
    if len(a) != len(b):
        return False
    
    # Find valid (non-NaN) positions in both arrays
    valid_mask = ~(np.isnan(a) | np.isnan(b))
    
    if not np.any(valid_mask):
        return True  # Both all NaN
    
    try:
        np.testing.assert_array_almost_equal(a[valid_mask], b[valid_mask], decimal=decimal)
        return True
    except AssertionError:
        return False
