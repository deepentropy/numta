"""
Accuracy testing framework for numta.

This module provides infrastructure for testing the accuracy of numta
functions against TA-Lib and pandas-ta implementations.
"""

import pytest
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Callable, Optional, Tuple

import numta

# Optional imports
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

try:
    import pandas_ta
    import pandas as pd
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    pd = None


# =====================================================================
# Data Classes
# =====================================================================

@dataclass
class AccuracyMetrics:
    """Metrics for comparing two result arrays."""
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Square Error
    max_error: float  # Maximum absolute error
    correlation: float  # Pearson correlation coefficient
    match_rate: float  # Percentage of values within tolerance
    valid_count: int  # Number of valid (non-NaN) comparisons
    total_count: int  # Total number of elements
    
    def classification(self) -> str:
        """
        Classify the accuracy level.
        
        Returns
        -------
        str
            Classification: EXACT, NEAR-EXACT, VERY HIGH, HIGH, or MODERATE
        """
        if self.mae < 1e-10 and self.correlation > 0.999999:
            return "EXACT"
        elif self.mae < 1e-6 and self.correlation > 0.99999:
            return "NEAR-EXACT"
        elif self.mae < 1e-3 and self.correlation > 0.9999:
            return "VERY HIGH"
        elif self.mae < 0.01 and self.correlation > 0.999:
            return "HIGH"
        else:
            return "MODERATE"


# =====================================================================
# Comparison Functions
# =====================================================================

def compare_results(
    result_a: np.ndarray,
    result_b: np.ndarray,
    tolerance: float = 1e-6
) -> AccuracyMetrics:
    """
    Compare two result arrays and compute accuracy metrics.
    
    Parameters
    ----------
    result_a, result_b : np.ndarray
        Arrays to compare
    tolerance : float
        Tolerance for match rate calculation
    
    Returns
    -------
    AccuracyMetrics
        Computed metrics
    """
    # Handle different lengths
    if len(result_a) != len(result_b):
        raise ValueError(f"Arrays must have same length: {len(result_a)} vs {len(result_b)}")
    
    # Find valid (non-NaN) positions in both arrays
    valid_mask = ~(np.isnan(result_a) | np.isnan(result_b))
    valid_a = result_a[valid_mask]
    valid_b = result_b[valid_mask]
    
    if len(valid_a) == 0:
        return AccuracyMetrics(
            mae=0.0,
            rmse=0.0,
            max_error=0.0,
            correlation=1.0,
            match_rate=100.0,
            valid_count=0,
            total_count=len(result_a),
        )
    
    # Calculate metrics
    diff = np.abs(valid_a - valid_b)
    mae = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean((valid_a - valid_b) ** 2)))
    max_error = float(np.max(diff))
    
    # Correlation coefficient
    if np.std(valid_a) > 0 and np.std(valid_b) > 0:
        correlation = float(np.corrcoef(valid_a, valid_b)[0, 1])
    else:
        correlation = 1.0 if mae < 1e-10 else 0.0
    
    # Match rate (within tolerance)
    matches = np.sum(diff < tolerance)
    match_rate = float((matches / len(valid_a)) * 100)
    
    return AccuracyMetrics(
        mae=mae,
        rmse=rmse,
        max_error=max_error,
        correlation=correlation,
        match_rate=match_rate,
        valid_count=len(valid_a),
        total_count=len(result_a),
    )


# =====================================================================
# Data Generators
# =====================================================================

DATA_TYPES = {
    'random': 'Random walk data',
    'trending': 'Upward trend with noise',
    'cyclical': 'Sinusoidal pattern with noise',
    'volatile': 'High volatility data',
}


def generate_test_data(size: int, data_type: str, seed: int = 42) -> np.ndarray:
    """
    Generate test data of a specific type.
    
    Parameters
    ----------
    size : int
        Number of data points
    data_type : str
        Type of data: 'random', 'trending', 'cyclical', 'volatile'
    seed : int
        Random seed
    
    Returns
    -------
    np.ndarray
        Generated data
    """
    np.random.seed(seed)
    
    if data_type == 'random':
        return 100 + np.cumsum(np.random.randn(size) * 0.5)
    elif data_type == 'trending':
        trend = np.linspace(100, 120, size)
        noise = np.random.randn(size) * 0.5
        return trend + noise
    elif data_type == 'cyclical':
        x = np.linspace(0, 10 * np.pi, size)
        cycle = 10 * np.sin(x) + 100
        noise = np.random.randn(size) * 0.3
        return cycle + noise
    elif data_type == 'volatile':
        return 100 + np.cumsum(np.random.randn(size) * 2.0)
    else:
        raise ValueError(f"Unknown data type: {data_type}")


def generate_ohlcv_data(
    size: int,
    data_type: str = 'random',
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate OHLCV test data.
    
    Parameters
    ----------
    size : int
        Number of data points
    data_type : str
        Type of data
    seed : int
        Random seed
    
    Returns
    -------
    Tuple
        (open, high, low, close, volume) arrays
    """
    close = generate_test_data(size, data_type, seed)
    np.random.seed(seed + 1)
    high = close + np.abs(np.random.randn(size) * 0.5)
    low = close - np.abs(np.random.randn(size) * 0.5)
    open_ = close + np.random.randn(size) * 0.3
    volume = np.random.randint(1000, 10000, size).astype(np.float64)
    return open_, high, low, close, volume


# =====================================================================
# Test Classes
# =====================================================================

@pytest.mark.talib
class TestTaLibAccuracy:
    """Tests comparing numta accuracy against TA-Lib."""
    
    @pytest.fixture(autouse=True)
    def check_talib(self):
        """Skip if TA-Lib not available."""
        if not HAS_TALIB:
            pytest.skip("TA-Lib not installed")
    
    @pytest.fixture
    def test_data(self):
        """Generate test data for each test."""
        return {
            data_type: generate_ohlcv_data(1000, data_type)
            for data_type in DATA_TYPES.keys()
        }
    
    @pytest.mark.parametrize("data_type", list(DATA_TYPES.keys()))
    def test_sma_accuracy(self, test_data, data_type):
        """Test SMA accuracy against TA-Lib."""
        _, _, _, close, _ = test_data[data_type]
        
        numta_result = numta.SMA(close, timeperiod=20)
        talib_result = talib.SMA(close, timeperiod=20)
        
        metrics = compare_results(numta_result, talib_result)
        
        assert metrics.correlation > 0.999, f"Low correlation: {metrics.correlation}"
        assert metrics.mae < 1e-6, f"High MAE: {metrics.mae}"
    
    @pytest.mark.parametrize("data_type", list(DATA_TYPES.keys()))
    def test_ema_accuracy(self, test_data, data_type):
        """Test EMA accuracy against TA-Lib."""
        _, _, _, close, _ = test_data[data_type]
        
        numta_result = numta.EMA(close, timeperiod=20)
        talib_result = talib.EMA(close, timeperiod=20)
        
        metrics = compare_results(numta_result, talib_result)
        
        assert metrics.correlation > 0.999, f"Low correlation: {metrics.correlation}"
    
    @pytest.mark.parametrize("data_type", list(DATA_TYPES.keys()))
    def test_rsi_accuracy(self, test_data, data_type):
        """Test RSI accuracy against TA-Lib."""
        _, _, _, close, _ = test_data[data_type]
        
        numta_result = numta.RSI(close, timeperiod=14)
        talib_result = talib.RSI(close, timeperiod=14)
        
        metrics = compare_results(numta_result, talib_result)
        
        assert metrics.correlation > 0.99, f"Low correlation: {metrics.correlation}"
    
    @pytest.mark.parametrize("data_type", list(DATA_TYPES.keys()))
    def test_macd_accuracy(self, test_data, data_type):
        """Test MACD accuracy against TA-Lib."""
        _, _, _, close, _ = test_data[data_type]
        
        numta_macd, numta_signal, numta_hist = numta.MACD(close)
        talib_macd, talib_signal, talib_hist = talib.MACD(close)
        
        # Check each output
        for name, numta_out, talib_out in [
            ('MACD', numta_macd, talib_macd),
            ('Signal', numta_signal, talib_signal),
            ('Histogram', numta_hist, talib_hist),
        ]:
            metrics = compare_results(numta_out, talib_out)
            assert metrics.correlation > 0.99, f"{name} low correlation: {metrics.correlation}"
    
    @pytest.mark.parametrize("data_type", list(DATA_TYPES.keys()))
    def test_bbands_accuracy(self, test_data, data_type):
        """Test Bollinger Bands accuracy against TA-Lib."""
        _, _, _, close, _ = test_data[data_type]
        
        numta_upper, numta_middle, numta_lower = numta.BBANDS(close, timeperiod=20)
        talib_upper, talib_middle, talib_lower = talib.BBANDS(close, timeperiod=20)
        
        for name, numta_out, talib_out in [
            ('Upper', numta_upper, talib_upper),
            ('Middle', numta_middle, talib_middle),
            ('Lower', numta_lower, talib_lower),
        ]:
            metrics = compare_results(numta_out, talib_out)
            assert metrics.correlation > 0.999, f"{name} low correlation: {metrics.correlation}"
    
    @pytest.mark.parametrize("data_type", list(DATA_TYPES.keys()))
    def test_atr_accuracy(self, test_data, data_type):
        """Test ATR accuracy against TA-Lib."""
        _, high, low, close, _ = test_data[data_type]
        
        numta_result = numta.ATR(high, low, close, timeperiod=14)
        talib_result = talib.ATR(high, low, close, timeperiod=14)
        
        metrics = compare_results(numta_result, talib_result)
        
        assert metrics.correlation > 0.99, f"Low correlation: {metrics.correlation}"
    
    @pytest.mark.parametrize("data_type", list(DATA_TYPES.keys()))
    def test_adx_accuracy(self, test_data, data_type):
        """Test ADX accuracy against TA-Lib."""
        _, high, low, close, _ = test_data[data_type]
        
        numta_result = numta.ADX(high, low, close, timeperiod=14)
        talib_result = talib.ADX(high, low, close, timeperiod=14)
        
        metrics = compare_results(numta_result, talib_result)
        
        assert metrics.correlation > 0.99, f"Low correlation: {metrics.correlation}"
    
    @pytest.mark.parametrize("data_type", list(DATA_TYPES.keys()))
    def test_stoch_accuracy(self, test_data, data_type):
        """Test Stochastic accuracy against TA-Lib."""
        _, high, low, close, _ = test_data[data_type]
        
        numta_slowk, numta_slowd = numta.STOCH(high, low, close)
        talib_slowk, talib_slowd = talib.STOCH(high, low, close)
        
        for name, numta_out, talib_out in [
            ('SlowK', numta_slowk, talib_slowk),
            ('SlowD', numta_slowd, talib_slowd),
        ]:
            metrics = compare_results(numta_out, talib_out)
            assert metrics.correlation > 0.99, f"{name} low correlation: {metrics.correlation}"


@pytest.mark.pandas_ta
class TestPandasTaAccuracy:
    """Tests comparing numta accuracy against pandas-ta."""
    
    @pytest.fixture(autouse=True)
    def check_pandas_ta(self):
        """Skip if pandas-ta not available."""
        if not HAS_PANDAS_TA:
            pytest.skip("pandas-ta not installed")
    
    @pytest.fixture
    def test_data(self):
        """Generate test data as pandas DataFrame."""
        data = {}
        for data_type in DATA_TYPES.keys():
            open_, high, low, close, volume = generate_ohlcv_data(1000, data_type)
            df = pd.DataFrame({
                'open': open_,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
            })
            data[data_type] = (df, close)
        return data
    
    @pytest.mark.parametrize("data_type", list(DATA_TYPES.keys()))
    def test_sma_accuracy(self, test_data, data_type):
        """Test SMA accuracy against pandas-ta."""
        df, close = test_data[data_type]
        
        numta_result = numta.SMA(close, timeperiod=20)
        pandas_ta_result = pandas_ta.sma(df['close'], length=20).values
        
        metrics = compare_results(numta_result, pandas_ta_result)
        
        assert metrics.correlation > 0.999, f"Low correlation: {metrics.correlation}"
    
    @pytest.mark.parametrize("data_type", list(DATA_TYPES.keys()))
    def test_ema_accuracy(self, test_data, data_type):
        """Test EMA accuracy against pandas-ta."""
        df, close = test_data[data_type]
        
        numta_result = numta.EMA(close, timeperiod=20)
        pandas_ta_result = pandas_ta.ema(df['close'], length=20).values
        
        metrics = compare_results(numta_result, pandas_ta_result)
        
        assert metrics.correlation > 0.99, f"Low correlation: {metrics.correlation}"
    
    @pytest.mark.parametrize("data_type", list(DATA_TYPES.keys()))
    def test_rsi_accuracy(self, test_data, data_type):
        """Test RSI accuracy against pandas-ta."""
        df, close = test_data[data_type]
        
        numta_result = numta.RSI(close, timeperiod=14)
        pandas_ta_result = pandas_ta.rsi(df['close'], length=14).values
        
        metrics = compare_results(numta_result, pandas_ta_result)
        
        # pandas-ta may use slightly different calculation
        assert metrics.correlation > 0.95, f"Low correlation: {metrics.correlation}"


# =====================================================================
# Utility Functions for Manual Testing
# =====================================================================

def run_accuracy_report() -> str:
    """
    Generate a comprehensive accuracy report.
    
    Returns
    -------
    str
        Markdown formatted accuracy report
    """
    if not HAS_TALIB:
        return "TA-Lib not installed. Cannot generate accuracy report."
    
    lines = [
        "# numta Accuracy Report",
        "",
        "Comparison against TA-Lib reference implementation.",
        "",
    ]
    
    # Test functions
    functions = [
        ('SMA', lambda c: numta.SMA(c, 20), lambda c: talib.SMA(c, 20)),
        ('EMA', lambda c: numta.EMA(c, 20), lambda c: talib.EMA(c, 20)),
        ('RSI', lambda c: numta.RSI(c, 14), lambda c: talib.RSI(c, 14)),
    ]
    
    for data_type in DATA_TYPES.keys():
        _, _, _, close, _ = generate_ohlcv_data(1000, data_type)
        
        lines.append(f"## {DATA_TYPES[data_type]}")
        lines.append("")
        lines.append("| Function | MAE | RMSE | Correlation | Classification |")
        lines.append("|----------|-----|------|-------------|----------------|")
        
        for name, numta_fn, talib_fn in functions:
            numta_result = numta_fn(close)
            talib_result = talib_fn(close)
            metrics = compare_results(numta_result, talib_result)
            
            lines.append(
                f"| {name} | {metrics.mae:.2e} | {metrics.rmse:.2e} | "
                f"{metrics.correlation:.6f} | {metrics.classification()} |"
            )
        
        lines.append("")
    
    return "\n".join(lines)


if __name__ == '__main__':
    print(run_accuracy_report())
