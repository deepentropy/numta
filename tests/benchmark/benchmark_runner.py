"""
Benchmark runner for numta performance testing.

This module provides infrastructure for benchmarking numta functions
against TA-Lib and pandas-ta implementations.
"""

import time
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Callable, Optional, Any, Tuple
import numpy as np

# Optional imports
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

try:
    import pandas_ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False

import numta


# =====================================================================
# Data Classes for Results
# =====================================================================

@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    iterations: int
    mean_time: float  # seconds
    median_time: float  # seconds
    std_time: float  # seconds
    min_time: float  # seconds
    max_time: float  # seconds
    data_size: int
    ops_per_second: float = field(init=False)
    
    def __post_init__(self):
        """Calculate operations per second."""
        if self.mean_time > 0:
            self.ops_per_second = 1.0 / self.mean_time
        else:
            self.ops_per_second = float('inf')
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ComparisonResult:
    """Result of comparing multiple implementations."""
    function_name: str
    data_size: int
    results: Dict[str, BenchmarkResult] = field(default_factory=dict)
    speedup_vs_baseline: Dict[str, float] = field(default_factory=dict)
    
    def add_result(self, impl_name: str, result: BenchmarkResult):
        """Add a benchmark result for an implementation."""
        self.results[impl_name] = result
    
    def calculate_speedups(self, baseline: str = 'numta'):
        """Calculate speedup ratios vs baseline implementation."""
        if baseline not in self.results:
            return
        
        baseline_time = self.results[baseline].mean_time
        for name, result in self.results.items():
            if result.mean_time > 0:
                self.speedup_vs_baseline[name] = baseline_time / result.mean_time
            else:
                self.speedup_vs_baseline[name] = float('inf')
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'function_name': self.function_name,
            'data_size': self.data_size,
            'results': {k: v.to_dict() for k, v in self.results.items()},
            'speedup_vs_baseline': self.speedup_vs_baseline,
        }


# =====================================================================
# Benchmark Runner
# =====================================================================

class BenchmarkRunner:
    """
    Runner for benchmarking numta functions.
    
    Provides methods for timing functions, comparing implementations,
    and generating reports.
    """
    
    DEFAULT_ITERATIONS = 100
    DEFAULT_WARMUP = 10
    DEFAULT_DATA_SIZES = [1000, 10000, 100000]
    
    def __init__(self, seed: int = 42):
        """
        Initialize benchmark runner.
        
        Parameters
        ----------
        seed : int
            Random seed for data generation
        """
        self.seed = seed
        self.results: List[ComparisonResult] = []
    
    def _generate_ohlcv_data(self, size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate OHLCV data for benchmarking.
        
        Parameters
        ----------
        size : int
            Number of data points
        
        Returns
        -------
        Tuple
            (open, high, low, close, volume) arrays
        """
        np.random.seed(self.seed)
        close = 100 + np.cumsum(np.random.randn(size) * 0.5)
        high = close + np.abs(np.random.randn(size) * 0.5)
        low = close - np.abs(np.random.randn(size) * 0.5)
        open_ = close + np.random.randn(size) * 0.3
        volume = np.random.randint(1000, 10000, size).astype(np.float64)
        return open_, high, low, close, volume
    
    def _time_function(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        iterations: int = DEFAULT_ITERATIONS,
        warmup: int = DEFAULT_WARMUP
    ) -> List[float]:
        """
        Time a function execution.
        
        Parameters
        ----------
        func : Callable
            Function to time
        args : tuple
            Positional arguments
        kwargs : dict
            Keyword arguments
        iterations : int
            Number of timing iterations
        warmup : int
            Number of warmup iterations (not timed)
        
        Returns
        -------
        List[float]
            List of execution times in seconds
        """
        # Warmup
        for _ in range(warmup):
            func(*args, **kwargs)
        
        # Timed runs
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func(*args, **kwargs)
            end = time.perf_counter()
            times.append(end - start)
        
        return times
    
    def benchmark_function(
        self,
        name: str,
        func: Callable,
        args: tuple,
        kwargs: dict,
        data_size: int,
        iterations: int = DEFAULT_ITERATIONS,
        warmup: int = DEFAULT_WARMUP
    ) -> BenchmarkResult:
        """
        Benchmark a single function.
        
        Parameters
        ----------
        name : str
            Name for this benchmark
        func : Callable
            Function to benchmark
        args : tuple
            Positional arguments
        kwargs : dict
            Keyword arguments
        data_size : int
            Size of input data
        iterations : int
            Number of timing iterations
        warmup : int
            Number of warmup iterations
        
        Returns
        -------
        BenchmarkResult
            Benchmark results
        """
        times = self._time_function(func, args, kwargs, iterations, warmup)
        times_array = np.array(times)
        
        return BenchmarkResult(
            name=name,
            iterations=iterations,
            mean_time=float(np.mean(times_array)),
            median_time=float(np.median(times_array)),
            std_time=float(np.std(times_array)),
            min_time=float(np.min(times_array)),
            max_time=float(np.max(times_array)),
            data_size=data_size,
        )
    
    def compare_implementations(
        self,
        func_name: str,
        numta_func: Callable,
        numta_args: tuple,
        numta_kwargs: dict,
        talib_func: Optional[Callable] = None,
        talib_args: Optional[tuple] = None,
        talib_kwargs: Optional[dict] = None,
        pandas_ta_func: Optional[Callable] = None,
        pandas_ta_args: Optional[tuple] = None,
        pandas_ta_kwargs: Optional[dict] = None,
        data_size: int = 10000,
        iterations: int = DEFAULT_ITERATIONS
    ) -> ComparisonResult:
        """
        Compare numta implementation against TA-Lib and/or pandas-ta.
        
        Parameters
        ----------
        func_name : str
            Name of the function being compared
        numta_func, talib_func, pandas_ta_func : Callable
            Functions to compare
        *_args, *_kwargs : tuple, dict
            Arguments for each function
        data_size : int
            Size of input data
        iterations : int
            Number of timing iterations
        
        Returns
        -------
        ComparisonResult
            Comparison results
        """
        comparison = ComparisonResult(
            function_name=func_name,
            data_size=data_size,
        )
        
        # Benchmark numta
        result = self.benchmark_function(
            name='numta',
            func=numta_func,
            args=numta_args,
            kwargs=numta_kwargs,
            data_size=data_size,
            iterations=iterations,
        )
        comparison.add_result('numta', result)
        
        # Benchmark TA-Lib if available
        if HAS_TALIB and talib_func is not None:
            result = self.benchmark_function(
                name='talib',
                func=talib_func,
                args=talib_args or numta_args,
                kwargs=talib_kwargs or {},
                data_size=data_size,
                iterations=iterations,
            )
            comparison.add_result('talib', result)
        
        # Benchmark pandas-ta if available
        if HAS_PANDAS_TA and pandas_ta_func is not None:
            result = self.benchmark_function(
                name='pandas_ta',
                func=pandas_ta_func,
                args=pandas_ta_args or (),
                kwargs=pandas_ta_kwargs or {},
                data_size=data_size,
                iterations=iterations,
            )
            comparison.add_result('pandas_ta', result)
        
        comparison.calculate_speedups(baseline='numta')
        self.results.append(comparison)
        
        return comparison
    
    def run_standard_benchmarks(
        self,
        data_sizes: Optional[List[int]] = None,
        iterations: int = DEFAULT_ITERATIONS
    ) -> List[ComparisonResult]:
        """
        Run standard benchmarks for common indicators.
        
        Parameters
        ----------
        data_sizes : List[int], optional
            Data sizes to test
        iterations : int
            Number of timing iterations
        
        Returns
        -------
        List[ComparisonResult]
            All comparison results
        """
        if data_sizes is None:
            data_sizes = self.DEFAULT_DATA_SIZES
        
        all_results = []
        
        for size in data_sizes:
            open_, high, low, close, volume = self._generate_ohlcv_data(size)
            
            # SMA
            result = self.compare_implementations(
                func_name='SMA',
                numta_func=numta.SMA,
                numta_args=(close,),
                numta_kwargs={'timeperiod': 20},
                talib_func=talib.SMA if HAS_TALIB else None,
                talib_args=(close,) if HAS_TALIB else None,
                talib_kwargs={'timeperiod': 20} if HAS_TALIB else None,
                data_size=size,
                iterations=iterations,
            )
            all_results.append(result)
            
            # EMA
            result = self.compare_implementations(
                func_name='EMA',
                numta_func=numta.EMA,
                numta_args=(close,),
                numta_kwargs={'timeperiod': 20},
                talib_func=talib.EMA if HAS_TALIB else None,
                talib_args=(close,) if HAS_TALIB else None,
                talib_kwargs={'timeperiod': 20} if HAS_TALIB else None,
                data_size=size,
                iterations=iterations,
            )
            all_results.append(result)
            
            # RSI
            result = self.compare_implementations(
                func_name='RSI',
                numta_func=numta.RSI,
                numta_args=(close,),
                numta_kwargs={'timeperiod': 14},
                talib_func=talib.RSI if HAS_TALIB else None,
                talib_args=(close,) if HAS_TALIB else None,
                talib_kwargs={'timeperiod': 14} if HAS_TALIB else None,
                data_size=size,
                iterations=iterations,
            )
            all_results.append(result)
            
            # MACD
            result = self.compare_implementations(
                func_name='MACD',
                numta_func=numta.MACD,
                numta_args=(close,),
                numta_kwargs={},
                talib_func=talib.MACD if HAS_TALIB else None,
                talib_args=(close,) if HAS_TALIB else None,
                talib_kwargs={} if HAS_TALIB else None,
                data_size=size,
                iterations=iterations,
            )
            all_results.append(result)
            
            # ATR
            result = self.compare_implementations(
                func_name='ATR',
                numta_func=numta.ATR,
                numta_args=(high, low, close),
                numta_kwargs={'timeperiod': 14},
                talib_func=talib.ATR if HAS_TALIB else None,
                talib_args=(high, low, close) if HAS_TALIB else None,
                talib_kwargs={'timeperiod': 14} if HAS_TALIB else None,
                data_size=size,
                iterations=iterations,
            )
            all_results.append(result)
        
        return all_results
    
    def generate_report(self, results: Optional[List[ComparisonResult]] = None) -> str:
        """
        Generate a markdown report of benchmark results.
        
        Parameters
        ----------
        results : List[ComparisonResult], optional
            Results to include in report. Uses stored results if not provided.
        
        Returns
        -------
        str
            Markdown formatted report
        """
        if results is None:
            results = self.results
        
        if not results:
            return "No benchmark results available."
        
        lines = [
            "# numta Benchmark Report",
            "",
            "## Summary",
            "",
        ]
        
        # Group by function
        by_function: Dict[str, List[ComparisonResult]] = {}
        for result in results:
            if result.function_name not in by_function:
                by_function[result.function_name] = []
            by_function[result.function_name].append(result)
        
        for func_name, func_results in by_function.items():
            lines.append(f"### {func_name}")
            lines.append("")
            lines.append("| Data Size | numta (μs) | TA-Lib (μs) | Speedup vs TA-Lib |")
            lines.append("|-----------|------------|-------------|-------------------|")
            
            for result in sorted(func_results, key=lambda x: x.data_size):
                numta_time = result.results.get('numta')
                talib_time = result.results.get('talib')
                
                numta_us = numta_time.mean_time * 1e6 if numta_time else 'N/A'
                talib_us = talib_time.mean_time * 1e6 if talib_time else 'N/A'
                
                if numta_time and talib_time:
                    speedup = talib_time.mean_time / numta_time.mean_time
                    speedup_str = f"{speedup:.2f}x"
                else:
                    speedup_str = 'N/A'
                
                if isinstance(numta_us, float):
                    numta_us = f"{numta_us:.2f}"
                if isinstance(talib_us, float):
                    talib_us = f"{talib_us:.2f}"
                
                lines.append(f"| {result.data_size:,} | {numta_us} | {talib_us} | {speedup_str} |")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def save_results(self, filepath: str, results: Optional[List[ComparisonResult]] = None):
        """
        Save benchmark results to JSON file.
        
        Parameters
        ----------
        filepath : str
            Path to save results
        results : List[ComparisonResult], optional
            Results to save. Uses stored results if not provided.
        """
        if results is None:
            results = self.results
        
        data = {
            'version': '1.0',
            'seed': self.seed,
            'results': [r.to_dict() for r in results],
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_results(self, filepath: str) -> List[ComparisonResult]:
        """
        Load benchmark results from JSON file.
        
        Parameters
        ----------
        filepath : str
            Path to load results from
        
        Returns
        -------
        List[ComparisonResult]
            Loaded results
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        results = []
        for r in data.get('results', []):
            comparison = ComparisonResult(
                function_name=r['function_name'],
                data_size=r['data_size'],
            )
            for name, result_data in r.get('results', {}).items():
                benchmark_result = BenchmarkResult(
                    name=result_data['name'],
                    iterations=result_data['iterations'],
                    mean_time=result_data['mean_time'],
                    median_time=result_data['median_time'],
                    std_time=result_data['std_time'],
                    min_time=result_data['min_time'],
                    max_time=result_data['max_time'],
                    data_size=result_data['data_size'],
                )
                comparison.add_result(name, benchmark_result)
            comparison.speedup_vs_baseline = r.get('speedup_vs_baseline', {})
            results.append(comparison)
        
        return results


# =====================================================================
# Convenience Functions
# =====================================================================

def run_quick_benchmark() -> str:
    """
    Run a quick benchmark and return markdown report.
    
    Returns
    -------
    str
        Markdown formatted report
    """
    runner = BenchmarkRunner()
    runner.run_standard_benchmarks(
        data_sizes=[1000, 10000],
        iterations=50,
    )
    return runner.generate_report()


if __name__ == '__main__':
    print(run_quick_benchmark())
