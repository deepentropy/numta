"""
Performance measurement utilities for comparing talib-pure with TA-Lib
"""

import time
import numpy as np
from typing import Callable, Dict, List, Any, Optional, Tuple
from statistics import mean, stdev


class PerformanceMeasurement:
    """
    General class to measure and compare the speed of indicator functions.

    This class allows you to benchmark any function (from talib-pure or TA-Lib)
    and compare their performance across multiple runs and data sizes.

    Examples
    --------
    >>> import numpy as np
    >>> from talib_pure.benchmark import PerformanceMeasurement
    >>> from talib_pure import SMA as SMA_pure
    >>> import talib
    >>>
    >>> # Create test data
    >>> data = np.random.uniform(100, 200, 1000)
    >>>
    >>> # Create benchmark
    >>> bench = PerformanceMeasurement()
    >>>
    >>> # Add functions to compare
    >>> bench.add_function("talib-pure SMA", SMA_pure, data, timeperiod=30)
    >>> bench.add_function("TA-Lib SMA", talib.SMA, data, timeperiod=30)
    >>>
    >>> # Run benchmark
    >>> results = bench.run(iterations=1000)
    >>> bench.print_results(results)
    """

    def __init__(self):
        """Initialize the performance measurement class"""
        self.functions: List[Dict[str, Any]] = []

    def add_function(
        self,
        name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> None:
        """
        Add a function to benchmark

        Parameters
        ----------
        name : str
            Display name for the function
        func : Callable
            The function to benchmark
        *args : Any
            Positional arguments to pass to the function
        **kwargs : Any
            Keyword arguments to pass to the function
        """
        self.functions.append({
            'name': name,
            'func': func,
            'args': args,
            'kwargs': kwargs
        })

    def clear(self) -> None:
        """Clear all registered functions"""
        self.functions = []

    def measure_single_run(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> float:
        """
        Measure a single execution time of a function

        Parameters
        ----------
        func : Callable
            Function to measure
        *args : Any
            Positional arguments
        **kwargs : Any
            Keyword arguments

        Returns
        -------
        float
            Execution time in seconds
        """
        start_time = time.perf_counter()
        func(*args, **kwargs)
        end_time = time.perf_counter()
        return end_time - start_time

    def run(
        self,
        iterations: int = 100,
        warmup: int = 10
    ) -> Dict[str, Dict[str, float]]:
        """
        Run benchmark for all registered functions

        Parameters
        ----------
        iterations : int, optional
            Number of iterations to run (default: 100)
        warmup : int, optional
            Number of warmup iterations (default: 10)

        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary with benchmark results for each function
            Contains: mean, median, stdev, min, max times and speedup
        """
        if not self.functions:
            raise ValueError("No functions registered. Use add_function() first.")

        results = {}

        # Run benchmarks
        for func_info in self.functions:
            name = func_info['name']
            func = func_info['func']
            args = func_info['args']
            kwargs = func_info['kwargs']

            # Warmup
            for _ in range(warmup):
                func(*args, **kwargs)

            # Measure
            times = []
            for _ in range(iterations):
                exec_time = self.measure_single_run(func, *args, **kwargs)
                times.append(exec_time)

            # Calculate statistics
            results[name] = {
                'mean': mean(times),
                'median': sorted(times)[len(times) // 2],
                'stdev': stdev(times) if len(times) > 1 else 0.0,
                'min': min(times),
                'max': max(times),
                'iterations': iterations,
                'times': times
            }

        # Calculate speedup relative to first function
        if len(results) > 1:
            baseline_name = list(results.keys())[0]
            baseline_mean = results[baseline_name]['mean']

            for name in results:
                if name == baseline_name:
                    results[name]['speedup'] = 1.0
                else:
                    results[name]['speedup'] = baseline_mean / results[name]['mean']

        return results

    def print_results(
        self,
        results: Dict[str, Dict[str, float]],
        precision: int = 6
    ) -> None:
        """
        Print benchmark results in a readable format

        Parameters
        ----------
        results : Dict[str, Dict[str, float]]
            Results from run() method
        precision : int, optional
            Number of decimal places for time display (default: 6)
        """
        print("\n" + "=" * 80)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("=" * 80)

        for name, stats in results.items():
            print(f"\n{name}:")
            print(f"  Mean:       {stats['mean']:.{precision}f} seconds")
            print(f"  Median:     {stats['median']:.{precision}f} seconds")
            print(f"  Std Dev:    {stats['stdev']:.{precision}f} seconds")
            print(f"  Min:        {stats['min']:.{precision}f} seconds")
            print(f"  Max:        {stats['max']:.{precision}f} seconds")
            print(f"  Iterations: {stats['iterations']}")

            if 'speedup' in stats:
                speedup = stats['speedup']
                if speedup > 1:
                    print(f"  Speedup:    {speedup:.2f}x faster than baseline")
                elif speedup < 1:
                    print(f"  Speedup:    {1/speedup:.2f}x slower than baseline")
                else:
                    print(f"  Speedup:    baseline (1.0x)")

        print("\n" + "=" * 80)

    def compare_with_data_sizes(
        self,
        func_pairs: List[Tuple[str, Callable, Dict[str, Any]]],
        data_sizes: List[int],
        iterations: int = 100,
        data_generator: Optional[Callable] = None
    ) -> Dict[int, Dict[str, Dict[str, float]]]:
        """
        Compare functions across different data sizes

        Parameters
        ----------
        func_pairs : List[Tuple[str, Callable, Dict[str, Any]]]
            List of (name, function, kwargs) tuples
        data_sizes : List[int]
            List of data sizes to test
        iterations : int, optional
            Number of iterations per size (default: 100)
        data_generator : Callable, optional
            Function to generate test data, receives size as argument
            Default: np.random.uniform(100, 200, size)

        Returns
        -------
        Dict[int, Dict[str, Dict[str, float]]]
            Results organized by data size
        """
        if data_generator is None:
            data_generator = lambda size: np.random.uniform(100, 200, size)

        all_results = {}

        for size in data_sizes:
            print(f"\nBenchmarking with data size: {size}")
            data = data_generator(size)

            self.clear()
            for name, func, kwargs in func_pairs:
                self.add_function(name, func, data, **kwargs)

            results = self.run(iterations=iterations)
            all_results[size] = results

        return all_results

    def print_comparison_table(
        self,
        results: Dict[int, Dict[str, Dict[str, float]]]
    ) -> None:
        """
        Print a comparison table for results across different data sizes

        Parameters
        ----------
        results : Dict[int, Dict[str, Dict[str, float]]]
            Results from compare_with_data_sizes()
        """
        print("\n" + "=" * 100)
        print("PERFORMANCE COMPARISON ACROSS DATA SIZES")
        print("=" * 100)

        # Get function names
        first_size = next(iter(results.keys()))
        func_names = list(results[first_size].keys())

        # Header
        print(f"\n{'Data Size':<12}", end="")
        for name in func_names:
            print(f"{name:<25}", end="")
        print()
        print("-" * 100)

        # Data rows
        for size, size_results in results.items():
            print(f"{size:<12}", end="")
            for name in func_names:
                mean_time = size_results[name]['mean']
                if 'speedup' in size_results[name]:
                    speedup = size_results[name]['speedup']
                    print(f"{mean_time:.6f}s ({speedup:.2f}x)  ", end="")
                else:
                    print(f"{mean_time:.6f}s          ", end="")
            print()

        print("=" * 100)
