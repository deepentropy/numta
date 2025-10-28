"""
Benchmark comparison for Statistic Functions: talib-pure vs TA-Lib

This script compares the performance of Statistic Functions implementations
between talib-pure (Numba/CPU) and the original TA-Lib library.
"""

import numpy as np
import talib
import time
from numta import CORREL, LINEARREG, LINEARREG_ANGLE, LINEARREG_INTERCEPT, LINEARREG_SLOPE, STDDEV, TSF, VAR


def generate_test_data(size):
    """Generate random price data for testing"""
    np.random.seed(42)
    close = np.random.randn(size).cumsum() + 100
    high = close + np.random.rand(size) * 2
    low = close - np.random.rand(size) * 2
    return high, low, close


def benchmark_function(func_talib, func_pure, name, talib_args, pure_args, iterations=100):
    """
    Benchmark a single function from both libraries

    Args:
        func_talib: TA-Lib function
        func_pure: talib-pure function
        name: Name of the function for display
        talib_args: Tuple of arguments for TA-Lib function
        pure_args: Tuple of arguments for talib-pure function
        iterations: Number of iterations for timing
    """
    # Warm up
    try:
        _ = func_talib(*talib_args)
    except Exception as e:
        print(f"  {name}: TA-Lib warm-up failed: {e}")
        return None

    try:
        _ = func_pure(*pure_args)
    except Exception as e:
        print(f"  {name}: talib-pure warm-up failed: {e}")
        return None

    # Benchmark TA-Lib
    start = time.perf_counter()
    for _ in range(iterations):
        result_talib = func_talib(*talib_args)
    end = time.perf_counter()
    time_talib = (end - start) / iterations

    # Benchmark talib-pure
    start = time.perf_counter()
    for _ in range(iterations):
        result_pure = func_pure(*pure_args)
    end = time.perf_counter()
    time_pure = (end - start) / iterations

    # Calculate speedup (>1 means talib-pure is faster)
    speedup = time_talib / time_pure if time_pure > 0 else float('inf')

    return {
        'name': name,
        'talib_time': time_talib * 1000,  # Convert to ms
        'pure_time': time_pure * 1000,     # Convert to ms
        'speedup': speedup
    }


def run_benchmarks(size, iterations=100):
    """Run all benchmarks for a given data size"""
    print(f"\n{'='*70}")
    print(f"Dataset size: {size:,} bars")
    print(f"Iterations: {iterations}")
    print(f"{'='*70}\n")

    high, low, close = generate_test_data(size)
    timeperiod = 30
    timeperiod_lr = 14  # Linear regression functions use 14 by default

    # Define all indicators to test
    # Format: (name, talib_func, pure_func, talib_args, pure_args)
    indicators = [
        ('CORREL', talib.CORREL, CORREL, (high, low, timeperiod), (high, low, timeperiod)),
        ('LINEARREG', talib.LINEARREG, LINEARREG, (close, timeperiod_lr), (close, timeperiod_lr)),
        ('LINEARREG_ANGLE', talib.LINEARREG_ANGLE, LINEARREG_ANGLE, (close, timeperiod_lr), (close, timeperiod_lr)),
        ('LINEARREG_INTERCEPT', talib.LINEARREG_INTERCEPT, LINEARREG_INTERCEPT, (close, timeperiod_lr), (close, timeperiod_lr)),
        ('LINEARREG_SLOPE', talib.LINEARREG_SLOPE, LINEARREG_SLOPE, (close, timeperiod_lr), (close, timeperiod_lr)),
        ('STDDEV', talib.STDDEV, STDDEV, (close, timeperiod), (close, timeperiod)),
        ('TSF', talib.TSF, TSF, (close, timeperiod_lr), (close, timeperiod_lr)),
        ('VAR', talib.VAR, VAR, (close, timeperiod), (close, timeperiod)),
    ]

    results = []
    for name, func_talib, func_pure, talib_args, pure_args in indicators:
        result = benchmark_function(func_talib, func_pure, name, talib_args, pure_args, iterations)
        if result:
            results.append(result)
            print(f"{name:20s} | TA-Lib: {result['talib_time']:8.4f} ms | "
                  f"talib-pure: {result['pure_time']:8.4f} ms | "
                  f"Speedup: {result['speedup']:5.2f}x")

    return results


def print_summary(all_results):
    """Print summary table of all results"""
    print(f"\n{'='*70}")
    print("SUMMARY - Average Performance Across All Dataset Sizes")
    print(f"{'='*70}\n")

    # Calculate averages for each function across all sizes
    func_stats = {}
    for results in all_results:
        for result in results:
            name = result['name']
            if name not in func_stats:
                func_stats[name] = {'speedups': []}
            func_stats[name]['speedups'].append(result['speedup'])

    # Print summary table
    print(f"{'Function':<20s} | {'Avg Speedup':>12s} | Status")
    print(f"{'-'*20}-+-{'-'*12}-+{'-'*30}")

    for name in sorted(func_stats.keys()):
        avg_speedup = np.mean(func_stats[name]['speedups'])
        if avg_speedup >= 1.5:
            status = "✅ Much Faster"
        elif avg_speedup >= 1.0:
            status = "✅ Faster"
        elif avg_speedup >= 0.8:
            status = "⚠️  Competitive"
        else:
            status = "❌ Slower"

        print(f"{name:<20s} | {avg_speedup:11.2f}x | {status}")


def main():
    """Main benchmark execution"""
    print("="*70)
    print("Statistic Functions Performance Benchmark")
    print("Comparing talib-pure (Numba/CPU) vs TA-Lib")
    print("="*70)

    # Test with different dataset sizes
    sizes = [1000, 10000, 100000]
    all_results = []

    for size in sizes:
        # Use fewer iterations for larger datasets
        iterations = 100 if size <= 10000 else 10
        results = run_benchmarks(size, iterations)
        all_results.append(results)

    # Print overall summary
    print_summary(all_results)

    print(f"\n{'='*70}")
    print("Benchmark complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
