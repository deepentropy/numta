"""
Performance benchmark comparison between talib-pure (Numba/CPU) and original TA-Lib
for Math Operators
"""

import numpy as np
import time
import talib
from talib_pure import (
    MAX, MIN, MINMAX, MAXINDEX, MININDEX, MINMAXINDEX, SUM
)


def benchmark_function(func_talib, func_pure, name, data, timeperiod=30, iterations=100):
    """Benchmark a function against both implementations"""

    # Warm up
    _ = func_talib(data, timeperiod=timeperiod)
    _ = func_pure(data, timeperiod=timeperiod)

    # Benchmark TA-Lib
    start = time.perf_counter()
    for _ in range(iterations):
        result_talib = func_talib(data, timeperiod=timeperiod)
    end = time.perf_counter()
    time_talib = (end - start) / iterations

    # Benchmark talib-pure
    start = time.perf_counter()
    for _ in range(iterations):
        result_pure = func_pure(data, timeperiod=timeperiod)
    end = time.perf_counter()
    time_pure = (end - start) / iterations

    speedup = time_talib / time_pure if time_pure > 0 else 0

    return {
        'name': name,
        'talib_ms': time_talib * 1000,
        'pure_ms': time_pure * 1000,
        'speedup': speedup
    }


def main():
    """Run all benchmarks"""

    print("=" * 80)
    print("Math Operators Performance Comparison")
    print("talib-pure (Numba/CPU) vs Original TA-Lib")
    print("=" * 80)
    print()

    # Test configurations
    sizes = [1000, 10000, 100000]
    timeperiod = 30

    # Math operator functions
    operators = [
        ('MAX', talib.MAX, MAX),
        ('MIN', talib.MIN, MIN),
        ('MINMAX', talib.MINMAX, MINMAX),
        ('MAXINDEX', talib.MAXINDEX, MAXINDEX),
        ('MININDEX', talib.MININDEX, MININDEX),
        ('MINMAXINDEX', talib.MINMAXINDEX, MINMAXINDEX),
        ('SUM', talib.SUM, SUM),
    ]

    results = {}

    for size in sizes:
        print(f"\nDataset size: {size:,} bars (timeperiod={timeperiod})")
        print("-" * 80)

        # Generate test data
        np.random.seed(42)
        data = np.random.randn(size) * 10 + 100

        iterations = 100 if size <= 10000 else 10

        results[size] = []

        for name, func_talib, func_pure in operators:
            result = benchmark_function(func_talib, func_pure, name, data, timeperiod, iterations)
            results[size].append(result)

            print(f"  {name:15} | TA-Lib: {result['talib_ms']:8.4f}ms | "
                  f"talib-pure: {result['pure_ms']:8.4f}ms | "
                  f"Speedup: {result['speedup']:5.2f}x")

    print("\n" + "=" * 80)
    print("\nSummary Table (for PERFORMANCE.md):")
    print("=" * 80)
    print()

    # Print markdown table
    print("| Function | 1K bars | 10K bars | 100K bars | Avg Speedup |")
    print("|----------|---------|----------|-----------|-------------|")

    for i, (name, _, _) in enumerate(operators):
        row = f"| {name}"
        speedups = []
        for size in sizes:
            result = results[size][i]
            row += f" | {result['speedup']:.2f}x"
            speedups.append(result['speedup'])
        avg_speedup = np.mean(speedups)
        row += f" | {avg_speedup:.2f}x |"
        print(row)

    print()
    print("Times in milliseconds:")
    print()

    for size in sizes:
        print(f"\n### {size:,} bars")
        print()
        print("| Function | TA-Lib (ms) | talib-pure (ms) | Speedup |")
        print("|----------|-------------|-----------------|---------|")
        for result in results[size]:
            print(f"| {result['name']:15} | {result['talib_ms']:11.4f} | "
                  f"{result['pure_ms']:15.4f} | {result['speedup']:7.2f}x |")


if __name__ == "__main__":
    main()
