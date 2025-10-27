"""
Performance benchmark comparison between talib-pure (Numba/CPU) and original TA-Lib
for Price Transform Indicators
"""

import numpy as np
import time
import talib
from talib_pure import (
    MEDPRICE, MIDPOINT, MIDPRICE, TYPPRICE, WCLPRICE
)


def benchmark_function(func_talib, func_pure, name, talib_args, pure_args, iterations=100):
    """Benchmark a function against both implementations"""

    # Warm up
    _ = func_talib(*talib_args)
    _ = func_pure(*pure_args)

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
    print("Price Transform Indicators Performance Comparison")
    print("talib-pure (Numba/CPU) vs Original TA-Lib")
    print("=" * 80)
    print()

    # Test configurations
    sizes = [1000, 10000, 100000]
    timeperiod = 14

    results = {}

    for size in sizes:
        print(f"\nDataset size: {size:,} bars (timeperiod={timeperiod} where applicable)")
        print("-" * 80)

        # Generate test data
        np.random.seed(42)
        close = np.random.randn(size) * 10 + 100
        high = close + np.abs(np.random.randn(size) * 2)
        low = close - np.abs(np.random.randn(size) * 2)

        iterations = 100 if size <= 10000 else 10

        results[size] = []

        # Price Transform indicators
        indicators = [
            ('MEDPRICE', talib.MEDPRICE, MEDPRICE, (high, low), (high, low)),
            ('TYPPRICE', talib.TYPPRICE, TYPPRICE, (high, low, close), (high, low, close)),
            ('WCLPRICE', talib.WCLPRICE, WCLPRICE, (high, low, close), (high, low, close)),
            ('MIDPOINT', talib.MIDPOINT, MIDPOINT, (close, timeperiod), (close, timeperiod)),
            ('MIDPRICE', talib.MIDPRICE, MIDPRICE, (high, low, timeperiod), (high, low, timeperiod)),
        ]

        for name, func_talib, func_pure, talib_args, pure_args in indicators:
            result = benchmark_function(func_talib, func_pure, name, talib_args, pure_args, iterations)
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

    for i in range(len(results[1000])):
        name = results[1000][i]['name']
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
