"""
Benchmark comparison between talib-pure (Numba/CPU) and original TA-Lib
for Cycle Indicators
"""

import numpy as np
import time
import talib
from talib_pure import (
    HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, HT_SINE, HT_TRENDLINE, HT_TRENDMODE
)


def benchmark_function(func_talib, func_pure, name, close_data, iterations=100):
    """Benchmark a function against both implementations"""

    # Warm up
    _ = func_talib(close_data)
    _ = func_pure(close_data)

    # Benchmark TA-Lib
    start = time.perf_counter()
    for _ in range(iterations):
        result_talib = func_talib(close_data)
    end = time.perf_counter()
    time_talib = (end - start) / iterations

    # Benchmark talib-pure
    start = time.perf_counter()
    for _ in range(iterations):
        result_pure = func_pure(close_data)
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
    print("Cycle Indicators Performance Comparison")
    print("talib-pure (Numba/CPU) vs Original TA-Lib")
    print("=" * 80)
    print()

    # Test configurations
    sizes = [1000, 10000, 100000]

    # Cycle indicator functions
    indicators = [
        ('HT_DCPERIOD', talib.HT_DCPERIOD, HT_DCPERIOD),
        ('HT_DCPHASE', talib.HT_DCPHASE, HT_DCPHASE),
        ('HT_PHASOR', talib.HT_PHASOR, HT_PHASOR),
        ('HT_SINE', talib.HT_SINE, HT_SINE),
        ('HT_TRENDLINE', talib.HT_TRENDLINE, HT_TRENDLINE),
        ('HT_TRENDMODE', talib.HT_TRENDMODE, HT_TRENDMODE),
    ]

    results = {}

    for size in sizes:
        print(f"\nDataset size: {size:,} bars")
        print("-" * 80)

        # Generate test data
        np.random.seed(42)
        close = np.random.randn(size) * 10 + 100

        iterations = 100 if size <= 10000 else 10

        results[size] = []

        for name, func_talib, func_pure in indicators:
            result = benchmark_function(func_talib, func_pure, name, close, iterations)
            results[size].append(result)

            print(f"  {name:15} | TA-Lib: {result['talib_ms']:8.4f}ms | "
                  f"talib-pure: {result['pure_ms']:8.4f}ms | "
                  f"Speedup: {result['speedup']:5.2f}x")

    print("\n" + "=" * 80)
    print("\nSummary Table (for PERFORMANCE.md):")
    print("=" * 80)
    print()

    # Print markdown table
    print("| Function | 1K bars | 10K bars | 100K bars |")
    print("|----------|---------|----------|-----------|")

    for i, (name, _, _) in enumerate(indicators):
        row = f"| {name}"
        for size in sizes:
            result = results[size][i]
            row += f" | {result['speedup']:.2f}x"
        row += " |"
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
