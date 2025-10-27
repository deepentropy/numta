"""
Example: Benchmarking SMA performance between numta and TA-Lib

This script demonstrates how to use the PerformanceMeasurement class
to compare the performance of numta against the original TA-Lib.
"""

import numpy as np


def main():
    # Import numta
    from numta import SMA as SMA_numta
    from numta.benchmark import PerformanceMeasurement

    # Try to import TA-Lib
    try:
        import talib
        has_talib = True
    except ImportError:
        has_talib = False
        print("Warning: TA-Lib not installed. Only benchmarking numta.")
        print("To install TA-Lib: pip install TA-Lib")

    # Create benchmark instance
    bench = PerformanceMeasurement()

    print("\n" + "=" * 80)
    print("SMA PERFORMANCE BENCHMARK")
    print("=" * 80)

    # Test 1: Single data size comparison
    print("\n### Test 1: Single Data Size (10,000 points) ###")
    data_size = 10000
    data = np.random.uniform(100, 200, data_size)
    timeperiod = 30

    bench.clear()
    bench.add_function("numta SMA", SMA_numta, data, timeperiod=timeperiod)
    if has_talib:
        bench.add_function("TA-Lib SMA", talib.SMA, data, timeperiod=timeperiod)

    results = bench.run(iterations=1000, warmup=10)
    bench.print_results(results)

    # Test 2: Multiple data sizes comparison
    if has_talib:
        print("\n### Test 2: Performance Across Different Data Sizes ###")
        func_pairs = [
            ("numta", SMA_numta, {"timeperiod": 30}),
            ("TA-Lib", talib.SMA, {"timeperiod": 30}),
        ]

        data_sizes = [100, 1000, 10000, 100000]
        size_results = bench.compare_with_data_sizes(
            func_pairs,
            data_sizes,
            iterations=100
        )

        bench.print_comparison_table(size_results)

    # Test 3: Different timeperiods
    print("\n### Test 3: Different Timeperiods (10,000 points) ###")
    data = np.random.uniform(100, 200, 10000)

    for period in [5, 20, 50, 200]:
        print(f"\nTimeperiod: {period}")
        bench.clear()
        bench.add_function("numta", SMA_numta, data, timeperiod=period)
        if has_talib:
            bench.add_function("TA-Lib", talib.SMA, data, timeperiod=period)

        results = bench.run(iterations=500, warmup=10)

        # Print compact results
        for name, stats in results.items():
            speedup_str = ""
            if 'speedup' in stats:
                speedup_str = f" (speedup: {stats['speedup']:.2f}x)"
            print(f"  {name}: {stats['mean']:.6f}s{speedup_str}")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
