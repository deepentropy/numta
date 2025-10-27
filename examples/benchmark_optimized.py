"""
Comprehensive benchmark comparing all SMA implementations:
- TA-Lib (C implementation)
- talib-pure original (np.convolve)
- talib-pure cumsum (O(n) algorithm)
- talib-pure Numba (JIT compiled)
"""

import numpy as np
import sys
from talib_pure.benchmark import PerformanceMeasurement
from talib_pure.optimized import (
    SMA_cumsum, SMA_auto, get_available_backends,
)

if HAS_NUMBA:
    from talib_pure.optimized import SMA_numba



def print_backend_status():
    """Print available optimization backends"""
    print("\n" + "=" * 80)
    print("AVAILABLE OPTIMIZATION BACKENDS")
    print("=" * 80)

    backends = get_available_backends()
    for name, info in backends.items():
        status = "✓ Available" if info["available"] else "✗ Not installed"
        print(f"{name:10s} {status:15s} - {info['description']}")

    print("=" * 80)


def benchmark_implementations():
    """Benchmark all available implementations"""
    from talib_pure import SMA as SMA_original

    # Try to import TA-Lib
    try:
        import talib
        has_talib = True
    except ImportError:
        has_talib = False
        print("\nWarning: TA-Lib not installed. Skipping TA-Lib comparison.")

    print_backend_status()

    bench = PerformanceMeasurement()

    # Test 1: Medium dataset (10k points) - typical use case
    print("\n" + "=" * 80)
    print("TEST 1: Medium Dataset (10,000 points, timeperiod=30)")
    print("=" * 80)

    data = np.random.uniform(100, 200, 10000)
    timeperiod = 30

    bench.clear()
    if has_talib:
        bench.add_function("TA-Lib (C)", talib.SMA, data, timeperiod=timeperiod)
    bench.add_function("talib-pure (convolve)", SMA_original, data, timeperiod=timeperiod)
    bench.add_function("talib-pure (cumsum)", SMA_cumsum, data, timeperiod=timeperiod)

    if HAS_NUMBA:
        bench.add_function("talib-pure (Numba)", SMA_numba, data, timeperiod=timeperiod)


    results = bench.run(iterations=1000, warmup=50)
    bench.print_results(results)

    # Test 2: Large dataset (100k points)
    print("\n" + "=" * 80)
    print("TEST 2: Large Dataset (100,000 points, timeperiod=30)")
    print("=" * 80)

    data_large = np.random.uniform(100, 200, 100000)

    bench.clear()
    if has_talib:
        bench.add_function("TA-Lib (C)", talib.SMA, data_large, timeperiod=timeperiod)
    bench.add_function("talib-pure (convolve)", SMA_original, data_large, timeperiod=timeperiod)
    bench.add_function("talib-pure (cumsum)", SMA_cumsum, data_large, timeperiod=timeperiod)

    if HAS_NUMBA:
        bench.add_function("talib-pure (Numba)", SMA_numba, data_large, timeperiod=timeperiod)


    results = bench.run(iterations=500, warmup=20)
    bench.print_results(results)

    # Test 3: Very large dataset (1M points)
    print("\n" + "=" * 80)
    print("TEST 3: Very Large Dataset (1,000,000 points, timeperiod=30)")
    print("=" * 80)

    data_xlarge = np.random.uniform(100, 200, 1000000)

    bench.clear()
    if has_talib:
        bench.add_function("TA-Lib (C)", talib.SMA, data_xlarge, timeperiod=timeperiod)
    bench.add_function("talib-pure (convolve)", SMA_original, data_xlarge, timeperiod=timeperiod)
    bench.add_function("talib-pure (cumsum)", SMA_cumsum, data_xlarge, timeperiod=timeperiod)

    if HAS_NUMBA:
        bench.add_function("talib-pure (Numba)", SMA_numba, data_xlarge, timeperiod=timeperiod)


    results = bench.run(iterations=100, warmup=10)
    bench.print_results(results)

    # Test 4: Different timeperiods (10k points)
    print("\n" + "=" * 80)
    print("TEST 4: Different Timeperiods (10,000 points)")
    print("=" * 80)

    data = np.random.uniform(100, 200, 10000)

    for period in [5, 20, 50, 200]:
        print(f"\nTimeperiod: {period}")
        bench.clear()

        if has_talib:
            bench.add_function("TA-Lib", talib.SMA, data, timeperiod=period)
        bench.add_function("cumsum", SMA_cumsum, data, timeperiod=period)

        if HAS_NUMBA:
            bench.add_function("Numba", SMA_numba, data, timeperiod=period)

        results = bench.run(iterations=500, warmup=20)

        # Compact output
        for name, stats in results.items():
            speedup_str = ""
            if 'speedup' in stats:
                speedup = stats['speedup']
                if speedup > 1:
                    speedup_str = f" ({speedup:.2f}x faster)"
                elif speedup < 1:
                    speedup_str = f" ({1/speedup:.2f}x slower)"
            print(f"  {name:20s}: {stats['mean']*1000:.3f}ms{speedup_str}")

    # Test 5: Scaling comparison
    if has_talib:
        print("\n" + "=" * 80)
        print("TEST 5: Performance Scaling Across Data Sizes")
        print("=" * 80)

        func_pairs = [
            ("TA-Lib", talib.SMA, {"timeperiod": 30}),
            ("cumsum", SMA_cumsum, {"timeperiod": 30}),
        ]

        if HAS_NUMBA:
            func_pairs.append(("Numba", SMA_numba, {"timeperiod": 30}))

        data_sizes = [100, 1000, 10000, 100000]
        results = bench.compare_with_data_sizes(func_pairs, data_sizes, iterations=100)
        bench.print_comparison_table(results)

    # Summary and recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    print("\nFor best performance:")
    print("1. Small datasets (<1k):     Use cumsum or auto")
    print("2. Medium datasets (1k-100k): Use Numba if available, else cumsum")
    print("4. Production use:            Use SMA_auto(backend='auto') for automatic selection")

    print("\nInstallation commands:")
    if not HAS_NUMBA:
        print("  pip install numba              # For Numba optimization")

    print("\nExample usage:")
    print("  from talib_pure.optimized import SMA_auto")
    print("  sma = SMA_auto(close, timeperiod=30, backend='auto')  # Auto-select best")
    print("  sma = SMA_auto(close, timeperiod=30, backend='numba') # Force Numba")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    benchmark_implementations()
