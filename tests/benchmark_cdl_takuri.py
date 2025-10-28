"""
Benchmark for CDLTAKURI
"""

import numpy as np
import time


def benchmark_cdltakuri():
    """Benchmark CDLTAKURI"""
    from talib_pure import CDLTAKURI

    print("=" * 80)
    print("CDLTAKURI Benchmark")
    print("=" * 80)

    sizes = [1000, 10000, 100000]

    for n in sizes:
        print(f"\n{'=' * 80}")
        print(f"Dataset size: {n:,} bars")
        print('=' * 80)

        # Generate test data
        np.random.seed(42)
        high = np.random.uniform(100, 200, n)
        low = high - np.random.uniform(1, 10, n)
        open_ = low + np.random.uniform(0, 1, n) * (high - low)
        close = low + np.random.uniform(0, 1, n) * (high - low)

        iterations = 100 if n <= 10000 else 10

        _ = CDLTAKURI(open_, high, low, close)
        start = time.perf_counter()
        for _ in range(iterations):
            result = CDLTAKURI(open_, high, low, close)
        end = time.perf_counter()
        time_taken = (end - start) / iterations
        print(f"  Time: {time_taken * 1000:8.4f}ms ({time_taken * 1e6:10.2f}us)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    benchmark_cdltakuri()
