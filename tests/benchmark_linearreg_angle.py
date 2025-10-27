"""
Benchmark for LINEARREG_ANGLE
"""

import numpy as np
import time


def benchmark_linearreg_angle():
    """Benchmark LINEARREG_ANGLE"""
    from numta import LINEARREG_ANGLE

    print("=" * 80)
    print("LINEARREG_ANGLE Benchmark")
    print("=" * 80)

    sizes = [1000, 10000, 100000]

    for n in sizes:
        print(f"\n{'=' * 80}")
        print(f"Dataset size: {n:,} bars")
        print('=' * 80)

        # Generate test data
        np.random.seed(42)
        close = np.random.randn(n) * 10 + 100

        iterations = 100 if n <= 10000 else 10

        _ = LINEARREG_ANGLE(close)
        start = time.perf_counter()
        for _ in range(iterations):
            result = LINEARREG_ANGLE(close)
        end = time.perf_counter()
        time_taken = (end - start) / iterations
        print(f"  Time: {time_taken * 1000:8.4f}ms ({time_taken * 1e6:10.2f}us)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    benchmark_linearreg_angle()
