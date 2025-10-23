"""
Benchmark comparing different SMA implementations:
1. Current: Rolling window with Numba
2. Cumsum: Using np.cumsum with Numba
3. TA-Lib: Reference C implementation
"""

import numpy as np
import time
from numba import jit


@jit(nopython=True, cache=True)
def _sma_rolling_numba(close: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Current implementation: Rolling window approach
    """
    n = len(close)

    # Fill lookback period with NaN
    for i in range(timeperiod - 1):
        output[i] = np.nan

    # Calculate first SMA value
    sum_val = 0.0
    for i in range(timeperiod):
        sum_val += close[i]
    output[timeperiod - 1] = sum_val / timeperiod

    # Use rolling window for subsequent values
    for i in range(timeperiod, n):
        sum_val = sum_val - close[i - timeperiod] + close[i]
        output[i] = sum_val / timeperiod


@jit(nopython=True, cache=True)
def _sma_cumsum_numba(close: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Alternative implementation: Using cumsum approach

    Formula:
    SMA[i] = (cumsum[i] - cumsum[i - timeperiod]) / timeperiod
    """
    n = len(close)

    # Fill lookback period with NaN
    for i in range(timeperiod - 1):
        output[i] = np.nan

    # Calculate cumulative sum
    cumsum = np.empty(n + 1, dtype=np.float64)
    cumsum[0] = 0.0
    for i in range(n):
        cumsum[i + 1] = cumsum[i] + close[i]

    # Calculate SMA using cumsum
    for i in range(timeperiod - 1, n):
        output[i] = (cumsum[i + 1] - cumsum[i - timeperiod + 1]) / timeperiod


def sma_rolling(close: np.ndarray, timeperiod: int = 30) -> np.ndarray:
    """SMA using rolling window approach"""
    if timeperiod < 2:
        raise ValueError("timeperiod must be >= 2")

    close = np.asarray(close, dtype=np.float64)
    n = len(close)

    if n == 0:
        return np.array([], dtype=np.float64)
    if n < timeperiod:
        return np.full(n, np.nan, dtype=np.float64)

    output = np.empty(n, dtype=np.float64)
    _sma_rolling_numba(close, timeperiod, output)
    return output


def sma_cumsum(close: np.ndarray, timeperiod: int = 30) -> np.ndarray:
    """SMA using cumsum approach"""
    if timeperiod < 2:
        raise ValueError("timeperiod must be >= 2")

    close = np.asarray(close, dtype=np.float64)
    n = len(close)

    if n == 0:
        return np.array([], dtype=np.float64)
    if n < timeperiod:
        return np.full(n, np.nan, dtype=np.float64)

    output = np.empty(n, dtype=np.float64)
    _sma_cumsum_numba(close, timeperiod, output)
    return output


def benchmark_sma():
    """Benchmark all SMA implementations"""
    try:
        import talib
        talib_available = True
    except ImportError:
        talib_available = False
        print("WARNING: TA-Lib not installed, skipping comparison")

    print("=" * 70)
    print("Simple Moving Average (SMA) Implementation Comparison")
    print("=" * 70)

    # Test different dataset sizes
    sizes = [100, 1000, 10000, 100000]
    timeperiods = [14, 50, 200]

    for n in sizes:
        print(f"\n{'=' * 70}")
        print(f"Dataset size: {n:,} bars")
        print('=' * 70)

        # Generate test data
        np.random.seed(42)
        close = np.random.uniform(90, 110, n)

        for timeperiod in timeperiods:
            if n < timeperiod:
                continue

            print(f"\nTimeperiod: {timeperiod}")
            print("-" * 70)

            # Benchmark rolling window (current implementation)
            # Warmup
            _ = sma_rolling(close, timeperiod)

            iterations = 1000 if n <= 10000 else 100
            start = time.perf_counter()
            for _ in range(iterations):
                result_rolling = sma_rolling(close, timeperiod)
            end = time.perf_counter()
            time_rolling = (end - start) / iterations

            print(f"Rolling (current): {time_rolling * 1000:.4f}ms ({time_rolling * 1e6:.2f}us)")

            # Benchmark cumsum
            # Warmup
            _ = sma_cumsum(close, timeperiod)

            start = time.perf_counter()
            for _ in range(iterations):
                result_cumsum = sma_cumsum(close, timeperiod)
            end = time.perf_counter()
            time_cumsum = (end - start) / iterations

            print(f"Cumsum:            {time_cumsum * 1000:.4f}ms ({time_cumsum * 1e6:.2f}us)")

            # Compare rolling vs cumsum
            speedup = time_cumsum / time_rolling
            if speedup > 1:
                print(f"Result:            Rolling is {speedup:.2f}x faster than cumsum")
            else:
                print(f"Result:            Cumsum is {1/speedup:.2f}x faster than rolling")

            # Verify accuracy between rolling and cumsum
            max_diff = np.max(np.abs(result_rolling - result_cumsum))
            print(f"Rolling vs Cumsum: Max diff = {max_diff:.2e}")

            # Benchmark TA-Lib if available
            if talib_available:
                # Warmup
                _ = talib.SMA(close, timeperiod=timeperiod)

                start = time.perf_counter()
                for _ in range(iterations):
                    result_talib = talib.SMA(close, timeperiod=timeperiod)
                end = time.perf_counter()
                time_talib = (end - start) / iterations

                print(f"TA-Lib:            {time_talib * 1000:.4f}ms ({time_talib * 1e6:.2f}us)")

                # Compare with TA-Lib
                speedup_rolling = time_talib / time_rolling
                speedup_cumsum = time_talib / time_cumsum

                if speedup_rolling > 1:
                    print(f"Rolling vs TA-Lib: Rolling is {speedup_rolling:.2f}x faster")
                else:
                    print(f"Rolling vs TA-Lib: TA-Lib is {1/speedup_rolling:.2f}x faster")

                if speedup_cumsum > 1:
                    print(f"Cumsum vs TA-Lib:  Cumsum is {speedup_cumsum:.2f}x faster")
                else:
                    print(f"Cumsum vs TA-Lib:  TA-Lib is {1/speedup_cumsum:.2f}x faster")

                # Verify accuracy with TA-Lib
                valid_mask = ~(np.isnan(result_rolling) | np.isnan(result_talib))
                if np.any(valid_mask):
                    diff_rolling = np.abs(result_rolling[valid_mask] - result_talib[valid_mask])
                    diff_cumsum = np.abs(result_cumsum[valid_mask] - result_talib[valid_mask])
                    max_diff_rolling = np.max(diff_rolling)
                    max_diff_cumsum = np.max(diff_cumsum)
                    print(f"Rolling vs TA-Lib: Max diff = {max_diff_rolling:.2e}")
                    print(f"Cumsum vs TA-Lib:  Max diff = {max_diff_cumsum:.2e}")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("Both implementations use Numba JIT compilation")
    print("Rolling window: O(n) time, O(1) extra space per iteration")
    print("Cumsum:         O(n) time, O(n) extra space for cumsum array")
    print("\nThe rolling window approach is typically faster due to:")
    print("1. Better cache locality (no extra array)")
    print("2. Fewer memory operations")
    print("3. More efficient for single-pass calculations")


if __name__ == "__main__":
    benchmark_sma()