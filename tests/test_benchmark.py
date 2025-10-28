"""
Test suite for the performance measurement class
"""

import numpy as np
import pytest
from talib_pure.benchmark import PerformanceMeasurement


def dummy_function_fast(data, param=10):
    """Fast dummy function for testing"""
    return np.mean(data[:param])


def dummy_function_slow(data, param=10):
    """Slower dummy function for testing"""
    import time
    time.sleep(0.001)  # Simulate some work
    return np.mean(data[:param])


def test_performance_measurement_init():
    """Test PerformanceMeasurement initialization"""
    bench = PerformanceMeasurement()
    assert bench.functions == []


def test_add_function():
    """Test adding functions to benchmark"""
    bench = PerformanceMeasurement()
    data = np.random.uniform(100, 200, 100)

    bench.add_function("test_func", dummy_function_fast, data, param=5)
    assert len(bench.functions) == 1
    assert bench.functions[0]['name'] == "test_func"


def test_clear():
    """Test clearing registered functions"""
    bench = PerformanceMeasurement()
    data = np.random.uniform(100, 200, 100)

    bench.add_function("test1", dummy_function_fast, data)
    bench.add_function("test2", dummy_function_slow, data)
    assert len(bench.functions) == 2

    bench.clear()
    assert len(bench.functions) == 0


def test_measure_single_run():
    """Test single execution measurement"""
    bench = PerformanceMeasurement()
    data = np.random.uniform(100, 200, 100)

    exec_time = bench.measure_single_run(dummy_function_fast, data, param=10)
    assert exec_time > 0
    assert isinstance(exec_time, float)


def test_run_benchmark():
    """Test running a benchmark"""
    bench = PerformanceMeasurement()
    data = np.random.uniform(100, 200, 100)

    bench.add_function("fast", dummy_function_fast, data, param=10)
    results = bench.run(iterations=10, warmup=2)

    assert "fast" in results
    assert "mean" in results["fast"]
    assert "median" in results["fast"]
    assert "stdev" in results["fast"]
    assert "min" in results["fast"]
    assert "max" in results["fast"]
    assert results["fast"]["iterations"] == 10


def test_run_no_functions_error():
    """Test that run() raises error when no functions are registered"""
    bench = PerformanceMeasurement()

    with pytest.raises(ValueError, match="No functions registered"):
        bench.run()


def test_speedup_calculation():
    """Test speedup calculation between functions"""
    bench = PerformanceMeasurement()
    data = np.random.uniform(100, 200, 100)

    bench.add_function("fast", dummy_function_fast, data)
    bench.add_function("slow", dummy_function_slow, data)

    results = bench.run(iterations=10, warmup=2)

    assert "speedup" in results["fast"]
    assert "speedup" in results["slow"]
    assert results["fast"]["speedup"] == 1.0  # Baseline
    assert results["slow"]["speedup"] < 1.0  # Slower than baseline


def test_compare_with_data_sizes():
    """Test comparison across different data sizes"""
    bench = PerformanceMeasurement()

    func_pairs = [
        ("fast", dummy_function_fast, {"param": 10}),
        ("slow", dummy_function_slow, {"param": 10}),
    ]

    data_sizes = [100, 500]
    results = bench.compare_with_data_sizes(
        func_pairs,
        data_sizes,
        iterations=5
    )

    assert len(results) == 2
    assert 100 in results
    assert 500 in results
    assert "fast" in results[100]
    assert "slow" in results[100]


def test_custom_data_generator():
    """Test using a custom data generator"""
    bench = PerformanceMeasurement()

    def custom_generator(size):
        return np.ones(size) * 42

    func_pairs = [
        ("test", dummy_function_fast, {"param": 10}),
    ]

    results = bench.compare_with_data_sizes(
        func_pairs,
        [50, 100],
        iterations=5,
        data_generator=custom_generator
    )

    assert len(results) == 2
