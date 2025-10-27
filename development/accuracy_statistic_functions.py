"""
Accuracy comparison for Statistic Functions: talib-pure vs TA-Lib

This script compares the accuracy of Statistic Functions implementations
between talib-pure (Numba/CPU) and the original TA-Lib library.
"""

import numpy as np
import talib
from numta import CORREL, LINEARREG, LINEARREG_ANGLE, LINEARREG_INTERCEPT, LINEARREG_SLOPE, STDDEV, TSF, VAR


def generate_test_data(size, data_type='random_walk'):
    """Generate different types of test data"""
    np.random.seed(42)

    if data_type == 'random_walk':
        close = np.random.randn(size).cumsum() + 100
    elif data_type == 'trending':
        trend = np.linspace(0, 20, size)
        noise = np.random.randn(size) * 2
        close = 100 + trend + noise
    elif data_type == 'cyclical':
        cycle = 10 * np.sin(np.linspace(0, 8*np.pi, size))
        noise = np.random.randn(size) * 1
        close = 110 + cycle + noise
    elif data_type == 'mixed':
        trend = np.linspace(0, 10, size)
        cycle = 5 * np.sin(np.linspace(0, 6*np.pi, size))
        noise = np.random.randn(size) * 1.5
        close = 100 + trend + cycle + noise

    # Generate high and low from close
    high = close + np.random.rand(size) * 2
    low = close - np.random.rand(size) * 2

    return high, low, close


def calculate_single_accuracy(result_talib, result_pure, name):
    """Calculate accuracy metrics for a single output"""
    # Handle NaN values
    valid_mask = ~(np.isnan(result_talib) | np.isnan(result_pure))

    if not np.any(valid_mask):
        return {
            'name': name,
            'mae': float('inf'),
            'rmse': float('inf'),
            'max_error': float('inf'),
            'correlation': 0.0,
            'exact_match_rate': 0.0
        }

    valid_talib = result_talib[valid_mask]
    valid_pure = result_pure[valid_mask]

    # Calculate metrics
    diff = np.abs(valid_talib - valid_pure)
    mae = np.mean(diff)
    rmse = np.sqrt(np.mean((valid_talib - valid_pure) ** 2))
    max_error = np.max(diff)

    # Correlation
    if np.std(valid_talib) > 0 and np.std(valid_pure) > 0:
        correlation = np.corrcoef(valid_talib, valid_pure)[0, 1]
    else:
        correlation = 0.0

    # Exact match rate (within tolerance)
    exact_match_rate = (np.sum(diff < 1e-10) / len(valid_talib)) * 100

    return {
        'name': name,
        'mae': mae,
        'rmse': rmse,
        'max_error': max_error,
        'correlation': correlation,
        'exact_match_rate': exact_match_rate
    }


def calculate_accuracy(result_talib, result_pure, name):
    """Calculate accuracy metrics between TA-Lib and talib-pure results"""
    # Handle both single and tuple outputs
    if isinstance(result_talib, tuple):
        results = []
        for i, (rt, rp) in enumerate(zip(result_talib, result_pure)):
            suffix = f" (output {i+1})"
            results.append(calculate_single_accuracy(rt, rp, name + suffix))
        return results
    else:
        return [calculate_single_accuracy(result_talib, result_pure, name)]


def test_indicator(name, func_talib, func_pure, talib_args, pure_args, data_type):
    """Test a single indicator and return accuracy metrics"""
    try:
        result_talib = func_talib(*talib_args)
        result_pure = func_pure(*pure_args)

        return calculate_accuracy(result_talib, result_pure, name)
    except Exception as e:
        print(f"Error testing {name} with {data_type}: {e}")
        return None


def run_accuracy_tests(size=10000):
    """Run accuracy tests with different data types"""
    data_types = ['random_walk', 'trending', 'cyclical', 'mixed']

    all_results = {dt: [] for dt in data_types}

    for data_type in data_types:
        print(f"\n{'='*70}")
        print(f"Testing with {data_type.replace('_', ' ').title()} Data ({size:,} bars)")
        print(f"{'='*70}\n")

        high, low, close = generate_test_data(size, data_type)
        timeperiod = 30
        timeperiod_lr = 14  # Linear regression functions use 14 by default

        # Define all indicators to test
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

        for name, func_talib, func_pure, talib_args, pure_args in indicators:
            results = test_indicator(name, func_talib, func_pure, talib_args, pure_args, data_type)
            if results:
                for result in results:
                    all_results[data_type].append(result)

                    # Determine status
                    if result['mae'] == 0.0 and result['exact_match_rate'] == 100.0:
                        status = "EXACT"
                    elif result['mae'] < 1e-10 and result['correlation'] > 0.9999:
                        status = "NEAR-EXACT"
                    elif result['mae'] < 1e-6 and result['correlation'] > 0.999:
                        status = "EXCELLENT"
                    elif result['correlation'] > 0.99:
                        status = "GOOD"
                    elif result['correlation'] > 0.90:
                        status = "MODERATE"
                    else:
                        status = "POOR"

                    print(f"{result['name']:25s}: {status:12s} "
                          f"(MAE: {result['mae']:.2e}, "
                          f"Correlation: {result['correlation']:.10f}, "
                          f"Exact Match: {result['exact_match_rate']:.2f}%)")

    return all_results


def print_summary(all_results):
    """Print summary of results across all data types"""
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY - Average Across All Data Types")
    print(f"{'='*70}\n")

    # Aggregate results by function name
    func_stats = {}
    for data_type, results in all_results.items():
        for result in results:
            name = result['name']
            if name not in func_stats:
                func_stats[name] = {
                    'mae': [],
                    'rmse': [],
                    'max_error': [],
                    'correlation': [],
                    'exact_match_rate': []
                }
            func_stats[name]['mae'].append(result['mae'])
            func_stats[name]['rmse'].append(result['rmse'])
            func_stats[name]['max_error'].append(result['max_error'])
            func_stats[name]['correlation'].append(result['correlation'])
            func_stats[name]['exact_match_rate'].append(result['exact_match_rate'])

    # Print summary table
    print(f"{'Function':<25s} | {'Avg MAE':>10s} | {'Avg Corr':>10s} | {'Match %':>8s} | Status")
    print(f"{'-'*25}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+{'-'*15}")

    for name in sorted(func_stats.keys()):
        avg_mae = np.mean(func_stats[name]['mae'])
        avg_corr = np.mean(func_stats[name]['correlation'])
        avg_match = np.mean(func_stats[name]['exact_match_rate'])

        if avg_mae == 0.0 and avg_match == 100.0:
            status = "✅ Exact"
        elif avg_mae < 1e-10 and avg_corr > 0.9999:
            status = "✅ Near-Exact"
        elif avg_mae < 1e-6 and avg_corr > 0.999:
            status = "✅ Excellent"
        elif avg_corr > 0.99:
            status = "✅ Good"
        elif avg_corr > 0.90:
            status = "⚠️ Moderate"
        else:
            status = "❌ Poor"

        print(f"{name:<25s} | {avg_mae:>10.2e} | {avg_corr:>10.6f} | {avg_match:>7.2f}% | {status}")


def main():
    """Main execution"""
    print("="*70)
    print("Statistic Functions Accuracy Comparison")
    print("Comparing talib-pure (Numba/CPU) vs TA-Lib")
    print("="*70)

    all_results = run_accuracy_tests(size=10000)
    print_summary(all_results)

    print(f"\n{'='*70}")
    print("Accuracy testing complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
