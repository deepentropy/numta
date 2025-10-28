"""
Accuracy comparison between numta (Numba/CPU) and original TA-Lib
for Overlap Indicators
"""

import numpy as np
import talib
from numta import (
    SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, MAMA, T3, BBANDS, SAR, SAREXT
)


def calculate_accuracy_metrics(result_talib, result_pure, name):
    """Calculate various accuracy metrics between two results"""

    # Handle tuple outputs (BBANDS, MAMA)
    if isinstance(result_talib, tuple):
        metrics = []
        if len(result_talib) == 3:  # BBANDS
            output_names = ['Upper Band', 'Middle Band', 'Lower Band']
        else:  # MAMA
            output_names = ['MAMA', 'FAMA']

        for i, (ta, pure) in enumerate(zip(result_talib, result_pure)):
            metrics.append(calculate_single_accuracy(ta, pure, f"{name} - {output_names[i]}"))
        return metrics
    else:
        return [calculate_single_accuracy(result_talib, result_pure, name)]


def calculate_single_accuracy(result_talib, result_pure, name):
    """Calculate accuracy metrics for a single output array"""

    # Find valid indices (non-NaN in both arrays)
    valid_mask = ~(np.isnan(result_talib) | np.isnan(result_pure))
    valid_talib = result_talib[valid_mask]
    valid_pure = result_pure[valid_mask]

    if len(valid_talib) == 0:
        return {
            'name': name,
            'mae': 0.0,
            'rmse': 0.0,
            'max_error': 0.0,
            'correlation': 1.0,
            'exact_match_rate': 100.0,
            'valid_count': 0,
            'total_count': len(result_talib)
        }

    # Calculate metrics
    diff = np.abs(valid_talib - valid_pure)
    mae = np.mean(diff)
    rmse = np.sqrt(np.mean((valid_talib - valid_pure) ** 2))
    max_error = np.max(diff)

    # Correlation coefficient
    if np.std(valid_talib) > 0 and np.std(valid_pure) > 0:
        correlation = np.corrcoef(valid_talib, valid_pure)[0, 1]
    else:
        correlation = 1.0 if mae < 1e-10 else 0.0

    # Exact match rate (within floating point tolerance)
    tolerance = 1e-10
    exact_matches = np.sum(diff < tolerance)
    exact_match_rate = (exact_matches / len(valid_talib)) * 100

    return {
        'name': name,
        'mae': mae,
        'rmse': rmse,
        'max_error': max_error,
        'correlation': correlation,
        'exact_match_rate': exact_match_rate,
        'valid_count': len(valid_talib),
        'total_count': len(result_talib)
    }


def generate_test_data(size, data_type='random', seed=42):
    """Generate various types of test data"""
    np.random.seed(seed)

    if data_type == 'random':
        # Random walk
        close = np.cumsum(np.random.randn(size) * 0.5) + 100
    elif data_type == 'trending':
        # Upward trend with noise
        trend = np.linspace(100, 120, size)
        noise = np.random.randn(size) * 0.5
        close = trend + noise
    elif data_type == 'cyclical':
        # Sine wave with noise
        x = np.linspace(0, 10 * np.pi, size)
        cycle = 10 * np.sin(x) + 100
        noise = np.random.randn(size) * 0.3
        close = cycle + noise
    elif data_type == 'mixed':
        # Combination of trend, cycle, and noise
        x = np.linspace(0, 10 * np.pi, size)
        trend = np.linspace(100, 110, size)
        cycle = 5 * np.sin(x)
        noise = np.random.randn(size) * 0.5
        close = trend + cycle + noise
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    # Generate high and low based on close
    high = close + np.abs(np.random.randn(size) * 2)
    low = close - np.abs(np.random.randn(size) * 2)

    return close, high, low


def test_indicator_accuracy(func_talib, func_pure, name, args):
    """Test accuracy of a single indicator"""

    result_talib = func_talib(*args)
    result_pure = func_pure(*args)

    metrics = calculate_accuracy_metrics(result_talib, result_pure, name)

    return metrics


def main():
    """Run all accuracy tests"""

    print("=" * 80)
    print("Overlap Indicators Accuracy Comparison")
    print("numta (Numba/CPU) vs Original TA-Lib")
    print("=" * 80)
    print()

    # Test configurations
    data_types = ['random', 'trending', 'cyclical', 'mixed']
    data_type_labels = {
        'random': 'Random Walk',
        'trending': 'Trending + Noise',
        'cyclical': 'Cyclical + Noise',
        'mixed': 'Mixed (Trend + Cycle + Noise)'
    }

    size = 10000  # Test with 10K bars
    timeperiod = 30

    all_results = {}

    for data_type in data_types:
        print(f"\n{'=' * 80}")
        print(f"Test Data Type: {data_type_labels[data_type]}")
        print(f"Dataset Size: {size:,} bars, timeperiod={timeperiod}")
        print('=' * 80)

        close, high, low = generate_test_data(size, data_type)

        all_results[data_type] = {}

        # Overlap indicators with their parameters (using positional args)
        indicators = [
            ('SMA', talib.SMA, SMA, (close, timeperiod)),
            ('EMA', talib.EMA, EMA, (close, timeperiod)),
            ('WMA', talib.WMA, WMA, (close, timeperiod)),
            ('DEMA', talib.DEMA, DEMA, (close, timeperiod)),
            ('TEMA', talib.TEMA, TEMA, (close, timeperiod)),
            ('TRIMA', talib.TRIMA, TRIMA, (close, timeperiod)),
            ('KAMA', talib.KAMA, KAMA, (close, timeperiod)),
            ('MAMA', talib.MAMA, MAMA, (close, 0.5, 0.05)),
            ('T3', talib.T3, T3, (close, 5, 0.7)),
            ('BBANDS', talib.BBANDS, BBANDS, (close, timeperiod, 2, 2)),
            ('SAR', talib.SAR, SAR, (high, low, 0.02, 0.2)),
            ('SAREXT', talib.SAREXT, SAREXT, (high, low, 0, 0, 0.02, 0.02, 0.2, 0.02, 0.02, 0.2)),
        ]

        for name, func_talib, func_pure, args in indicators:
            metrics_list = test_indicator_accuracy(func_talib, func_pure, name, args)

            all_results[data_type][name] = metrics_list

            # Print results for each output
            for metrics in metrics_list:
                print(f"\n{metrics['name']}:")
                print(f"  MAE:              {metrics['mae']:.15f}")
                print(f"  RMSE:             {metrics['rmse']:.15f}")
                print(f"  Max Error:        {metrics['max_error']:.15f}")
                print(f"  Correlation:      {metrics['correlation']:.15f}")
                print(f"  Exact Match Rate: {metrics['exact_match_rate']:.2f}%")
                print(f"  Valid/Total:      {metrics['valid_count']}/{metrics['total_count']}")

    print("\n" + "=" * 80)
    print("\nSummary Tables (for ACCURACY.md):")
    print("=" * 80)

    # Summary table by data type
    for data_type in data_types:
        print(f"\n### {data_type_labels[data_type]}")
        print()
        print("| Function | MAE | RMSE | Max Error | Correlation | Exact Match |")
        print("|----------|-----|------|-----------|-------------|-------------|")

        indicator_names = ['SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 'TRIMA', 'KAMA', 'MAMA', 'T3', 'BBANDS', 'SAR', 'SAREXT']

        for name in indicator_names:
            metrics_list = all_results[data_type][name]

            # For multi-output functions, show each output
            for metrics in metrics_list:
                display_name = metrics['name']
                print(f"| {display_name:25} | {metrics['mae']:.2e} | "
                      f"{metrics['rmse']:.2e} | {metrics['max_error']:.2e} | "
                      f"{metrics['correlation']:.6f} | {metrics['exact_match_rate']:6.2f}% |")

    # Overall summary table
    print("\n### Overall Summary (Average across all data types)")
    print()
    print("| Function | Avg MAE | Avg RMSE | Avg Max Error | Avg Correlation | Avg Exact Match |")
    print("|----------|---------|----------|---------------|-----------------|-----------------|")

    indicator_names = ['SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 'TRIMA', 'KAMA', 'MAMA', 'T3', 'BBANDS', 'SAR', 'SAREXT']

    for name in indicator_names:
        # Calculate averages across all data types
        all_metrics = []
        for data_type in data_types:
            all_metrics.extend(all_results[data_type][name])

        avg_mae = np.mean([m['mae'] for m in all_metrics])
        avg_rmse = np.mean([m['rmse'] for m in all_metrics])
        avg_max_error = np.mean([m['max_error'] for m in all_metrics])
        avg_correlation = np.mean([m['correlation'] for m in all_metrics])
        avg_exact_match = np.mean([m['exact_match_rate'] for m in all_metrics])

        # Determine if multi-output
        if len(all_results[data_types[0]][name]) > 1:
            name_display = f"{name} (all outputs)"
        else:
            name_display = name

        print(f"| {name_display:25} | {avg_mae:.2e} | {avg_rmse:.2e} | "
              f"{avg_max_error:.2e} | {avg_correlation:.6f} | {avg_exact_match:6.2f}% |")

    print("\n" + "=" * 80)
    print("\nAccuracy Classification:")
    print("=" * 80)

    # Classify accuracy
    for name in indicator_names:
        all_metrics = []
        for data_type in data_types:
            all_metrics.extend(all_results[data_type][name])

        avg_mae = np.mean([m['mae'] for m in all_metrics])
        avg_correlation = np.mean([m['correlation'] for m in all_metrics])
        avg_exact_match = np.mean([m['exact_match_rate'] for m in all_metrics])

        if avg_exact_match > 99.9 and avg_correlation > 0.999999:
            accuracy = "EXACT"
        elif avg_mae < 1e-10 and avg_correlation > 0.99999:
            accuracy = "NEAR-EXACT"
        elif avg_mae < 1e-6 and avg_correlation > 0.9999:
            accuracy = "VERY HIGH"
        elif avg_mae < 1e-3 and avg_correlation > 0.999:
            accuracy = "HIGH"
        elif avg_correlation > 0.99:
            accuracy = "GOOD"
        else:
            accuracy = "MODERATE"

        print(f"{name:15} : {accuracy} (MAE: {avg_mae:.2e}, Correlation: {avg_correlation:.10f}, Exact Match: {avg_exact_match:.2f}%)")


if __name__ == "__main__":
    main()
