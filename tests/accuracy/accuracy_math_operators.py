"""
Accuracy comparison between talib-pure (Numba/CPU) and original TA-Lib
for Math Operators
"""

import numpy as np
import talib
from numta import (
    MAX, MIN, MINMAX, MAXINDEX, MININDEX, MINMAXINDEX, SUM
)


def calculate_accuracy_metrics(result_talib, result_pure, name):
    """Calculate various accuracy metrics between two results"""

    # Handle tuple outputs (MINMAX, MINMAXINDEX)
    if isinstance(result_talib, tuple):
        metrics = []
        output_names = ['Output 1', 'Output 2']
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
        return np.cumsum(np.random.randn(size) * 0.5) + 100

    elif data_type == 'trending':
        # Upward trend with noise
        trend = np.linspace(100, 120, size)
        noise = np.random.randn(size) * 0.5
        return trend + noise

    elif data_type == 'cyclical':
        # Sine wave with noise
        x = np.linspace(0, 10 * np.pi, size)
        cycle = 10 * np.sin(x) + 100
        noise = np.random.randn(size) * 0.3
        return cycle + noise

    elif data_type == 'mixed':
        # Combination of trend, cycle, and noise
        x = np.linspace(0, 10 * np.pi, size)
        trend = np.linspace(100, 110, size)
        cycle = 5 * np.sin(x)
        noise = np.random.randn(size) * 0.5
        return trend + cycle + noise

    else:
        raise ValueError(f"Unknown data type: {data_type}")


def test_operator_accuracy(func_talib, func_pure, name, test_data, timeperiod=30):
    """Test accuracy of a single operator"""

    result_talib = func_talib(test_data, timeperiod=timeperiod)
    result_pure = func_pure(test_data, timeperiod=timeperiod)

    metrics = calculate_accuracy_metrics(result_talib, result_pure, name)

    return metrics


def main():
    """Run all accuracy tests"""

    print("=" * 80)
    print("Math Operators Accuracy Comparison")
    print("talib-pure (Numba/CPU) vs Original TA-Lib")
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

    # Math operator functions
    operators = [
        ('MAX', talib.MAX, MAX),
        ('MIN', talib.MIN, MIN),
        ('MINMAX', talib.MINMAX, MINMAX),
        ('MAXINDEX', talib.MAXINDEX, MAXINDEX),
        ('MININDEX', talib.MININDEX, MININDEX),
        ('MINMAXINDEX', talib.MINMAXINDEX, MINMAXINDEX),
        ('SUM', talib.SUM, SUM),
    ]

    all_results = {}

    for data_type in data_types:
        print(f"\n{'=' * 80}")
        print(f"Test Data Type: {data_type_labels[data_type]}")
        print(f"Dataset Size: {size:,} bars, timeperiod={timeperiod}")
        print('=' * 80)

        test_data = generate_test_data(size, data_type)

        all_results[data_type] = {}

        for name, func_talib, func_pure in operators:
            metrics_list = test_operator_accuracy(func_talib, func_pure, name, test_data, timeperiod)

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

        for name, _, _ in operators:
            metrics_list = all_results[data_type][name]

            # For multi-output functions, show both
            for metrics in metrics_list:
                display_name = metrics['name']
                print(f"| {display_name:20} | {metrics['mae']:.2e} | "
                      f"{metrics['rmse']:.2e} | {metrics['max_error']:.2e} | "
                      f"{metrics['correlation']:.6f} | {metrics['exact_match_rate']:6.2f}% |")

    # Overall summary table
    print("\n### Overall Summary (Average across all data types)")
    print()
    print("| Function | Avg MAE | Avg RMSE | Avg Max Error | Avg Correlation | Avg Exact Match |")
    print("|----------|---------|----------|---------------|-----------------|-----------------|")

    for name, _, _ in operators:
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
            name_display = f"{name} (both outputs)"
        else:
            name_display = name

        print(f"| {name_display:20} | {avg_mae:.2e} | {avg_rmse:.2e} | "
              f"{avg_max_error:.2e} | {avg_correlation:.6f} | {avg_exact_match:6.2f}% |")

    print("\n" + "=" * 80)
    print("\nAccuracy Classification:")
    print("=" * 80)

    # Classify accuracy
    for name, _, _ in operators:
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
        else:
            accuracy = "MODERATE"

        print(f"{name:15} : {accuracy} (MAE: {avg_mae:.2e}, Correlation: {avg_correlation:.10f}, Exact Match: {avg_exact_match:.2f}%)")


if __name__ == "__main__":
    main()
