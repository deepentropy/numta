"""
Accuracy comparison between talib-pure (Numba/CPU) and original TA-Lib
for Cycle Indicators
"""

import numpy as np
import talib
from numta import (
    HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, HT_SINE, HT_TRENDLINE, HT_TRENDMODE
)


def calculate_accuracy_metrics(result_talib, result_pure, name):
    """Calculate various accuracy metrics between two results"""

    # Handle tuple outputs (HT_PHASOR, HT_SINE)
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
            'match_rate': 100.0,
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

    # Match rate (within tolerance)
    tolerance = 1e-6
    matches = np.sum(diff < tolerance)
    match_rate = (matches / len(valid_talib)) * 100

    return {
        'name': name,
        'mae': mae,
        'rmse': rmse,
        'max_error': max_error,
        'correlation': correlation,
        'match_rate': match_rate,
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


def test_indicator_accuracy(func_talib, func_pure, name, test_data):
    """Test accuracy of a single indicator"""

    result_talib = func_talib(test_data)
    result_pure = func_pure(test_data)

    metrics = calculate_accuracy_metrics(result_talib, result_pure, name)

    return metrics


def main():
    """Run all accuracy tests"""

    print("=" * 80)
    print("Cycle Indicators Accuracy Comparison")
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

    # Cycle indicator functions
    indicators = [
        ('HT_DCPERIOD', talib.HT_DCPERIOD, HT_DCPERIOD),
        ('HT_DCPHASE', talib.HT_DCPHASE, HT_DCPHASE),
        ('HT_PHASOR', talib.HT_PHASOR, HT_PHASOR),
        ('HT_SINE', talib.HT_SINE, HT_SINE),
        ('HT_TRENDLINE', talib.HT_TRENDLINE, HT_TRENDLINE),
        ('HT_TRENDMODE', talib.HT_TRENDMODE, HT_TRENDMODE),
    ]

    all_results = {}

    for data_type in data_types:
        print(f"\n{'=' * 80}")
        print(f"Test Data Type: {data_type_labels[data_type]}")
        print(f"Dataset Size: {size:,} bars")
        print('=' * 80)

        test_data = generate_test_data(size, data_type)

        all_results[data_type] = {}

        for name, func_talib, func_pure in indicators:
            metrics_list = test_indicator_accuracy(func_talib, func_pure, name, test_data)

            all_results[data_type][name] = metrics_list

            # Print results for each output
            for metrics in metrics_list:
                print(f"\n{metrics['name']}:")
                print(f"  MAE:         {metrics['mae']:.10f}")
                print(f"  RMSE:        {metrics['rmse']:.10f}")
                print(f"  Max Error:   {metrics['max_error']:.10f}")
                print(f"  Correlation: {metrics['correlation']:.10f}")
                print(f"  Match Rate:  {metrics['match_rate']:.2f}%")
                print(f"  Valid/Total: {metrics['valid_count']}/{metrics['total_count']}")

    print("\n" + "=" * 80)
    print("\nSummary Tables (for ACCURACY.md):")
    print("=" * 80)

    # Summary table by data type
    for data_type in data_types:
        print(f"\n### {data_type_labels[data_type]}")
        print()
        print("| Function | MAE | RMSE | Max Error | Correlation |")
        print("|----------|-----|------|-----------|-------------|")

        for name, _, _ in indicators:
            metrics_list = all_results[data_type][name]

            # For multi-output functions, show both
            for metrics in metrics_list:
                display_name = metrics['name']
                print(f"| {display_name:25} | {metrics['mae']:.2e} | "
                      f"{metrics['rmse']:.2e} | {metrics['max_error']:.2e} | "
                      f"{metrics['correlation']:.6f} |")

    # Overall summary table
    print("\n### Overall Summary (Average across all data types)")
    print()
    print("| Function | Avg MAE | Avg RMSE | Avg Max Error | Avg Correlation |")
    print("|----------|---------|----------|---------------|-----------------|")

    for name, _, _ in indicators:
        # Calculate averages across all data types
        all_metrics = []
        for data_type in data_types:
            all_metrics.extend(all_results[data_type][name])

        avg_mae = np.mean([m['mae'] for m in all_metrics])
        avg_rmse = np.mean([m['rmse'] for m in all_metrics])
        avg_max_error = np.mean([m['max_error'] for m in all_metrics])
        avg_correlation = np.mean([m['correlation'] for m in all_metrics])

        # Determine if multi-output
        if len(all_results[data_types[0]][name]) > 1:
            name_display = f"{name} (both outputs)"
        else:
            name_display = name

        print(f"| {name_display:25} | {avg_mae:.2e} | {avg_rmse:.2e} | "
              f"{avg_max_error:.2e} | {avg_correlation:.6f} |")

    print("\n" + "=" * 80)
    print("\nAccuracy Classification:")
    print("=" * 80)

    # Classify accuracy
    for name, _, _ in indicators:
        all_metrics = []
        for data_type in data_types:
            all_metrics.extend(all_results[data_type][name])

        avg_mae = np.mean([m['mae'] for m in all_metrics])
        avg_correlation = np.mean([m['correlation'] for m in all_metrics])

        if avg_mae < 1e-10 and avg_correlation > 0.999999:
            accuracy = "EXACT"
        elif avg_mae < 1e-6 and avg_correlation > 0.99999:
            accuracy = "NEAR-EXACT"
        elif avg_mae < 1e-3 and avg_correlation > 0.9999:
            accuracy = "VERY HIGH"
        elif avg_mae < 0.01 and avg_correlation > 0.999:
            accuracy = "HIGH"
        else:
            accuracy = "MODERATE"

        print(f"{name:15} : {accuracy} (MAE: {avg_mae:.2e}, Correlation: {avg_correlation:.8f})")


if __name__ == "__main__":
    main()
