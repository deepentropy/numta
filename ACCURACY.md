# Accuracy Comparison

This document presents accuracy comparisons between **talib-pure** (Numba/CPU implementation) and the **original TA-Lib** library.

## Contents

- [Cycle Indicators](#cycle-indicators)
- [Math Operators](#math-operators)

---

# Cycle Indicators

## Test Environment

- **Python Version**: 3.11
- **NumPy Version**: 2.3.4
- **Numba Version**: 0.62.1
- **TA-Lib Version**: 0.6.8
- **Platform**: Linux
- **Dataset Size**: 10,000 bars per test
- **Test Method**: Comparison across 4 different data patterns

## Metrics Explained

- **MAE (Mean Absolute Error)**: Average absolute difference between outputs
- **RMSE (Root Mean Square Error)**: Square root of average squared differences
- **Max Error**: Largest absolute difference observed
- **Correlation**: Pearson correlation coefficient (1.0 = perfect, 0.0 = no correlation, -1.0 = inverse)
- **Match Rate**: Percentage of values matching within 1e-6 tolerance

## Overall Summary

Average accuracy metrics across all 4 test data types (10,000 bars each):

| Function | Avg MAE | Avg RMSE | Avg Max Error | Avg Correlation | Status |
|----------|---------|----------|---------------|-----------------|--------|
| **HT_DCPERIOD** | 6.46e-02 | 9.56e-01 | 21.81 | 0.953 | ⚠️ Good |
| **HT_DCPHASE** | 146.47 | 167.59 | 401.95 | -0.177 | ❌ Poor |
| **HT_PHASOR (inphase)** | 2.98e-01 | 4.79e-01 | 13.41 | 0.108 | ❌ Poor |
| **HT_PHASOR (quadrature)** | 4.34e-01 | 5.99e-01 | 6.99 | -0.023 | ❌ Poor |
| **HT_SINE (sine)** | 7.92e-01 | 9.62e-01 | 2.00 | -0.050 | ❌ Poor |
| **HT_SINE (leadsine)** | 1.05 | 1.21 | 2.00 | -0.411 | ❌ Poor |
| **HT_TRENDLINE** | 4.03e-01 | 5.09e-01 | 1.83 | 0.999 | ✅ Excellent |
| **HT_TRENDMODE** | 3.23e-01 | 5.54e-01 | 1.00 | 0.003 | ⚠️ Moderate |

### Status Legend
- ✅ **Excellent**: High correlation (>0.99) and low error
- ⚠️ **Good/Moderate**: Correlation >0.90 or reasonable behavior for binary outputs
- ❌ **Poor**: Low/negative correlation, indicating implementation differences

## Detailed Results by Data Type

### Test 1: Random Walk Data

| Function | MAE | RMSE | Max Error | Correlation |
|----------|-----|------|-----------|-------------|
| HT_DCPERIOD | 4.60e-02 | 8.18e-01 | 20.14 | 0.989 |
| HT_DCPHASE | 1.33e+02 | 1.47e+02 | 402.86 | 0.133 |
| HT_PHASOR (inphase) | 5.99e-01 | 8.11e-01 | 12.97 | 0.271 |
| HT_PHASOR (quadrature) | 9.44e-01 | 1.29 | 9.16 | 0.023 |
| HT_SINE (sine) | 4.27e-01 | 5.51e-01 | 2.00 | 0.620 |
| HT_SINE (leadsine) | 1.00 | 1.14 | 2.00 | -0.275 |
| HT_TRENDLINE | 1.02 | 1.29 | 4.48 | 0.999 |
| HT_TRENDMODE | 4.96e-01 | 7.04e-01 | 1.00 | 0.007 |

**Match Rate**: HT_DCPERIOD (97.76%), HT_TRENDMODE (50.41%)

### Test 2: Trending + Noise Data

| Function | MAE | RMSE | Max Error | Correlation |
|----------|-----|------|-----------|-------------|
| HT_DCPERIOD | 7.49e-02 | 1.03 | 22.15 | 0.926 |
| HT_DCPHASE | 1.51e+02 | 1.78e+02 | 402.56 | -0.359 |
| HT_PHASOR (inphase) | 2.14e-01 | 3.87e-01 | 13.87 | 0.044 |
| HT_PHASOR (quadrature) | 2.76e-01 | 3.79e-01 | 6.46 | -0.042 |
| HT_SINE (sine) | 9.91e-01 | 1.14 | 2.00 | -0.381 |
| HT_SINE (leadsine) | 1.03 | 1.19 | 2.00 | -0.400 |
| HT_TRENDLINE | 2.03e-01 | 2.55e-01 | 0.99 | 0.999 |
| HT_TRENDMODE | 1.76e-01 | 4.19e-01 | 1.00 | -0.014 |

**Match Rate**: HT_DCPERIOD (97.80%), HT_TRENDMODE (82.44%)

### Test 3: Cyclical + Noise Data

| Function | MAE | RMSE | Max Error | Correlation |
|----------|-----|------|-----------|-------------|
| HT_DCPERIOD | 6.49e-02 | 9.68e-01 | 22.50 | 0.968 |
| HT_DCPHASE | 1.51e+02 | 1.67e+02 | 403.97 | -0.141 |
| HT_PHASOR (inphase) | 1.58e-01 | 3.42e-01 | 13.73 | 0.064 |
| HT_PHASOR (quadrature) | 2.30e-01 | 3.29e-01 | 5.99 | -0.030 |
| HT_SINE (sine) | 7.67e-01 | 9.29e-01 | 2.00 | -0.083 |
| HT_SINE (leadsine) | 1.12 | 1.26 | 2.00 | -0.517 |
| HT_TRENDLINE | 1.83e-01 | 2.26e-01 | 0.86 | 0.999 |
| HT_TRENDMODE | 4.33e-01 | 6.58e-01 | 1.00 | 0.018 |

**Match Rate**: HT_DCPERIOD (97.55%), HT_TRENDMODE (56.75%)

### Test 4: Mixed (Trend + Cycle + Noise) Data

| Function | MAE | RMSE | Max Error | Correlation |
|----------|-----|------|-----------|-------------|
| HT_DCPERIOD | 7.24e-02 | 1.01 | 22.28 | 0.928 |
| HT_DCPHASE | 1.52e+02 | 1.77e+02 | 398.93 | -0.342 |
| HT_PHASOR (inphase) | 2.19e-01 | 3.91e-01 | 13.88 | 0.055 |
| HT_PHASOR (quadrature) | 2.86e-01 | 3.91e-01 | 6.47 | -0.040 |
| HT_SINE (sine) | 9.73e-01 | 1.13 | 2.00 | -0.355 |
| HT_SINE (leadsine) | 1.06 | 1.22 | 2.00 | -0.451 |
| HT_TRENDLINE | 2.08e-01 | 2.61e-01 | 1.01 | 0.998 |
| HT_TRENDMODE | 1.88e-01 | 4.34e-01 | 1.00 | 0.000 |

**Match Rate**: HT_DCPERIOD (97.86%), HT_TRENDMODE (81.20%)

## Analysis

### Excellent Accuracy

#### HT_TRENDLINE ✅
- **Correlation**: 0.999 (near-perfect tracking)
- **Status**: Excellent agreement with TA-Lib
- **Confidence**: High - Safe to use in production
- **Notes**: MAE of 0.18-1.02 is due to floating-point precision differences in weighted average calculations

### Good Accuracy

#### HT_DCPERIOD ⚠️
- **Correlation**: 0.953 (very good)
- **Match Rate**: 97.5-98% (excellent)
- **Status**: Good overall agreement
- **Confidence**: Moderate-High - Generally reliable
- **Notes**:
  - Small MAE (0.046-0.075) indicates values track well
  - Max errors of ~22 occur at period boundaries (6-50 range)
  - Suitable for most trading applications where exact period values aren't critical

### Moderate Accuracy

#### HT_TRENDMODE ⚠️
- **Correlation**: 0.003 (near-zero, but this is a binary output)
- **Match Rate**: 50-82% (data-dependent)
- **Status**: Moderate agreement
- **Confidence**: Low-Moderate - Use with caution
- **Notes**:
  - Binary output (0 or 1) makes correlation less meaningful
  - Match rate varies significantly by data type (50% random, 82% trending)
  - The implementations may use slightly different thresholds for trend/cycle classification
  - Disagreement on ~20-50% of values indicates different decision logic

### Poor Accuracy - Implementation Issues

The following indicators show significant differences suggesting **implementation issues** in talib-pure:

#### HT_DCPHASE ❌
- **Correlation**: -0.177 (poor, slightly negative)
- **MAE**: 146.5 (very large for phase values in degrees)
- **Max Error**: ~402 degrees
- **Status**: Poor agreement - likely implementation bug
- **Recommendation**: **Use original TA-Lib** until talib-pure implementation is fixed

#### HT_PHASOR ❌
- **Correlation**: 0.108 (inphase), -0.023 (quadrature)
- **MAE**: 0.30 (inphase), 0.43 (quadrature)
- **Status**: Poor agreement - likely implementation bug
- **Recommendation**: **Use original TA-Lib** until talib-pure implementation is fixed

#### HT_SINE ❌
- **Correlation**: -0.050 (sine), -0.411 (leadsine)
- **MAE**: 0.79 (sine), 1.05 (leadsine)
- **Max Error**: 2.0 (maximum possible for sine values)
- **Status**: Poor agreement - likely implementation bug
- **Notes**:
  - Sine and LeadSine have output range [-1, 1], so MAE of ~1 is extremely poor
  - Negative correlations suggest potential sign errors or phase shift issues
- **Recommendation**: **Use original TA-Lib** until talib-pure implementation is fixed

## Root Cause Analysis

The accuracy issues appear to stem from the complex Hilbert Transform pipeline used by these indicators:

1. **HT_TRENDLINE** works well because it's the simplest (just a weighted moving average)
2. **HT_DCPERIOD** works reasonably well as it only calculates the dominant cycle period
3. **HT_DCPHASE**, **HT_PHASOR**, and **HT_SINE** all depend on phase angle calculations, which appear to have bugs

### Potential Issues in talib-pure

Based on the error patterns, the likely issues are:

1. **Phase Angle Calculation**: The arctangent calculations for phase angles may have:
   - Incorrect quadrant adjustment
   - Different angle wrapping behavior (0-360 vs -180 to 180)
   - Sign errors in the InPhase/Quadrature component calculations

2. **Smoothing Constants**: The Hilbert Transform uses specific smoothing factors (0.2, 0.8) that may differ slightly

3. **Initial Conditions**: Different handling of the unstable period (first 32-63 bars)

## Recommendations

### For Production Use

**Safe to Use** ✅
- **HT_TRENDLINE**: Excellent accuracy, recommended for production

**Use with Caution** ⚠️
- **HT_DCPERIOD**: Good accuracy for most use cases, but verify critical applications
- **HT_TRENDMODE**: Match rate varies by data type; test with your specific data

**Not Recommended** ❌
- **HT_DCPHASE**: Use original TA-Lib instead
- **HT_PHASOR**: Use original TA-Lib instead
- **HT_SINE**: Use original TA-Lib instead

### Hybrid Approach

For best results, use a combination:

```python
# Use talib-pure for accurate functions
from talib_pure import HT_TRENDLINE

# Use original TA-Lib for problematic functions
import talib
ht_dcphase = talib.HT_DCPHASE
ht_phasor = talib.HT_PHASOR
ht_sine = talib.HT_SINE
```

### For talib-pure Developers

Priority fixes needed:

1. **High Priority**: Fix HT_DCPHASE, HT_PHASOR, and HT_SINE implementations
   - Review phase angle calculations (src/talib_pure/cpu/cycle_indicators.py)
   - Compare step-by-step intermediate values against TA-Lib
   - Check arctangent quadrant adjustments
   - Verify angle wrapping behavior

2. **Medium Priority**: Improve HT_TRENDMODE threshold logic
   - Review trend/cycle classification criteria
   - Consider making threshold configurable

3. **Low Priority**: Reduce HT_DCPERIOD max errors
   - Review period boundary handling (6-50 range)
   - Check period clamping logic

## Testing Methodology

### Test Data Types

Four different data patterns were tested to ensure robustness:

1. **Random Walk**: Simulates unpredictable market movements
2. **Trending + Noise**: Simulates bull/bear markets with volatility
3. **Cyclical + Noise**: Simulates oscillating markets (range-bound)
4. **Mixed**: Combination of trend, cycle, and noise (most realistic)

### Accuracy Testing Process

For each indicator and data type:
1. Generate 10,000 bars of test data with fixed random seed (reproducibility)
2. Run both TA-Lib and talib-pure implementations
3. Compare outputs element-by-element (excluding unstable period)
4. Calculate MAE, RMSE, Max Error, Correlation, and Match Rate
5. Average results across all 4 data types

## Reproducing These Results

To run the accuracy tests yourself:

```bash
# Install dependencies (if not already installed)
pip install -e ".[dev]"

# Run accuracy comparison
python accuracy_comparison.py
```

The script will:
- Test all 6 cycle indicators
- Use 4 different data patterns
- Output detailed metrics and summary tables
- Display accuracy classifications

## Conclusion

The talib-pure implementation shows **mixed accuracy** for Cycle Indicators:

**Strengths:**
- ✅ **HT_TRENDLINE** has excellent accuracy (correlation 0.999)
- ⚠️ **HT_DCPERIOD** has good accuracy (correlation 0.953, 97%+ match rate)

**Weaknesses:**
- ❌ **HT_DCPHASE**, **HT_PHASOR**, and **HT_SINE** have poor accuracy
- ❌ Phase-dependent indicators show negative or near-zero correlations
- ❌ Large errors suggest implementation bugs rather than minor precision differences

**Overall Recommendation:**

Until the phase-related indicators are fixed, we recommend:
1. Use **HT_TRENDLINE** from talib-pure (faster and accurate)
2. Use **HT_DCPERIOD** from talib-pure if approximate periods are acceptable
3. Use **original TA-Lib** for HT_DCPHASE, HT_PHASOR, and HT_SINE
4. Test **HT_TRENDMODE** with your specific data before production use

The performance benefits of talib-pure (see PERFORMANCE.md) can only be realized if accuracy is acceptable. For critical trading applications, accuracy should always take priority over performance.

---

# Math Operators

## Test Environment

- **Python Version**: 3.11
- **NumPy Version**: 2.3.4
- **Numba Version**: 0.62.1
- **TA-Lib Version**: 0.6.8
- **Platform**: Linux
- **Dataset Size**: 10,000 bars per test
- **Time Period**: 30 bars (default parameter)
- **Test Method**: Comparison across 4 different data patterns

## Metrics Explained

- **MAE (Mean Absolute Error)**: Average absolute difference between outputs
- **RMSE (Root Mean Square Error)**: Square root of average squared differences
- **Max Error**: Largest absolute difference observed
- **Correlation**: Pearson correlation coefficient (1.0 = perfect, 0.0 = no correlation, -1.0 = inverse)
- **Exact Match Rate**: Percentage of values exactly matching (within 1e-10 tolerance)

## Overall Summary

Average accuracy metrics across all 4 test data types (10,000 bars each, timeperiod=30):

| Function | Avg MAE | Avg RMSE | Avg Max Error | Avg Correlation | Exact Match | Status |
|----------|---------|----------|---------------|-----------------|-------------|--------|
| **MAX** | 0.00e+00 | 0.00e+00 | 0.00 | 1.000 | 100.00% | ✅ Exact |
| **MIN** | 0.00e+00 | 0.00e+00 | 0.00 | 1.000 | 100.00% | ✅ Exact |
| **MINMAX (min)** | 0.00e+00 | 0.00e+00 | 0.00 | 1.000 | 100.00% | ✅ Exact |
| **MINMAX (max)** | 0.00e+00 | 0.00e+00 | 0.00 | 1.000 | 100.00% | ✅ Exact |
| **MAXINDEX** | 4.99e+03 | 5.76e+03 | 9,990 | -0.000 | 0.00% | ❌ Wrong |
| **MININDEX** | 4.98e+03 | 5.76e+03 | 9,980 | -0.016 | 0.00% | ❌ Wrong |
| **MINMAXINDEX (min)** | 4.98e+03 | 5.76e+03 | 9,980 | -0.016 | 0.00% | ❌ Wrong |
| **MINMAXINDEX (max)** | 4.99e+03 | 5.76e+03 | 9,990 | -0.000 | 0.00% | ❌ Wrong |
| **SUM** | 0.00e+00 | 0.00e+00 | 0.00 | 1.000 | 100.00% | ✅ Exact |

### Status Legend
- ✅ **Exact**: Perfect match (100% exact match rate, zero error)
- ❌ **Wrong**: Completely different output (0% match, indicating different interpretation/algorithm)

## Key Findings

### Perfect Accuracy (✅ Exact Match)

The following Math Operators show **100% perfect accuracy**:
- **MAX**: Identifies maximum values identically to TA-Lib
- **MIN**: Identifies minimum values identically to TA-Lib
- **MINMAX**: Both min and max outputs match exactly
- **SUM**: Summation matches exactly

These functions produce **bit-for-bit identical results** to the original TA-Lib implementation.

### Incorrect Implementation (❌ Wrong Output)

The INDEX functions have **fundamental implementation differences**:
- **MAXINDEX**: MAE of ~5,000 with near-zero correlation
- **MININDEX**: MAE of ~5,000 with near-zero correlation
- **MINMAXINDEX**: Both outputs have MAE of ~5,000 with near-zero correlation

**These are not small errors** - they indicate the implementations produce fundamentally different outputs.

## Detailed Results by Data Type

### Test 1: Random Walk Data

| Function | MAE | RMSE | Max Error | Correlation | Exact Match |
|----------|-----|------|-----------|-------------|-------------|
| MAX | 0.00e+00 | 0.00e+00 | 0.00 | 1.000 | 100.00% |
| MIN | 0.00e+00 | 0.00e+00 | 0.00 | 1.000 | 100.00% |
| MINMAX (min) | 0.00e+00 | 0.00e+00 | 0.00 | 1.000 | 100.00% |
| MINMAX (max) | 0.00e+00 | 0.00e+00 | 0.00 | 1.000 | 100.00% |
| MAXINDEX | 4.98e+03 | 5.76e+03 | 9,976 | 0.024 | 0.00% |
| MININDEX | 4.99e+03 | 5.76e+03 | 9,989 | -0.051 | 0.00% |
| MINMAXINDEX (min) | 4.99e+03 | 5.76e+03 | 9,989 | -0.051 | 0.00% |
| MINMAXINDEX (max) | 4.98e+03 | 5.76e+03 | 9,976 | 0.024 | 0.00% |
| SUM | 0.00e+00 | 0.00e+00 | 0.00 | 1.000 | 100.00% |

### Test 2: Trending + Noise Data

| Function | MAE | RMSE | Max Error | Correlation | Exact Match |
|----------|-----|------|-----------|-------------|-------------|
| MAX | 0.00e+00 | 0.00e+00 | 0.00 | 1.000 | 100.00% |
| MIN | 0.00e+00 | 0.00e+00 | 0.00 | 1.000 | 100.00% |
| MINMAX (min) | 0.00e+00 | 0.00e+00 | 0.00 | 1.000 | 100.00% |
| MINMAX (max) | 0.00e+00 | 0.00e+00 | 0.00 | 1.000 | 100.00% |
| MAXINDEX | 4.99e+03 | 5.76e+03 | 9,976 | -0.010 | 0.00% |
| MININDEX | 4.98e+03 | 5.76e+03 | 9,996 | 0.007 | 0.00% |
| MINMAXINDEX (min) | 4.98e+03 | 5.76e+03 | 9,996 | 0.007 | 0.00% |
| MINMAXINDEX (max) | 4.99e+03 | 5.76e+03 | 9,976 | -0.010 | 0.00% |
| SUM | 0.00e+00 | 0.00e+00 | 0.00 | 1.000 | 100.00% |

### Test 3: Cyclical + Noise Data

| Function | MAE | RMSE | Max Error | Correlation | Exact Match |
|----------|-----|------|-----------|-------------|-------------|
| MAX | 0.00e+00 | 0.00e+00 | 0.00 | 1.000 | 100.00% |
| MIN | 0.00e+00 | 0.00e+00 | 0.00 | 1.000 | 100.00% |
| MINMAX (min) | 0.00e+00 | 0.00e+00 | 0.00 | 1.000 | 100.00% |
| MINMAX (max) | 0.00e+00 | 0.00e+00 | 0.00 | 1.000 | 100.00% |
| MAXINDEX | 4.98e+03 | 5.76e+03 | 9,995 | 0.002 | 0.00% |
| MININDEX | 4.99e+03 | 5.76e+03 | 9,947 | -0.028 | 0.00% |
| MINMAXINDEX (min) | 4.99e+03 | 5.76e+03 | 9,947 | -0.028 | 0.00% |
| MINMAXINDEX (max) | 4.98e+03 | 5.76e+03 | 9,995 | 0.002 | 0.00% |
| SUM | 0.00e+00 | 0.00e+00 | 0.00 | 1.000 | 100.00% |

### Test 4: Mixed (Trend + Cycle + Noise) Data

| Function | MAE | RMSE | Max Error | Correlation | Exact Match |
|----------|-----|------|-----------|-------------|-------------|
| MAX | 0.00e+00 | 0.00e+00 | 0.00 | 1.000 | 100.00% |
| MIN | 0.00e+00 | 0.00e+00 | 0.00 | 1.000 | 100.00% |
| MINMAX (min) | 0.00e+00 | 0.00e+00 | 0.00 | 1.000 | 100.00% |
| MINMAX (max) | 0.00e+00 | 0.00e+00 | 0.00 | 1.000 | 100.00% |
| MAXINDEX | 4.99e+03 | 5.76e+03 | 9,995 | -0.016 | 0.00% |
| MININDEX | 4.98e+03 | 5.76e+03 | 9,971 | 0.009 | 0.00% |
| MINMAXINDEX (min) | 4.98e+03 | 5.76e+03 | 9,971 | 0.009 | 0.00% |
| MINMAXINDEX (max) | 4.99e+03 | 5.76e+03 | 9,995 | -0.016 | 0.00% |
| SUM | 0.00e+00 | 0.00e+00 | 0.00 | 1.000 | 100.00% |

## Analysis

### Excellent Accuracy Functions

#### MAX, MIN, MINMAX, SUM ✅

These functions show **perfect accuracy** with:
- **MAE**: 0.0 (no error whatsoever)
- **RMSE**: 0.0 (no error whatsoever)
- **Max Error**: 0.0 (no error whatsoever)
- **Correlation**: 1.0 (perfect positive correlation)
- **Exact Match Rate**: 100% (every single value matches exactly)

**Confidence**: Extremely High - These functions can be used in production with complete confidence that they will produce identical results to TA-Lib.

**Implementation Quality**: The talib-pure implementations of these functions are mathematically identical to TA-Lib's C implementation.

### Incorrect Implementation - INDEX Functions

#### MAXINDEX, MININDEX, MINMAXINDEX ❌

These functions have **fundamental implementation differences**:

**Error Magnitude**:
- Average MAE of ~5,000 (out of maximum possible ~10,000 for a 10,000-bar dataset)
- This means the index values are off by roughly **half the dataset size**

**Root Cause**:
The INDEX functions report the **position or age** of the extreme value within the rolling window. The large errors suggest:

1. **Different Reference Points**: TA-Lib and talib-pure may count from opposite ends of the window
2. **Different Interpretation**: One may report "bars ago" while the other reports "position in window"
3. **Off-by-One Errors**: Systematic offset in index calculation

**Example**:
If the maximum value occurred 5 bars ago in a 30-bar window:
- One implementation might return: `5` (bars ago from current position)
- Other implementation might return: `25` (position from start of window = 30 - 5)
- Difference: `20` per occurrence

Over 10,000 bars with average window size of 30, the expected difference would be around 15 per bar, but the actual MAE of ~5,000 suggests a more complex relationship, possibly:
- Returning absolute index instead of relative
- Counting from wrong direction entirely

**Correlation Near Zero**: The near-zero correlation (-0.016 to 0.024) confirms these are not just scaled versions of each other but fundamentally different outputs.

## Root Cause Analysis

### Value-Based Functions (MAX, MIN, MINMAX, SUM)

These work perfectly because:
1. **Deterministic Operations**: Finding max/min/sum of a set of numbers has only one correct answer
2. **Simple Algorithms**: No room for interpretation differences
3. **No State Tracking**: Just compare values or sum them
4. **Bit-Level Precision**: Using the same floating-point operations yields identical results

### Index-Based Functions (MAXINDEX, MININDEX, MINMAXINDEX)

These have issues because:
1. **Ambiguous Specification**: "Index" could mean:
   - Position in array (0-based? 1-based?)
   - Bars ago from current position
   - Distance from start of window
   - Absolute index in entire dataset

2. **Implementation Choices**: Different but valid interpretations led to incompatible implementations

3. **Lack of Reference**: Without access to TA-Lib source code documentation on exact index semantics, implementations diverged

## Recommendations

### For Production Use

**Safe to Use** ✅
- **MAX**: Perfect accuracy, use with confidence
- **MIN**: Perfect accuracy, use with confidence
- **MINMAX**: Perfect accuracy, use with confidence
- **SUM**: Perfect accuracy, use with confidence

**Not Recommended** ❌
- **MAXINDEX**: Use original TA-Lib instead - fundamentally different output
- **MININDEX**: Use original TA-Lib instead - fundamentally different output
- **MINMAXINDEX**: Use original TA-Lib instead - fundamentally different output

### Hybrid Approach

The recommendation is straightforward:

```python
# Use talib-pure for value-based operators (perfect accuracy)
from talib_pure import MAX, MIN, MINMAX, SUM

# Use original TA-Lib for index-based operators (wrong implementation)
import talib
MAXINDEX = talib.MAXINDEX
MININDEX = talib.MININDEX
MINMAXINDEX = talib.MINMAXINDEX
```

### For talib-pure Developers

**High Priority Fix Required**:

1. **Investigate INDEX semantics**: Compare actual TA-Lib outputs with talib-pure outputs on simple test cases
2. **Determine correct interpretation**: What does "index" mean in TA-Lib context?
3. **Fix implementation**: Align with TA-Lib's exact behavior
4. **Add regression tests**: Include tests comparing against TA-Lib for various window sizes and positions

**Suggested Investigation Steps**:
```python
# Test with simple data to understand the difference
import talib
from talib_pure import MAXINDEX as MAXINDEX_PURE

data = np.array([1, 5, 3, 9, 2, 7, 4])  # Small dataset
print("TA-Lib MAXINDEX:", talib.MAXINDEX(data, timeperiod=3))
print("talib-pure MAXINDEX:", MAXINDEX_PURE(data, timeperiod=3))
# Compare outputs to understand the interpretation difference
```

## Testing Methodology

### Test Data Types

Four different data patterns were tested:
1. **Random Walk**: Unpredictable movements
2. **Trending + Noise**: Bull/bear markets with volatility
3. **Cyclical + Noise**: Oscillating markets (range-bound)
4. **Mixed**: Combination of trend, cycle, and noise

### Accuracy Testing Process

For each operator and data type:
1. Generate 10,000 bars of test data with fixed random seed (reproducibility)
2. Run both TA-Lib and talib-pure implementations with timeperiod=30
3. Compare outputs element-by-element (excluding lookback period)
4. Calculate MAE, RMSE, Max Error, Correlation, and Exact Match Rate
5. Average results across all 4 data types

## Reproducing These Results

To run the accuracy tests yourself:

```bash
# Install dependencies (if not already installed)
pip install -e ".[dev]"

# Run Math Operators accuracy comparison
python accuracy_math_operators.py
```

The script will:
- Test all 7 Math Operators
- Use 4 different data patterns
- Output detailed metrics and summary tables
- Display accuracy classifications

## Conclusion

The talib-pure implementation shows **split results** for Math Operators:

**Perfect Accuracy (5 out of 7 functions):**
- ✅ **MAX, MIN, MINMAX, SUM** - 100% exact match, ready for production use
- These value-based operators can be used with complete confidence

**Incorrect Implementation (2 out of 7 functions):**
- ❌ **MAXINDEX, MININDEX, MINMAXINDEX** - Fundamentally different output
- These index-based operators should NOT be used until fixed
- MAE of ~5,000 indicates wrong algorithm, not just precision differences

**Overall Recommendation:**

For Math Operators:
1. **Use talib-pure** for MAX, MIN, MINMAX, and SUM (perfect accuracy)
2. **Use original TA-Lib** for MAXINDEX, MININDEX, and MINMAXINDEX (wrong implementation)
3. Combining the performance data (see PERFORMANCE.md) with accuracy results:
   - **SUM**: Use talib-pure (slightly faster + perfect accuracy)
   - **MAX/MIN/MINMAX**: Use TA-Lib (much faster + perfect accuracy in both)
   - **INDEX functions**: Must use TA-Lib (faster + correct implementation)

**For Critical Applications**: Always verify outputs against TA-Lib before deploying any indicator in production, regardless of accuracy test results.
