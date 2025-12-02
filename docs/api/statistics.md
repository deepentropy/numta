# Statistical Functions

Statistical analysis functions for price data analysis.

## Overview

These functions provide statistical analysis capabilities commonly used in technical analysis:

- Linear regression for trend analysis
- Standard deviation and variance for volatility measurement
- Correlation and beta for relative performance analysis

## Available Functions

### LINEARREG - Linear Regression

Calculates the linear regression line for a given period.

```python
from numta.api.statistic_functions import LINEARREG

# Calculate linear regression value
lr = LINEARREG(close, timeperiod=14)
```

### STDDEV - Standard Deviation

Calculates the standard deviation of prices over a period.

```python
from numta.api.statistic_functions import STDDEV

# Calculate standard deviation
std = STDDEV(close, timeperiod=14, nbdev=1.0)
```

### VAR - Variance

Calculates the variance of prices over a period.

```python
from numta.api.statistic_functions import VAR

# Calculate variance
var = VAR(close, timeperiod=14, nbdev=1.0)
```

### CORREL - Pearson Correlation

Calculates the correlation coefficient between two price series.

```python
from numta.api.statistic_functions import CORREL

# Calculate correlation between two series
corr = CORREL(series1, series2, timeperiod=30)
```

### BETA - Beta Coefficient

Calculates the beta coefficient of a stock relative to a market index.

```python
from numta.api.statistic_functions import BETA

# Calculate beta
beta = BETA(stock_prices, market_prices, timeperiod=30)
```

### TSF - Time Series Forecast

Calculates the time series forecast based on linear regression.

```python
from numta.api.statistic_functions import TSF

# Calculate time series forecast
tsf = TSF(close, timeperiod=14)
```

## Usage Example

```python
import numpy as np
from numta.api import statistic_functions as stats

# Generate sample data
np.random.seed(42)
close = np.cumsum(np.random.randn(100)) + 100

# Calculate various statistics
lr = stats.LINEARREG(close, timeperiod=20)
std = stats.STDDEV(close, timeperiod=20)
var = stats.VAR(close, timeperiod=20)

print(f"Linear Regression: {lr[-1]:.2f}")
print(f"Standard Deviation: {std[-1]:.2f}")
print(f"Variance: {var[-1]:.2f}")
```
