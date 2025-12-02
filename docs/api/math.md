# Math Operators

Mathematical operations on price arrays.

## Overview

Math operators provide basic mathematical operations commonly used in technical analysis.

## Available Functions

### MAX - Highest Value

Returns the highest value over a specified period.

```python
from numta.api.math_operators import MAX

# Find highest value over period
highest = MAX(close, timeperiod=10)
```

### MIN - Lowest Value

Returns the lowest value over a specified period.

```python
from numta.api.math_operators import MIN

# Find lowest value over period
lowest = MIN(close, timeperiod=10)
```

### SUM - Summation

Returns the sum of values over a specified period.

```python
from numta.api.math_operators import SUM

# Calculate rolling sum
rolling_sum = SUM(close, timeperiod=10)
```

### MINMAX - Minimum and Maximum Values

Returns both the minimum and maximum values over a period.

```python
from numta.api.math_operators import MINMAX

# Get both min and max
min_vals, max_vals = MINMAX(close, timeperiod=10)
```

### MINMAXINDEX - Min/Max Index

Returns the indices of minimum and maximum values over a period.

```python
from numta.api.math_operators import MINMAXINDEX

# Get indices of min and max
min_idx, max_idx = MINMAXINDEX(close, timeperiod=10)
```

## Usage Example

```python
import numpy as np
from numta.api import math_operators as math

# Generate sample data
close = np.array([100, 102, 98, 105, 103, 107, 101, 110, 108, 112])

# Calculate various operations
highest = math.MAX(close, timeperiod=5)
lowest = math.MIN(close, timeperiod=5)
rolling_sum = math.SUM(close, timeperiod=5)

print(f"Highest (5): {highest}")
print(f"Lowest (5): {lowest}")
print(f"Sum (5): {rolling_sum}")
```
