# Price Transform

Price transformation functions that convert OHLC data into derived price values.

## Overview

Price transform functions calculate various representations of price that are commonly used in other indicators.

## Available Functions

### MEDPRICE - Median Price

Calculates the median price: (High + Low) / 2

```python
from numta.api.price_transform import MEDPRICE

median = MEDPRICE(high, low)
```

### TYPPRICE - Typical Price

Calculates the typical price: (High + Low + Close) / 3

```python
from numta.api.price_transform import TYPPRICE

typical = TYPPRICE(high, low, close)
```

### WCLPRICE - Weighted Close Price

Calculates the weighted close price: (High + Low + 2*Close) / 4

```python
from numta.api.price_transform import WCLPRICE

weighted = WCLPRICE(high, low, close)
```

### MIDPOINT - Midpoint over Period

Calculates the midpoint of the highest and lowest values over a period.

```python
from numta.api.price_transform import MIDPOINT

midpoint = MIDPOINT(close, timeperiod=14)
```

### MIDPRICE - Midpoint Price over Period

Calculates the midpoint of the highest high and lowest low over a period.

```python
from numta.api.price_transform import MIDPRICE

midprice = MIDPRICE(high, low, timeperiod=14)
```

## Usage Example

```python
import numpy as np
from numta.api import price_transform as pt

# Sample OHLC data
high = np.array([105, 107, 106, 108, 110])
low = np.array([100, 102, 101, 103, 105])
close = np.array([103, 105, 104, 106, 108])

# Calculate price transforms
median = pt.MEDPRICE(high, low)
typical = pt.TYPPRICE(high, low, close)
weighted = pt.WCLPRICE(high, low, close)

print(f"Median Price: {median}")
print(f"Typical Price: {typical}")
print(f"Weighted Close: {weighted}")
```
