# Pattern Recognition

numta provides comprehensive pattern recognition capabilities including candlestick patterns and chart patterns.

## Candlestick Patterns

numta includes 60+ candlestick pattern recognition functions. All candlestick functions return:

- `+100` for bullish patterns
- `-100` for bearish patterns
- `0` for no pattern detected

### Common Candlestick Patterns

#### CDLDOJI - Doji

```python
from numta import CDLDOJI

result = CDLDOJI(open_, high, low, close)
# Returns +100 where doji pattern is found
```

#### CDLENGULFING - Engulfing Pattern

```python
from numta import CDLENGULFING

result = CDLENGULFING(open_, high, low, close)
# Returns +100 for bullish engulfing, -100 for bearish engulfing
```

#### CDLHAMMER - Hammer

```python
from numta import CDLHAMMER

result = CDLHAMMER(open_, high, low, close)
# Returns +100 where hammer pattern is found
```

#### CDLMORNINGSTAR - Morning Star

```python
from numta import CDLMORNINGSTAR

result = CDLMORNINGSTAR(open_, high, low, close, penetration=0.3)
# Returns +100 where morning star pattern is found
```

#### CDLEVENINGSTAR - Evening Star

```python
from numta import CDLEVENINGSTAR

result = CDLEVENINGSTAR(open_, high, low, close, penetration=0.3)
# Returns -100 where evening star pattern is found
```

### Available Candlestick Patterns

| Function | Pattern Name |
|----------|--------------|
| CDL2CROWS | Two Crows |
| CDL3BLACKCROWS | Three Black Crows |
| CDL3INSIDE | Three Inside Up/Down |
| CDL3LINESTRIKE | Three-Line Strike |
| CDL3OUTSIDE | Three Outside Up/Down |
| CDL3STARSINSOUTH | Three Stars In The South |
| CDL3WHITESOLDIERS | Three Advancing White Soldiers |
| CDLABANDONEDBABY | Abandoned Baby |
| CDLADVANCEBLOCK | Advance Block |
| CDLBELTHOLD | Belt-hold |
| CDLBREAKAWAY | Breakaway |
| CDLCLOSINGMARUBOZU | Closing Marubozu |
| CDLCONCEALBABYSWALL | Concealing Baby Swallow |
| CDLCOUNTERATTACK | Counterattack |
| CDLDARKCLOUDCOVER | Dark Cloud Cover |
| CDLDOJI | Doji |
| CDLDOJISTAR | Doji Star |
| CDLDRAGONFLYDOJI | Dragonfly Doji |
| CDLENGULFING | Engulfing Pattern |
| CDLEVENINGDOJISTAR | Evening Doji Star |
| CDLEVENINGSTAR | Evening Star |
| CDLGAPSIDESIDEWHITE | Up/Down-gap side-by-side white lines |
| CDLGRAVESTONEDOJI | Gravestone Doji |
| CDLHAMMER | Hammer |
| CDLHANGINGMAN | Hanging Man |
| CDLHARAMI | Harami Pattern |
| CDLHARAMICROSS | Harami Cross Pattern |
| CDLHIGHWAVE | High-Wave Candle |
| CDLHIKKAKE | Hikkake Pattern |
| CDLHIKKAKEMOD | Modified Hikkake Pattern |
| CDLHOMINGPIGEON | Homing Pigeon |
| CDLIDENTICAL3CROWS | Identical Three Crows |
| CDLINNECK | In-Neck Pattern |
| CDLINVERTEDHAMMER | Inverted Hammer |
| CDLKICKING | Kicking |
| CDLKICKINGBYLENGTH | Kicking by Length |
| CDLLADDERBOTTOM | Ladder Bottom |
| CDLLONGLEGGEDDOJI | Long Legged Doji |
| CDLLONGLINE | Long Line Candle |
| CDLMARUBOZU | Marubozu |
| CDLMATCHINGLOW | Matching Low |
| CDLMATHOLD | Mat Hold |
| CDLMORNINGDOJISTAR | Morning Doji Star |
| CDLMORNINGSTAR | Morning Star |
| CDLONNECK | On-Neck Pattern |
| CDLPIERCING | Piercing Pattern |
| CDLRICKSHAWMAN | Rickshaw Man |
| CDLRISEFALL3METHODS | Rising/Falling Three Methods |
| CDLSEPARATINGLINES | Separating Lines |
| CDLSHOOTINGSTAR | Shooting Star |
| CDLSHORTLINE | Short Line Candle |
| CDLSPINNINGTOP | Spinning Top |
| CDLSTALLEDPATTERN | Stalled Pattern |
| CDLSTICKSANDWICH | Stick Sandwich |
| CDLTAKURI | Takuri |
| CDLTASUKIGAP | Tasuki Gap |
| CDLTHRUSTING | Thrusting Pattern |
| CDLTRISTAR | Tristar Pattern |
| CDLUNIQUE3RIVER | Unique 3 River |
| CDLUPSIDEGAP2CROWS | Upside Gap Two Crows |
| CDLXSIDEGAP3METHODS | Upside/Downside Gap Three Methods |

## Chart Patterns

### Head and Shoulders

```python
from numta import detect_head_shoulders

patterns = detect_head_shoulders(highs, lows, order=5)
```

### Double Top/Bottom

```python
from numta import detect_double_top

patterns = detect_double_top(highs, lows)
```

### Triangle Patterns

```python
from numta import detect_triangle

patterns = detect_triangle(highs, lows)
```

### Wedge Patterns

```python
from numta import detect_wedge

patterns = detect_wedge(highs, lows)
```

### Flag Patterns

```python
from numta import detect_flag

patterns = detect_flag(highs, lows)
```

## Pandas Integration

```python
import pandas as pd
import numta

df = pd.DataFrame({
    'open': open_,
    'high': high,
    'low': low,
    'close': close
})

# Add candlestick pattern columns
df.ta.cdldoji(append=True)
df.ta.cdlengulfing(append=True)
df.ta.cdlhammer(append=True)

# Find all patterns
patterns = df.ta.find_patterns(pattern_type='all')

# Find harmonic patterns
harmonics = df.ta.find_harmonic_patterns()
```

## Usage Example

```python
import numpy as np
from numta import CDLDOJI, CDLENGULFING, CDLHAMMER

# Sample OHLC data
open_ = np.array([100, 102, 103, 101, 104])
high = np.array([105, 107, 108, 106, 109])
low = np.array([99, 101, 102, 100, 103])
close = np.array([104, 103, 104, 105, 108])

# Detect patterns
doji = CDLDOJI(open_, high, low, close)
engulfing = CDLENGULFING(open_, high, low, close)
hammer = CDLHAMMER(open_, high, low, close)

# Find pattern occurrences
doji_indices = np.where(doji != 0)[0]
print(f"Doji patterns at indices: {doji_indices}")
```
