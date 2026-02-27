# GPU Batch Processing

Run any indicator on thousands of tickers simultaneously using NVIDIA CUDA. Each `*_batch` function accepts 2D arrays of shape `(num_tickers, num_bars)` and returns results of the same shape.

## Requirements

```bash
pip install "numta[gpu]"
```

- NVIDIA GPU with CUDA support
- Compatible NVIDIA drivers

## How It Works

Each batch function launches one CUDA thread per ticker. Every thread processes its ticker's full time series independently — the same sequential algorithm as the CPU version, but thousands of tickers run in parallel.

```python
import numpy as np
from numta import SMA_batch, RSI_batch, CDLDOJI_batch

# 2D arrays: (num_tickers, num_bars)
close = np.random.uniform(50, 150, (10000, 500))
high = close + np.random.uniform(0, 5, (10000, 500))
low = close - np.random.uniform(0, 5, (10000, 500))
open_ = close + np.random.uniform(-2, 2, (10000, 500))

# All 10,000 tickers processed at once
sma = SMA_batch(close, timeperiod=20)         # shape: (10000, 500)
rsi = RSI_batch(close, timeperiod=14)         # shape: (10000, 500)
doji = CDLDOJI_batch(open_, high, low, close)  # shape: (10000, 500)
```

## Checking CUDA Availability

```python
from numta import HAS_CUDA

if HAS_CUDA:
    from numta import SMA_batch
    result = SMA_batch(close_2d, timeperiod=20)
else:
    # Fall back to CPU loop
    from numta import SMA
    result = np.array([SMA(close_2d[t], timeperiod=20) for t in range(len(close_2d))])
```

## Multi-Output Indicators

Some indicators return multiple arrays:

```python
from numta import BBANDS_batch, MACD_batch, STOCH_batch

# Bollinger Bands: 3 outputs
upper, middle, lower = BBANDS_batch(close, timeperiod=20)

# MACD: 3 outputs
macd, signal, histogram = MACD_batch(close)

# Stochastic: 2 outputs
slowk, slowd = STOCH_batch(high, low, close)
```

## Available Functions

### Overlap Studies (12)

| Function | Parameters | Returns |
|----------|-----------|---------|
| `SMA_batch` | close, timeperiod | array |
| `EMA_batch` | close, timeperiod | array |
| `BBANDS_batch` | close, timeperiod, nbdevup, nbdevdn | (upper, middle, lower) |
| `DEMA_batch` | close, timeperiod | array |
| `KAMA_batch` | close, timeperiod | array |
| `WMA_batch` | close, timeperiod | array |
| `TEMA_batch` | close, timeperiod | array |
| `T3_batch` | close, timeperiod, vfactor | array |
| `TRIMA_batch` | close, timeperiod | array |
| `MAMA_batch` | close, fastlimit, slowlimit | (mama, fama) |
| `SAR_batch` | high, low, acceleration, maximum | array |
| `SAREXT_batch` | high, low, ... | array |

### Momentum Indicators (24)

| Function | Parameters | Returns |
|----------|-----------|---------|
| `RSI_batch` | close, timeperiod | array |
| `MACD_batch` | close, fastperiod, slowperiod, signalperiod | (macd, signal, hist) |
| `ADX_batch` | high, low, close, timeperiod | array |
| `ATR_batch` | high, low, close, timeperiod | array |
| `CCI_batch` | high, low, close, timeperiod | array |
| `CMO_batch` | close, timeperiod | array |
| `DX_batch` | high, low, close, timeperiod | array |
| `MOM_batch` | close, timeperiod | array |
| `ROC_batch` | close, timeperiod | array |
| `ROCP_batch` | close, timeperiod | array |
| `ROCR_batch` | close, timeperiod | array |
| `ROCR100_batch` | close, timeperiod | array |
| `WILLR_batch` | high, low, close, timeperiod | array |
| `STOCH_batch` | high, low, close, fastk_period, slowk_period, slowd_period | (slowk, slowd) |
| `STOCHF_batch` | high, low, close, fastk_period, fastd_period | (fastk, fastd) |
| `BOP_batch` | open, high, low, close | array |
| `AROON_batch` | high, low, timeperiod | (down, up) |
| `AROONOSC_batch` | high, low, timeperiod | array |
| `MFI_batch` | high, low, close, volume, timeperiod | array |
| `MINUS_DM_batch` | high, low, timeperiod | array |
| `PLUS_DM_batch` | high, low, timeperiod | array |
| `MINUS_DI_batch` | high, low, close, timeperiod | array |
| `PLUS_DI_batch` | high, low, close, timeperiod | array |
| `ULTOSC_batch` | high, low, close, timeperiod1, timeperiod2, timeperiod3 | array |

### Volatility Indicators (2)

| Function | Parameters | Returns |
|----------|-----------|---------|
| `TRANGE_batch` | high, low, close | array |
| `NATR_batch` | high, low, close, timeperiod | array |

### Volume Indicators (3)

| Function | Parameters | Returns |
|----------|-----------|---------|
| `AD_batch` | high, low, close, volume | array |
| `OBV_batch` | close, volume | array |
| `ADOSC_batch` | high, low, close, volume, fastperiod, slowperiod | array |

### Statistics (3)

| Function | Parameters | Returns |
|----------|-----------|---------|
| `STDDEV_batch` | close, timeperiod | array |
| `VAR_batch` | close, timeperiod | array |
| `TSF_batch` | close, timeperiod | array |

### Statistic Functions (6)

| Function | Parameters | Returns |
|----------|-----------|---------|
| `BETA_batch` | high, close, timeperiod | array |
| `CORREL_batch` | high, close, timeperiod | array |
| `LINEARREG_batch` | close, timeperiod | array |
| `LINEARREG_ANGLE_batch` | close, timeperiod | array |
| `LINEARREG_INTERCEPT_batch` | close, timeperiod | array |
| `LINEARREG_SLOPE_batch` | close, timeperiod | array |

### Math Operators (7)

| Function | Parameters | Returns |
|----------|-----------|---------|
| `MAX_batch` | close, timeperiod | array |
| `MAXINDEX_batch` | close, timeperiod | array |
| `MIN_batch` | close, timeperiod | array |
| `MININDEX_batch` | close, timeperiod | array |
| `MINMAX_batch` | close, timeperiod | (min, max) |
| `MINMAXINDEX_batch` | close, timeperiod | (minidx, maxidx) |
| `SUM_batch` | close, timeperiod | array |

### Price Transform (5)

| Function | Parameters | Returns |
|----------|-----------|---------|
| `MEDPRICE_batch` | high, low | array |
| `MIDPOINT_batch` | close, timeperiod | array |
| `MIDPRICE_batch` | high, low, timeperiod | array |
| `TYPPRICE_batch` | high, low, close | array |
| `WCLPRICE_batch` | high, low, close | array |

### Cycle Indicators (6)

| Function | Parameters | Returns |
|----------|-----------|---------|
| `HT_TRENDLINE_batch` | close | array |
| `HT_TRENDMODE_batch` | close | array |
| `HT_DCPERIOD_batch` | close | array |
| `HT_DCPHASE_batch` | close | array |
| `HT_PHASOR_batch` | close | (inphase, quadrature) |
| `HT_SINE_batch` | close | (sine, leadsine) |

### Candlestick Patterns (60)

All 60 candlestick patterns have GPU batch equivalents. Each accepts 2D OHLC arrays `(open, high, low, close)` and returns integer signals (+100, -100, 0).

**Standard patterns (57):** `CDL2CROWS_batch`, `CDL3BLACKCROWS_batch`, `CDL3INSIDE_batch`, `CDL3OUTSIDE_batch`, `CDL3STARSINSOUTH_batch`, `CDL3WHITESOLDIERS_batch`, `CDLABANDONEDBABY_batch`, `CDLADVANCEBLOCK_batch`, `CDLBELTHOLD_batch`, `CDLBREAKAWAY_batch`, `CDLCLOSINGMARUBOZU_batch`, `CDLCONCEALBABYSWALL_batch`, `CDLCOUNTERATTACK_batch`, `CDLDARKCLOUDCOVER_batch`, `CDLDOJI_batch`, `CDLDOJISTAR_batch`, `CDLDRAGONFLYDOJI_batch`, `CDLENGULFING_batch`, `CDLEVENINGDOJISTAR_batch`, `CDLEVENINGSTAR_batch`, `CDLGAPSIDESIDEWHITE_batch`, `CDLGRAVESTONEDOJI_batch`, `CDLHAMMER_batch`, `CDLHANGINGMAN_batch`, `CDLHARAMI_batch`, `CDLHARAMICROSS_batch`, `CDLHIGHWAVE_batch`, `CDLHIKKAKE_batch`, `CDLHIKKAKEMOD_batch`, `CDLHOMINGPIGEON_batch`, `CDLIDENTICAL3CROWS_batch`, `CDLINNECK_batch`, `CDLINVERTEDHAMMER_batch`, `CDLKICKING_batch`, `CDLKICKINGBYLENGTH_batch`, `CDLLADDERBOTTOM_batch`, `CDLLONGLEGGEDDOJI_batch`, `CDLLONGLINE_batch`, `CDLMARUBOZU_batch`, `CDLMATCHINGLOW_batch`, `CDLONNECK_batch`, `CDLPIERCING_batch`, `CDLRICKSHAWMAN_batch`, `CDLRISEFALL3METHODS_batch`, `CDLSEPARATINGLINES_batch`, `CDLSHOOTINGSTAR_batch`, `CDLSHORTLINE_batch`, `CDLSPINNINGTOP_batch`, `CDLSTALLEDPATTERN_batch`, `CDLSTICKSANDWICH_batch`, `CDLTAKURI_batch`, `CDLTASUKIGAP_batch`, `CDLTHRUSTING_batch`, `CDLTRISTAR_batch`, `CDLUNIQUE3RIVER_batch`, `CDLUPSIDEGAP2CROWS_batch`, `CDLXSIDEGAP3METHODS_batch`

**Patterns with penetration parameter (3):**

| Function | Default Penetration |
|----------|-------------------|
| `CDLMATHOLD_batch` | 0.5 |
| `CDLMORNINGDOJISTAR_batch` | 0.3 |
| `CDLMORNINGSTAR_batch` | 0.3 |
