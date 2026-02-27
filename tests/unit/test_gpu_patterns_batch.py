"""Tests for GPU batch candlestick pattern recognition — verify GPU matches CPU."""
import numpy as np
import pytest

pytestmark = pytest.mark.cuda

NUM_TICKERS = 50
NUM_BARS = 200

# All standard pattern names (no penetration parameter)
STANDARD_PATTERNS = [
    "CDL2CROWS", "CDL3BLACKCROWS", "CDL3INSIDE", "CDL3OUTSIDE",
    "CDL3STARSINSOUTH", "CDL3WHITESOLDIERS",
    "CDLABANDONEDBABY", "CDLADVANCEBLOCK", "CDLBELTHOLD", "CDLBREAKAWAY",
    "CDLCLOSINGMARUBOZU", "CDLCONCEALBABYSWALL", "CDLCOUNTERATTACK",
    "CDLDARKCLOUDCOVER", "CDLDOJI", "CDLDOJISTAR",
    "CDLDRAGONFLYDOJI", "CDLENGULFING", "CDLEVENINGDOJISTAR", "CDLEVENINGSTAR",
    "CDLGAPSIDESIDEWHITE", "CDLGRAVESTONEDOJI",
    "CDLHAMMER", "CDLHANGINGMAN", "CDLHARAMI", "CDLHARAMICROSS",
    "CDLHIGHWAVE", "CDLHIKKAKE", "CDLHIKKAKEMOD", "CDLHOMINGPIGEON",
    "CDLIDENTICAL3CROWS", "CDLINNECK",
    "CDLINVERTEDHAMMER", "CDLKICKING", "CDLKICKINGBYLENGTH", "CDLLADDERBOTTOM",
    "CDLLONGLEGGEDDOJI", "CDLLONGLINE",
    "CDLMARUBOZU", "CDLMATCHINGLOW",
    "CDLONNECK", "CDLPIERCING",
    "CDLRICKSHAWMAN", "CDLRISEFALL3METHODS", "CDLSEPARATINGLINES", "CDLSHOOTINGSTAR",
    "CDLSHORTLINE", "CDLSPINNINGTOP", "CDLSTALLEDPATTERN", "CDLSTICKSANDWICH",
    "CDLTAKURI", "CDLTASUKIGAP",
    "CDLTHRUSTING", "CDLTRISTAR", "CDLUNIQUE3RIVER", "CDLUPSIDEGAP2CROWS",
    "CDLXSIDEGAP3METHODS",
]

# Patterns with penetration parameter: (name, default_penetration)
PENETRATION_PATTERNS = [
    ("CDLMATHOLD", 0.5),
    ("CDLMORNINGDOJISTAR", 0.3),
    ("CDLMORNINGSTAR", 0.3),
]


@pytest.fixture(scope="module")
def ohlc_2d():
    np.random.seed(42)
    close = np.random.uniform(50, 150, (NUM_TICKERS, NUM_BARS))
    open_ = close + np.random.uniform(-3, 3, (NUM_TICKERS, NUM_BARS))
    high = np.maximum(open_, close) + np.random.uniform(0, 3, (NUM_TICKERS, NUM_BARS))
    low = np.minimum(open_, close) - np.random.uniform(0, 3, (NUM_TICKERS, NUM_BARS))
    return open_, high, low, close


def _get_fn(module_path, name):
    """Import a function by name from a module."""
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, name)


@pytest.mark.parametrize("pattern_name", STANDARD_PATTERNS)
def test_standard_pattern(ohlc_2d, pattern_name):
    """Test a standard 4-input candlestick pattern: GPU batch vs CPU single-ticker."""
    open_, high, low, close = ohlc_2d
    cpu_fn = _get_fn("numta.api.pattern_recognition", pattern_name)
    gpu_fn = _get_fn("numta.api.batch", f"{pattern_name}_batch")

    gpu_result = gpu_fn(open_, high, low, close)
    for t in range(NUM_TICKERS):
        cpu_result = cpu_fn(open_[t], high[t], low[t], close[t])
        np.testing.assert_array_equal(
            gpu_result[t], cpu_result,
            err_msg=f"{pattern_name} ticker {t}"
        )


@pytest.mark.parametrize("pattern_name,penetration", PENETRATION_PATTERNS)
def test_penetration_pattern(ohlc_2d, pattern_name, penetration):
    """Test a candlestick pattern with penetration param: GPU batch vs CPU."""
    open_, high, low, close = ohlc_2d
    cpu_fn = _get_fn("numta.api.pattern_recognition", pattern_name)
    gpu_fn = _get_fn("numta.api.batch", f"{pattern_name}_batch")

    gpu_result = gpu_fn(open_, high, low, close, penetration=penetration)
    for t in range(NUM_TICKERS):
        cpu_result = cpu_fn(open_[t], high[t], low[t], close[t], penetration=penetration)
        np.testing.assert_array_equal(
            gpu_result[t], cpu_result,
            err_msg=f"{pattern_name} ticker {t}"
        )
