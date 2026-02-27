"""Tests for GPU batch remaining indicators (Phase 4) — verify GPU matches CPU."""
import numpy as np
import pytest

pytestmark = pytest.mark.cuda

NUM_TICKERS = 50
NUM_BARS = 200


@pytest.fixture(scope="module")
def ohlcv_2d():
    np.random.seed(42)
    close = np.random.uniform(50, 150, (NUM_TICKERS, NUM_BARS))
    high = close + np.random.uniform(0, 5, (NUM_TICKERS, NUM_BARS))
    low = close - np.random.uniform(0, 5, (NUM_TICKERS, NUM_BARS))
    open_ = close + np.random.uniform(-2, 2, (NUM_TICKERS, NUM_BARS))
    volume = np.random.uniform(1e6, 1e8, (NUM_TICKERS, NUM_BARS))
    return open_, high, low, close, volume


def _check(gpu_result, cpu_fn, inputs, rtol=1e-10):
    for t in range(NUM_TICKERS):
        cpu = cpu_fn(*[inp[t] for inp in inputs])
        np.testing.assert_allclose(gpu_result[t], cpu, rtol=rtol, equal_nan=True,
                                   err_msg=f"Ticker {t}")


class TestVolatilityBatch:
    def test_trange(self, ohlcv_2d):
        from numta import TRANGE
        from numta.api.batch import TRANGE_batch
        _, high, low, close, _ = ohlcv_2d
        gpu = TRANGE_batch(high, low, close)
        _check(gpu, lambda h, l, c: TRANGE(h, l, c), [high, low, close])

    def test_natr(self, ohlcv_2d):
        from numta import NATR
        from numta.api.batch import NATR_batch
        _, high, low, close, _ = ohlcv_2d
        gpu = NATR_batch(high, low, close, timeperiod=14)
        _check(gpu, lambda h, l, c: NATR(h, l, c, timeperiod=14), [high, low, close])


class TestVolumeBatch:
    def test_ad(self, ohlcv_2d):
        from numta import AD
        from numta.api.batch import AD_batch
        _, high, low, close, volume = ohlcv_2d
        gpu = AD_batch(high, low, close, volume)
        _check(gpu, lambda h, l, c, v: AD(h, l, c, v), [high, low, close, volume])

    def test_obv(self, ohlcv_2d):
        from numta import OBV
        from numta.api.batch import OBV_batch
        _, _, _, close, volume = ohlcv_2d
        gpu = OBV_batch(close, volume)
        _check(gpu, lambda c, v: OBV(c, v), [close, volume])

    def test_adosc(self, ohlcv_2d):
        from numta import ADOSC
        from numta.api.batch import ADOSC_batch
        _, high, low, close, volume = ohlcv_2d
        gpu = ADOSC_batch(high, low, close, volume, fastperiod=3, slowperiod=10)
        _check(gpu, lambda h, l, c, v: ADOSC(h, l, c, v, fastperiod=3, slowperiod=10),
               [high, low, close, volume], rtol=1e-8)


class TestStatisticsBatch:
    def test_stddev(self, ohlcv_2d):
        from numta import STDDEV
        from numta.api.batch import STDDEV_batch
        _, _, _, close, _ = ohlcv_2d
        gpu = STDDEV_batch(close, timeperiod=14)
        _check(gpu, lambda c: STDDEV(c, timeperiod=14), [close])

    def test_var(self, ohlcv_2d):
        from numta import VAR
        from numta.api.batch import VAR_batch
        _, _, _, close, _ = ohlcv_2d
        gpu = VAR_batch(close, timeperiod=14)
        _check(gpu, lambda c: VAR(c, timeperiod=14), [close])

    def test_tsf(self, ohlcv_2d):
        from numta import TSF
        from numta.api.batch import TSF_batch
        _, _, _, close, _ = ohlcv_2d
        gpu = TSF_batch(close, timeperiod=14)
        _check(gpu, lambda c: TSF(c, timeperiod=14), [close])


class TestStatisticFunctionsBatch:
    def test_beta(self, ohlcv_2d):
        from numta import BETA
        from numta.api.batch import BETA_batch
        _, high, _, close, _ = ohlcv_2d
        gpu = BETA_batch(high, close, timeperiod=5)
        _check(gpu, lambda h, c: BETA(h, c, timeperiod=5), [high, close])

    def test_correl(self, ohlcv_2d):
        from numta import CORREL
        from numta.api.batch import CORREL_batch
        _, high, _, close, _ = ohlcv_2d
        gpu = CORREL_batch(high, close, timeperiod=14)
        _check(gpu, lambda h, c: CORREL(h, c, timeperiod=14), [high, close])

    def test_linearreg(self, ohlcv_2d):
        from numta import LINEARREG
        from numta.api.batch import LINEARREG_batch
        _, _, _, close, _ = ohlcv_2d
        gpu = LINEARREG_batch(close, timeperiod=14)
        _check(gpu, lambda c: LINEARREG(c, timeperiod=14), [close])

    def test_linearreg_angle(self, ohlcv_2d):
        from numta import LINEARREG_ANGLE
        from numta.api.batch import LINEARREG_ANGLE_batch
        _, _, _, close, _ = ohlcv_2d
        gpu = LINEARREG_ANGLE_batch(close, timeperiod=14)
        _check(gpu, lambda c: LINEARREG_ANGLE(c, timeperiod=14), [close])

    def test_linearreg_intercept(self, ohlcv_2d):
        from numta import LINEARREG_INTERCEPT
        from numta.api.batch import LINEARREG_INTERCEPT_batch
        _, _, _, close, _ = ohlcv_2d
        gpu = LINEARREG_INTERCEPT_batch(close, timeperiod=14)
        _check(gpu, lambda c: LINEARREG_INTERCEPT(c, timeperiod=14), [close])

    def test_linearreg_slope(self, ohlcv_2d):
        from numta import LINEARREG_SLOPE
        from numta.api.batch import LINEARREG_SLOPE_batch
        _, _, _, close, _ = ohlcv_2d
        gpu = LINEARREG_SLOPE_batch(close, timeperiod=14)
        _check(gpu, lambda c: LINEARREG_SLOPE(c, timeperiod=14), [close])


class TestMathOperatorsBatch:
    def test_max(self, ohlcv_2d):
        from numta import MAX
        from numta.api.batch import MAX_batch
        _, _, _, close, _ = ohlcv_2d
        gpu = MAX_batch(close, timeperiod=14)
        _check(gpu, lambda c: MAX(c, timeperiod=14), [close])

    def test_maxindex(self, ohlcv_2d):
        from numta import MAXINDEX
        from numta.api.batch import MAXINDEX_batch
        _, _, _, close, _ = ohlcv_2d
        gpu = MAXINDEX_batch(close, timeperiod=14)
        _check(gpu, lambda c: MAXINDEX(c, timeperiod=14), [close])

    def test_min(self, ohlcv_2d):
        from numta import MIN
        from numta.api.batch import MIN_batch
        _, _, _, close, _ = ohlcv_2d
        gpu = MIN_batch(close, timeperiod=14)
        _check(gpu, lambda c: MIN(c, timeperiod=14), [close])

    def test_minindex(self, ohlcv_2d):
        from numta import MININDEX
        from numta.api.batch import MININDEX_batch
        _, _, _, close, _ = ohlcv_2d
        gpu = MININDEX_batch(close, timeperiod=14)
        _check(gpu, lambda c: MININDEX(c, timeperiod=14), [close])

    def test_minmax(self, ohlcv_2d):
        from numta import MINMAX
        from numta.api.batch import MINMAX_batch
        _, _, _, close, _ = ohlcv_2d
        g_min, g_max = MINMAX_batch(close, timeperiod=14)
        for t in range(NUM_TICKERS):
            c_min, c_max = MINMAX(close[t], timeperiod=14)
            np.testing.assert_allclose(g_min[t], c_min, rtol=1e-10, equal_nan=True)
            np.testing.assert_allclose(g_max[t], c_max, rtol=1e-10, equal_nan=True)

    def test_minmaxindex(self, ohlcv_2d):
        from numta import MINMAXINDEX
        from numta.api.batch import MINMAXINDEX_batch
        _, _, _, close, _ = ohlcv_2d
        g_mini, g_maxi = MINMAXINDEX_batch(close, timeperiod=14)
        for t in range(NUM_TICKERS):
            c_mini, c_maxi = MINMAXINDEX(close[t], timeperiod=14)
            np.testing.assert_allclose(g_mini[t], c_mini, rtol=1e-10, equal_nan=True)
            np.testing.assert_allclose(g_maxi[t], c_maxi, rtol=1e-10, equal_nan=True)

    def test_sum(self, ohlcv_2d):
        from numta import SUM
        from numta.api.batch import SUM_batch
        _, _, _, close, _ = ohlcv_2d
        gpu = SUM_batch(close, timeperiod=14)
        _check(gpu, lambda c: SUM(c, timeperiod=14), [close])


class TestPriceTransformBatch:
    def test_medprice(self, ohlcv_2d):
        from numta import MEDPRICE
        from numta.api.batch import MEDPRICE_batch
        _, high, low, _, _ = ohlcv_2d
        gpu = MEDPRICE_batch(high, low)
        _check(gpu, lambda h, l: MEDPRICE(h, l), [high, low])

    def test_midpoint(self, ohlcv_2d):
        from numta import MIDPOINT
        from numta.api.batch import MIDPOINT_batch
        _, _, _, close, _ = ohlcv_2d
        gpu = MIDPOINT_batch(close, timeperiod=14)
        _check(gpu, lambda c: MIDPOINT(c, timeperiod=14), [close])

    def test_midprice(self, ohlcv_2d):
        from numta import MIDPRICE
        from numta.api.batch import MIDPRICE_batch
        _, high, low, _, _ = ohlcv_2d
        gpu = MIDPRICE_batch(high, low, timeperiod=14)
        _check(gpu, lambda h, l: MIDPRICE(h, l, timeperiod=14), [high, low])

    def test_typprice(self, ohlcv_2d):
        from numta import TYPPRICE
        from numta.api.batch import TYPPRICE_batch
        _, high, low, close, _ = ohlcv_2d
        gpu = TYPPRICE_batch(high, low, close)
        _check(gpu, lambda h, l, c: TYPPRICE(h, l, c), [high, low, close])

    def test_wclprice(self, ohlcv_2d):
        from numta import WCLPRICE
        from numta.api.batch import WCLPRICE_batch
        _, high, low, close, _ = ohlcv_2d
        gpu = WCLPRICE_batch(high, low, close)
        _check(gpu, lambda h, l, c: WCLPRICE(h, l, c), [high, low, close])


class TestCycleIndicatorsBatch:
    def test_ht_trendline(self, ohlcv_2d):
        from numta import HT_TRENDLINE
        from numta.api.batch import HT_TRENDLINE_batch
        _, _, _, close, _ = ohlcv_2d
        gpu = HT_TRENDLINE_batch(close)
        _check(gpu, lambda c: HT_TRENDLINE(c), [close])

    def test_ht_trendmode(self, ohlcv_2d):
        from numta import HT_TRENDMODE
        from numta.api.batch import HT_TRENDMODE_batch
        _, _, _, close, _ = ohlcv_2d
        gpu = HT_TRENDMODE_batch(close)
        _check(gpu, lambda c: HT_TRENDMODE(c), [close])

    def test_ht_dcperiod(self, ohlcv_2d):
        from numta import HT_DCPERIOD
        from numta.api.batch import HT_DCPERIOD_batch
        _, _, _, close, _ = ohlcv_2d
        gpu = HT_DCPERIOD_batch(close)
        _check(gpu, lambda c: HT_DCPERIOD(c), [close])

    def test_ht_dcphase(self, ohlcv_2d):
        from numta import HT_DCPHASE
        from numta.api.batch import HT_DCPHASE_batch
        _, _, _, close, _ = ohlcv_2d
        gpu = HT_DCPHASE_batch(close)
        _check(gpu, lambda c: HT_DCPHASE(c), [close])

    def test_ht_phasor(self, ohlcv_2d):
        from numta import HT_PHASOR
        from numta.api.batch import HT_PHASOR_batch
        _, _, _, close, _ = ohlcv_2d
        g_inp, g_quad = HT_PHASOR_batch(close)
        for t in range(NUM_TICKERS):
            c_inp, c_quad = HT_PHASOR(close[t])
            np.testing.assert_allclose(g_inp[t], c_inp, rtol=1e-8, equal_nan=True)
            np.testing.assert_allclose(g_quad[t], c_quad, rtol=1e-8, equal_nan=True)

    def test_ht_sine(self, ohlcv_2d):
        from numta import HT_SINE
        from numta.api.batch import HT_SINE_batch
        _, _, _, close, _ = ohlcv_2d
        g_sine, g_lead = HT_SINE_batch(close)
        for t in range(NUM_TICKERS):
            c_sine, c_lead = HT_SINE(close[t])
            np.testing.assert_allclose(g_sine[t], c_sine, rtol=1e-8, equal_nan=True)
            np.testing.assert_allclose(g_lead[t], c_lead, rtol=1e-8, equal_nan=True)
