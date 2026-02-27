"""Tests for GPU batch overlap indicators — verify GPU matches CPU for all tickers."""
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


def _check_gpu_vs_cpu(gpu_result, cpu_fn, inputs, rtol=1e-10):
    """Assert GPU batch result matches CPU single-ticker for all tickers."""
    for t in range(NUM_TICKERS):
        cpu = cpu_fn(*[inp[t] for inp in inputs])
        np.testing.assert_allclose(gpu_result[t], cpu, rtol=rtol, equal_nan=True,
                                   err_msg=f"Ticker {t}")


class TestOverlapBatch:
    def test_sma(self, ohlcv_2d):
        from numta import SMA
        from numta.api.batch import SMA_batch
        _, _, _, close, _ = ohlcv_2d
        gpu = SMA_batch(close, timeperiod=14)
        _check_gpu_vs_cpu(gpu, lambda c: SMA(c, timeperiod=14), [close])

    def test_ema(self, ohlcv_2d):
        from numta import EMA
        from numta.api.batch import EMA_batch
        _, _, _, close, _ = ohlcv_2d
        gpu = EMA_batch(close, timeperiod=14)
        _check_gpu_vs_cpu(gpu, lambda c: EMA(c, timeperiod=14), [close])

    def test_bbands(self, ohlcv_2d):
        from numta import BBANDS
        from numta.api.batch import BBANDS_batch
        _, _, _, close, _ = ohlcv_2d
        upper, middle, lower = BBANDS_batch(close, timeperiod=20)
        for t in range(NUM_TICKERS):
            cu, cm, cl = BBANDS(close[t], timeperiod=20)
            np.testing.assert_allclose(upper[t], cu, rtol=1e-10, equal_nan=True)
            np.testing.assert_allclose(middle[t], cm, rtol=1e-10, equal_nan=True)
            np.testing.assert_allclose(lower[t], cl, rtol=1e-10, equal_nan=True)

    def test_dema(self, ohlcv_2d):
        from numta import DEMA
        from numta.api.batch import DEMA_batch
        _, _, _, close, _ = ohlcv_2d
        gpu = DEMA_batch(close, timeperiod=14)
        _check_gpu_vs_cpu(gpu, lambda c: DEMA(c, timeperiod=14), [close])

    def test_kama(self, ohlcv_2d):
        from numta import KAMA
        from numta.api.batch import KAMA_batch
        _, _, _, close, _ = ohlcv_2d
        gpu = KAMA_batch(close, timeperiod=30)
        _check_gpu_vs_cpu(gpu, lambda c: KAMA(c, timeperiod=30), [close])

    def test_wma(self, ohlcv_2d):
        from numta import WMA
        from numta.api.batch import WMA_batch
        _, _, _, close, _ = ohlcv_2d
        gpu = WMA_batch(close, timeperiod=14)
        _check_gpu_vs_cpu(gpu, lambda c: WMA(c, timeperiod=14), [close])

    def test_tema(self, ohlcv_2d):
        from numta import TEMA
        from numta.api.batch import TEMA_batch
        _, _, _, close, _ = ohlcv_2d
        gpu = TEMA_batch(close, timeperiod=14)
        _check_gpu_vs_cpu(gpu, lambda c: TEMA(c, timeperiod=14), [close])

    def test_t3(self, ohlcv_2d):
        from numta import T3
        from numta.api.batch import T3_batch
        _, _, _, close, _ = ohlcv_2d
        gpu = T3_batch(close, timeperiod=5)
        _check_gpu_vs_cpu(gpu, lambda c: T3(c, timeperiod=5), [close])

    def test_trima(self, ohlcv_2d):
        from numta import TRIMA
        from numta.api.batch import TRIMA_batch
        _, _, _, close, _ = ohlcv_2d
        gpu = TRIMA_batch(close, timeperiod=14)
        _check_gpu_vs_cpu(gpu, lambda c: TRIMA(c, timeperiod=14), [close])

    def test_mama(self, ohlcv_2d):
        from numta import MAMA
        from numta.api.batch import MAMA_batch
        _, _, _, close, _ = ohlcv_2d
        g_mama, g_fama = MAMA_batch(close)
        for t in range(NUM_TICKERS):
            c_mama, c_fama = MAMA(close[t])
            np.testing.assert_allclose(g_mama[t], c_mama, rtol=1e-10, equal_nan=True)
            np.testing.assert_allclose(g_fama[t], c_fama, rtol=1e-10, equal_nan=True)

    def test_sar(self, ohlcv_2d):
        from numta import SAR
        from numta.api.batch import SAR_batch
        _, high, low, _, _ = ohlcv_2d
        gpu = SAR_batch(high, low)
        _check_gpu_vs_cpu(gpu, lambda h, l: SAR(h, l), [high, low])

    def test_sarext(self, ohlcv_2d):
        from numta import SAREXT
        from numta.api.batch import SAREXT_batch
        _, high, low, _, _ = ohlcv_2d
        gpu = SAREXT_batch(high, low)
        _check_gpu_vs_cpu(gpu, lambda h, l: SAREXT(h, l), [high, low])
