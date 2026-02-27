"""Tests for GPU batch momentum indicators — verify GPU matches CPU for all tickers."""
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


class TestMomentumBatch:
    def test_rsi(self, ohlcv_2d):
        from numta import RSI
        from numta.api.batch import RSI_batch
        _, _, _, close, _ = ohlcv_2d
        gpu = RSI_batch(close, timeperiod=14)
        _check(gpu, lambda c: RSI(c, timeperiod=14), [close])

    def test_macd(self, ohlcv_2d):
        from numta import MACD
        from numta.api.batch import MACD_batch
        _, _, _, close, _ = ohlcv_2d
        g_macd, g_signal, g_hist = MACD_batch(close)
        for t in range(NUM_TICKERS):
            c_macd, c_signal, c_hist = MACD(close[t])
            np.testing.assert_allclose(g_macd[t], c_macd, rtol=1e-10, equal_nan=True)
            np.testing.assert_allclose(g_signal[t], c_signal, rtol=1e-10, equal_nan=True)
            np.testing.assert_allclose(g_hist[t], c_hist, rtol=1e-10, equal_nan=True)

    def test_adx(self, ohlcv_2d):
        from numta import ADX
        from numta.api.batch import ADX_batch
        _, high, low, close, _ = ohlcv_2d
        gpu = ADX_batch(high, low, close, timeperiod=14)
        _check(gpu, lambda h, l, c: ADX(h, l, c, timeperiod=14), [high, low, close])

    def test_atr(self, ohlcv_2d):
        from numta import ATR
        from numta.api.batch import ATR_batch
        _, high, low, close, _ = ohlcv_2d
        gpu = ATR_batch(high, low, close, timeperiod=14)
        _check(gpu, lambda h, l, c: ATR(h, l, c, timeperiod=14), [high, low, close])

    def test_cci(self, ohlcv_2d):
        from numta import CCI
        from numta.api.batch import CCI_batch
        _, high, low, close, _ = ohlcv_2d
        gpu = CCI_batch(high, low, close, timeperiod=14)
        _check(gpu, lambda h, l, c: CCI(h, l, c, timeperiod=14), [high, low, close])

    def test_cmo(self, ohlcv_2d):
        from numta import CMO
        from numta.api.batch import CMO_batch
        _, _, _, close, _ = ohlcv_2d
        gpu = CMO_batch(close, timeperiod=14)
        _check(gpu, lambda c: CMO(c, timeperiod=14), [close])

    def test_dx(self, ohlcv_2d):
        from numta import DX
        from numta.api.batch import DX_batch
        _, high, low, close, _ = ohlcv_2d
        gpu = DX_batch(high, low, close, timeperiod=14)
        _check(gpu, lambda h, l, c: DX(h, l, c, timeperiod=14), [high, low, close])

    def test_mom(self, ohlcv_2d):
        from numta import MOM
        from numta.api.batch import MOM_batch
        _, _, _, close, _ = ohlcv_2d
        gpu = MOM_batch(close, timeperiod=10)
        _check(gpu, lambda c: MOM(c, timeperiod=10), [close])

    def test_roc(self, ohlcv_2d):
        from numta import ROC
        from numta.api.batch import ROC_batch
        _, _, _, close, _ = ohlcv_2d
        gpu = ROC_batch(close, timeperiod=10)
        _check(gpu, lambda c: ROC(c, timeperiod=10), [close])

    def test_rocp(self, ohlcv_2d):
        from numta import ROCP
        from numta.api.batch import ROCP_batch
        _, _, _, close, _ = ohlcv_2d
        gpu = ROCP_batch(close, timeperiod=10)
        _check(gpu, lambda c: ROCP(c, timeperiod=10), [close])

    def test_rocr(self, ohlcv_2d):
        from numta import ROCR
        from numta.api.batch import ROCR_batch
        _, _, _, close, _ = ohlcv_2d
        gpu = ROCR_batch(close, timeperiod=10)
        _check(gpu, lambda c: ROCR(c, timeperiod=10), [close])

    def test_rocr100(self, ohlcv_2d):
        from numta import ROCR100
        from numta.api.batch import ROCR100_batch
        _, _, _, close, _ = ohlcv_2d
        gpu = ROCR100_batch(close, timeperiod=10)
        _check(gpu, lambda c: ROCR100(c, timeperiod=10), [close])

    def test_willr(self, ohlcv_2d):
        from numta import WILLR
        from numta.api.batch import WILLR_batch
        _, high, low, close, _ = ohlcv_2d
        gpu = WILLR_batch(high, low, close, timeperiod=14)
        _check(gpu, lambda h, l, c: WILLR(h, l, c, timeperiod=14), [high, low, close])

    def test_stoch(self, ohlcv_2d):
        from numta import STOCH
        from numta.api.batch import STOCH_batch
        _, high, low, close, _ = ohlcv_2d
        g_slowk, g_slowd = STOCH_batch(high, low, close)
        for t in range(NUM_TICKERS):
            c_slowk, c_slowd = STOCH(high[t], low[t], close[t])
            np.testing.assert_allclose(g_slowk[t], c_slowk, rtol=1e-10, equal_nan=True)
            np.testing.assert_allclose(g_slowd[t], c_slowd, rtol=1e-10, equal_nan=True)

    def test_stochf(self, ohlcv_2d):
        from numta import STOCHF
        from numta.api.batch import STOCHF_batch
        _, high, low, close, _ = ohlcv_2d
        g_fastk, g_fastd = STOCHF_batch(high, low, close)
        for t in range(NUM_TICKERS):
            c_fastk, c_fastd = STOCHF(high[t], low[t], close[t])
            np.testing.assert_allclose(g_fastk[t], c_fastk, rtol=1e-10, equal_nan=True)
            np.testing.assert_allclose(g_fastd[t], c_fastd, rtol=1e-10, equal_nan=True)

    def test_bop(self, ohlcv_2d):
        from numta import BOP
        from numta.api.batch import BOP_batch
        open_, high, low, close, _ = ohlcv_2d
        gpu = BOP_batch(open_, high, low, close)
        _check(gpu, lambda o, h, l, c: BOP(o, h, l, c), [open_, high, low, close])

    def test_aroon(self, ohlcv_2d):
        from numta import AROON
        from numta.api.batch import AROON_batch
        _, high, low, _, _ = ohlcv_2d
        g_down, g_up = AROON_batch(high, low, timeperiod=14)
        for t in range(NUM_TICKERS):
            c_down, c_up = AROON(high[t], low[t], timeperiod=14)
            np.testing.assert_allclose(g_down[t], c_down, rtol=1e-10, equal_nan=True)
            np.testing.assert_allclose(g_up[t], c_up, rtol=1e-10, equal_nan=True)

    def test_aroonosc(self, ohlcv_2d):
        from numta import AROONOSC
        from numta.api.batch import AROONOSC_batch
        _, high, low, _, _ = ohlcv_2d
        gpu = AROONOSC_batch(high, low, timeperiod=14)
        _check(gpu, lambda h, l: AROONOSC(h, l, timeperiod=14), [high, low])

    def test_mfi(self, ohlcv_2d):
        from numta import MFI
        from numta.api.batch import MFI_batch
        _, high, low, close, volume = ohlcv_2d
        gpu = MFI_batch(high, low, close, volume, timeperiod=14)
        _check(gpu, lambda h, l, c, v: MFI(h, l, c, v, timeperiod=14), [high, low, close, volume])

    def test_minus_dm(self, ohlcv_2d):
        from numta import MINUS_DM
        from numta.api.batch import MINUS_DM_batch
        _, high, low, _, _ = ohlcv_2d
        gpu = MINUS_DM_batch(high, low, timeperiod=14)
        _check(gpu, lambda h, l: MINUS_DM(h, l, timeperiod=14), [high, low])

    def test_plus_dm(self, ohlcv_2d):
        from numta import PLUS_DM
        from numta.api.batch import PLUS_DM_batch
        _, high, low, _, _ = ohlcv_2d
        gpu = PLUS_DM_batch(high, low, timeperiod=14)
        _check(gpu, lambda h, l: PLUS_DM(h, l, timeperiod=14), [high, low])

    def test_minus_di(self, ohlcv_2d):
        from numta import MINUS_DI
        from numta.api.batch import MINUS_DI_batch
        _, high, low, close, _ = ohlcv_2d
        gpu = MINUS_DI_batch(high, low, close, timeperiod=14)
        _check(gpu, lambda h, l, c: MINUS_DI(h, l, c, timeperiod=14), [high, low, close])

    def test_plus_di(self, ohlcv_2d):
        from numta import PLUS_DI
        from numta.api.batch import PLUS_DI_batch
        _, high, low, close, _ = ohlcv_2d
        gpu = PLUS_DI_batch(high, low, close, timeperiod=14)
        _check(gpu, lambda h, l, c: PLUS_DI(h, l, c, timeperiod=14), [high, low, close])

    def test_ultosc(self, ohlcv_2d):
        from numta import ULTOSC
        from numta.api.batch import ULTOSC_batch
        _, high, low, close, _ = ohlcv_2d
        gpu = ULTOSC_batch(high, low, close)
        _check(gpu, lambda h, l, c: ULTOSC(h, l, c), [high, low, close])
