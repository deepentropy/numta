"""
Tests for streaming indicators.

These tests verify that streaming indicators produce the same results
as their batch counterparts when fed the same data.
"""

import pytest
import numpy as np

from numta.streaming import (
    # Base classes
    StreamingIndicator,
    StreamingOHLCVIndicator,
    CircularBuffer,
    # Overlap
    StreamingSMA,
    StreamingEMA,
    StreamingBBANDS,
    StreamingDEMA,
    StreamingTEMA,
    StreamingWMA,
    # Momentum
    StreamingRSI,
    StreamingMACD,
    StreamingSTOCH,
    StreamingMOM,
    StreamingROC,
    # Volatility
    StreamingATR,
    StreamingTRANGE,
    # Volume
    StreamingOBV,
    StreamingAD,
)

from numta import (
    SMA, EMA, BBANDS, DEMA, TEMA, WMA,
    RSI, MACD, STOCH, MOM, ROC,
    ATR, TRANGE,
    OBV, AD,
)


# Module-level random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def _create_sample_prices(n: int = 100) -> np.ndarray:
    """Create sample price data."""
    return 100 + np.cumsum(np.random.randn(n) * 0.5)


def _create_sample_ohlcv(n: int = 100):
    """Create sample OHLCV data."""
    close = _create_sample_prices(n)
    high = close + np.abs(np.random.randn(n) * 0.5)
    low = close - np.abs(np.random.randn(n) * 0.5)
    open_ = close + np.random.randn(n) * 0.3
    volume = np.random.randint(1000, 10000, n).astype(float)
    return open_, high, low, close, volume


class TestCircularBuffer:
    """Tests for CircularBuffer class."""
    
    def test_empty_buffer(self):
        """Test empty buffer properties."""
        buf = CircularBuffer(5)
        assert len(buf) == 0
        assert not buf.full
        assert buf.sum == 0.0
    
    def test_append_and_access(self):
        """Test appending and accessing values."""
        buf = CircularBuffer(3)
        buf.append(1.0)
        buf.append(2.0)
        buf.append(3.0)
        
        assert len(buf) == 3
        assert buf.full
        assert buf[0] == 1.0
        assert buf[1] == 2.0
        assert buf[2] == 3.0
        assert buf[-1] == 3.0
    
    def test_circular_behavior(self):
        """Test that buffer wraps around correctly."""
        buf = CircularBuffer(3)
        for val in [1.0, 2.0, 3.0, 4.0, 5.0]:
            buf.append(val)
        
        assert len(buf) == 3
        assert buf[0] == 3.0  # Oldest
        assert buf[1] == 4.0
        assert buf[2] == 5.0  # Newest
    
    def test_sum(self):
        """Test sum calculation."""
        buf = CircularBuffer(3)
        buf.append(1.0)
        buf.append(2.0)
        buf.append(3.0)
        assert buf.sum == 6.0
        
        buf.append(4.0)
        assert buf.sum == 9.0  # 2 + 3 + 4
    
    def test_values(self):
        """Test getting all values."""
        buf = CircularBuffer(3)
        buf.append(1.0)
        buf.append(2.0)
        
        values = buf.values
        assert len(values) == 2
        np.testing.assert_array_equal(values, [1.0, 2.0])
        
        buf.append(3.0)
        buf.append(4.0)
        
        values = buf.values
        assert len(values) == 3
        np.testing.assert_array_equal(values, [2.0, 3.0, 4.0])
    
    def test_clear(self):
        """Test clearing buffer."""
        buf = CircularBuffer(3)
        buf.append(1.0)
        buf.append(2.0)
        buf.clear()
        
        assert len(buf) == 0
        assert not buf.full


class TestStreamingSMA:
    """Tests for StreamingSMA."""
    
    def test_basic_sma(self):
        """Test basic SMA calculation."""
        sma = StreamingSMA(timeperiod=3)
        
        assert sma.update(1.0) is None  # Not ready
        assert sma.update(2.0) is None  # Not ready
        result = sma.update(3.0)  # Ready
        
        assert result is not None
        np.testing.assert_almost_equal(result, 2.0)
    
    def test_matches_batch_sma(self):
        """Test that streaming SMA matches batch SMA."""
        prices = _create_sample_prices(50)
        timeperiod = 10
        
        # Batch calculation
        batch_sma = SMA(prices, timeperiod=timeperiod)
        
        # Streaming calculation
        sma = StreamingSMA(timeperiod=timeperiod)
        streaming_values = []
        for price in prices:
            result = sma.update(price)
            streaming_values.append(result if result is not None else np.nan)
        
        streaming_sma = np.array(streaming_values)
        
        # Compare valid values
        valid_mask = ~np.isnan(batch_sma)
        np.testing.assert_array_almost_equal(
            streaming_sma[valid_mask],
            batch_sma[valid_mask],
            decimal=10
        )
    
    def test_reset(self):
        """Test reset functionality."""
        sma = StreamingSMA(timeperiod=3)
        sma.update(1.0)
        sma.update(2.0)
        sma.update(3.0)
        
        assert sma.ready
        sma.reset()
        assert not sma.ready
        assert sma.value is None


class TestStreamingEMA:
    """Tests for StreamingEMA."""
    
    def test_basic_ema(self):
        """Test basic EMA calculation."""
        ema = StreamingEMA(timeperiod=3)
        
        # First timeperiod values seed the EMA
        ema.update(1.0)
        ema.update(2.0)
        result = ema.update(3.0)
        
        assert result is not None
        # First EMA = SMA = 2.0
        np.testing.assert_almost_equal(result, 2.0)
    
    def test_matches_batch_ema(self):
        """Test that streaming EMA matches batch EMA."""
        prices = _create_sample_prices(50)
        timeperiod = 10
        
        # Batch calculation
        batch_ema = EMA(prices, timeperiod=timeperiod)
        
        # Streaming calculation
        ema = StreamingEMA(timeperiod=timeperiod)
        streaming_values = []
        for price in prices:
            result = ema.update(price)
            streaming_values.append(result if result is not None else np.nan)
        
        streaming_ema = np.array(streaming_values)
        
        # Compare valid values
        valid_mask = ~np.isnan(batch_ema)
        np.testing.assert_array_almost_equal(
            streaming_ema[valid_mask],
            batch_ema[valid_mask],
            decimal=10
        )


class TestStreamingBBANDS:
    """Tests for StreamingBBANDS."""
    
    def test_basic_bbands(self):
        """Test basic Bollinger Bands calculation."""
        bbands = StreamingBBANDS(timeperiod=5, nbdevup=2.0, nbdevdn=2.0)
        
        for i in range(4):
            result = bbands.update(float(i + 1))
            assert result is None
        
        result = bbands.update(5.0)
        assert result is not None
        upper, middle, lower = result
        
        np.testing.assert_almost_equal(middle, 3.0)  # SMA of 1,2,3,4,5
        assert upper > middle
        assert lower < middle
    
    def test_matches_batch_bbands(self):
        """Test that streaming BBANDS matches batch BBANDS."""
        prices = _create_sample_prices(50)
        timeperiod = 10
        
        # Batch calculation
        batch_upper, batch_middle, batch_lower = BBANDS(prices, timeperiod=timeperiod)
        
        # Streaming calculation
        bbands = StreamingBBANDS(timeperiod=timeperiod)
        streaming_upper = []
        streaming_middle = []
        streaming_lower = []
        
        for price in prices:
            result = bbands.update(price)
            if result is not None:
                streaming_upper.append(result[0])
                streaming_middle.append(result[1])
                streaming_lower.append(result[2])
            else:
                streaming_upper.append(np.nan)
                streaming_middle.append(np.nan)
                streaming_lower.append(np.nan)
        
        streaming_upper = np.array(streaming_upper)
        streaming_middle = np.array(streaming_middle)
        streaming_lower = np.array(streaming_lower)
        
        # Compare valid values
        valid_mask = ~np.isnan(batch_middle)
        np.testing.assert_array_almost_equal(
            streaming_middle[valid_mask],
            batch_middle[valid_mask],
            decimal=10
        )


class TestStreamingDEMA:
    """Tests for StreamingDEMA."""
    
    def test_basic_dema(self):
        """Test basic DEMA calculation."""
        dema = StreamingDEMA(timeperiod=3)
        
        # Need 2*timeperiod-2 = 4 values to start
        for i in range(50):
            result = dema.update(float(i + 1))
        
        assert dema.ready
        assert dema.value is not None


class TestStreamingTEMA:
    """Tests for StreamingTEMA."""
    
    def test_basic_tema(self):
        """Test basic TEMA calculation."""
        tema = StreamingTEMA(timeperiod=3)
        
        # Need 3*timeperiod-3 = 6 values to start
        for i in range(50):
            result = tema.update(float(i + 1))
        
        assert tema.ready
        assert tema.value is not None


class TestStreamingWMA:
    """Tests for StreamingWMA."""
    
    def test_basic_wma(self):
        """Test basic WMA calculation."""
        wma = StreamingWMA(timeperiod=3)
        
        wma.update(1.0)
        wma.update(2.0)
        result = wma.update(3.0)
        
        assert result is not None
        # WMA = (1*1 + 2*2 + 3*3) / (1+2+3) = 14/6 = 2.333...
        np.testing.assert_almost_equal(result, 14.0 / 6.0)
    
    def test_matches_batch_wma(self):
        """Test that streaming WMA matches batch WMA."""
        prices = _create_sample_prices(50)
        timeperiod = 10
        
        # Batch calculation
        batch_wma = WMA(prices, timeperiod=timeperiod)
        
        # Streaming calculation
        wma = StreamingWMA(timeperiod=timeperiod)
        streaming_values = []
        for price in prices:
            result = wma.update(price)
            streaming_values.append(result if result is not None else np.nan)
        
        streaming_wma = np.array(streaming_values)
        
        # Compare valid values
        valid_mask = ~np.isnan(batch_wma)
        np.testing.assert_array_almost_equal(
            streaming_wma[valid_mask],
            batch_wma[valid_mask],
            decimal=10
        )


class TestStreamingRSI:
    """Tests for StreamingRSI."""
    
    def test_basic_rsi(self):
        """Test basic RSI calculation."""
        rsi = StreamingRSI(timeperiod=14)
        
        # Need timeperiod+1 values
        for i in range(20):
            result = rsi.update(100 + i * 0.5)
        
        assert rsi.ready
        assert 0 <= rsi.value <= 100
    
    def test_rsi_range(self):
        """Test that RSI stays in 0-100 range."""
        rsi = StreamingRSI(timeperiod=14)
        
        # Uptrend
        for i in range(50):
            result = rsi.update(100 + i)
            if result is not None:
                assert 0 <= result <= 100
        
        # Downtrend
        for i in range(50):
            result = rsi.update(150 - i)
            if result is not None:
                assert 0 <= result <= 100


class TestStreamingMACD:
    """Tests for StreamingMACD."""
    
    def test_basic_macd(self):
        """Test basic MACD calculation."""
        macd = StreamingMACD(fastperiod=12, slowperiod=26, signalperiod=9)
        
        for i in range(50):
            result = macd.update(100 + i * 0.5)
        
        assert macd.ready
        assert macd.macd is not None
        assert macd.signal is not None
        assert macd.histogram is not None


class TestStreamingSTOCH:
    """Tests for StreamingSTOCH."""
    
    def test_basic_stoch(self):
        """Test basic Stochastic calculation."""
        stoch = StreamingSTOCH(fastk_period=5, slowk_period=3, slowd_period=3)
        
        for i in range(30):
            result = stoch.update_bar(
                open_=100 + i,
                high=105 + i,
                low=95 + i,
                close=102 + i
            )
        
        assert stoch.ready
        assert 0 <= stoch.slowk <= 100
        assert 0 <= stoch.slowd <= 100


class TestStreamingMOM:
    """Tests for StreamingMOM."""
    
    def test_basic_mom(self):
        """Test basic Momentum calculation."""
        mom = StreamingMOM(timeperiod=10)
        
        for i in range(15):
            result = mom.update(100 + i)
        
        assert mom.ready
        # MOM = 114 - 104 = 10
        np.testing.assert_almost_equal(mom.value, 10.0)


class TestStreamingROC:
    """Tests for StreamingROC."""
    
    def test_basic_roc(self):
        """Test basic Rate of Change calculation."""
        roc = StreamingROC(timeperiod=10)
        
        for i in range(15):
            result = roc.update(100 + i)
        
        assert roc.ready
        # ROC = ((114 - 104) / 104) * 100
        expected = ((114 - 104) / 104) * 100
        np.testing.assert_almost_equal(roc.value, expected)


class TestStreamingTRANGE:
    """Tests for StreamingTRANGE."""
    
    def test_basic_trange(self):
        """Test basic True Range calculation."""
        tr = StreamingTRANGE()
        
        # First bar
        result = tr.update_bar(100, 105, 95, 102)
        assert result == 10.0  # high - low
        
        # Second bar with gap
        result = tr.update_bar(102, 110, 100, 108)
        # TR = max(110-100, |110-102|, |100-102|) = max(10, 8, 2) = 10
        assert result == 10.0


class TestStreamingATR:
    """Tests for StreamingATR."""
    
    def test_basic_atr(self):
        """Test basic ATR calculation."""
        atr = StreamingATR(timeperiod=14)
        
        for i in range(20):
            result = atr.update_bar(
                open_=100 + i,
                high=105 + i,
                low=95 + i,
                close=102 + i
            )
        
        assert atr.ready
        assert atr.value > 0


class TestStreamingOBV:
    """Tests for StreamingOBV."""
    
    def test_basic_obv(self):
        """Test basic OBV calculation."""
        obv = StreamingOBV()
        
        # First bar - OBV = 0
        result = obv.update_bar(100, 105, 95, 102, 1000)
        assert result == 0
        
        # Up day - add volume
        result = obv.update_bar(102, 108, 100, 105, 1500)
        assert result == 1500
        
        # Down day - subtract volume
        result = obv.update_bar(105, 106, 99, 100, 1200)
        assert result == 300


class TestStreamingAD:
    """Tests for StreamingAD."""
    
    def test_basic_ad(self):
        """Test basic AD Line calculation."""
        ad = StreamingAD()
        
        # Bar where close is at high (full accumulation)
        result = ad.update_bar(100, 110, 100, 110, 1000)
        # MFM = ((110-100) - (110-110)) / (110-100) = 10/10 = 1.0
        # MFV = 1.0 * 1000 = 1000
        np.testing.assert_almost_equal(result, 1000.0)
        
        # Bar where close is at low (full distribution)
        result = ad.update_bar(110, 120, 110, 110, 2000)
        # MFM = ((110-110) - (120-110)) / (120-110) = -10/10 = -1.0
        # MFV = -1.0 * 2000 = -2000
        # AD = 1000 + (-2000) = -1000
        np.testing.assert_almost_equal(result, -1000.0)


class TestStreamingIndicatorReset:
    """Tests for reset functionality across all indicators."""
    
    def test_all_indicators_reset(self):
        """Test that all streaming indicators reset correctly."""
        indicators = [
            StreamingSMA(5),
            StreamingEMA(5),
            StreamingDEMA(5),
            StreamingTEMA(5),
            StreamingWMA(5),
            StreamingRSI(5),
            StreamingMOM(5),
            StreamingROC(5),
        ]
        
        # Fill with data
        for _ in range(20):
            for ind in indicators:
                ind.update(100.0)
        
        # Verify all are ready
        for ind in indicators:
            assert ind.ready, f"{type(ind).__name__} should be ready"
        
        # Reset all
        for ind in indicators:
            ind.reset()
        
        # Verify all are not ready
        for ind in indicators:
            assert not ind.ready, f"{type(ind).__name__} should not be ready after reset"
            assert ind.value is None, f"{type(ind).__name__} value should be None after reset"
