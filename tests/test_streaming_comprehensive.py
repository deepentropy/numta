"""
Comprehensive tests for streaming indicators.

This module provides extensive testing for streaming indicator functionality,
including batch equivalence, reset behavior, and edge cases.
"""

import pytest
import numpy as np

from numta.streaming import (
    # Base classes
    StreamingIndicator,
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


# =====================================================================
# Fixtures
# =====================================================================

RANDOM_SEED = 42


@pytest.fixture
def sample_prices():
    """Generate sample price data."""
    np.random.seed(RANDOM_SEED)
    return 100 + np.cumsum(np.random.randn(100) * 0.5)


@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data."""
    np.random.seed(RANDOM_SEED)
    n = 100
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.5)
    low = close - np.abs(np.random.randn(n) * 0.5)
    open_ = close + np.random.randn(n) * 0.3
    volume = np.random.randint(1000, 10000, n).astype(float)
    return open_, high, low, close, volume


@pytest.fixture
def large_sample_prices():
    """Generate larger sample price data for robustness testing."""
    np.random.seed(RANDOM_SEED)
    return 100 + np.cumsum(np.random.randn(1000) * 0.5)


# =====================================================================
# Test Classes
# =====================================================================

class TestStreamingSMABasic:
    """Basic tests for StreamingSMA."""
    
    def test_initialization(self):
        """Test proper initialization."""
        sma = StreamingSMA(timeperiod=10)
        assert sma.timeperiod == 10
        assert not sma.ready
        assert sma.value is None
    
    def test_first_values_none(self):
        """Test that first values return None until ready."""
        sma = StreamingSMA(timeperiod=5)
        
        for i in range(4):
            result = sma.update(100 + i)
            assert result is None, f"Expected None at index {i}"
        
        # 5th value should produce result
        result = sma.update(104)
        assert result is not None
    
    def test_simple_calculation(self):
        """Test simple SMA calculation."""
        sma = StreamingSMA(timeperiod=3)
        
        sma.update(1.0)
        sma.update(2.0)
        result = sma.update(3.0)
        
        assert result is not None
        np.testing.assert_almost_equal(result, 2.0)  # (1+2+3)/3 = 2
    
    def test_ready_property(self):
        """Test ready property."""
        sma = StreamingSMA(timeperiod=3)
        
        assert not sma.ready
        sma.update(1.0)
        assert not sma.ready
        sma.update(2.0)
        assert not sma.ready
        sma.update(3.0)
        assert sma.ready


class TestStreamingEMABasic:
    """Basic tests for StreamingEMA."""
    
    def test_initialization(self):
        """Test proper initialization."""
        ema = StreamingEMA(timeperiod=10)
        assert ema.timeperiod == 10
        assert not ema.ready
    
    def test_first_value_is_sma(self):
        """Test that first EMA value equals SMA."""
        ema = StreamingEMA(timeperiod=3)
        
        ema.update(1.0)
        ema.update(2.0)
        result = ema.update(3.0)
        
        # First EMA = SMA = (1+2+3)/3 = 2
        np.testing.assert_almost_equal(result, 2.0)
    
    def test_smoothing_factor(self):
        """Test that subsequent values use smoothing."""
        ema = StreamingEMA(timeperiod=3)
        
        ema.update(1.0)
        ema.update(2.0)
        ema.update(3.0)  # First EMA = 2.0
        
        # Alpha = 2/(3+1) = 0.5
        # Next EMA = 4 * 0.5 + 2 * 0.5 = 3.0
        result = ema.update(4.0)
        np.testing.assert_almost_equal(result, 3.0)


class TestStreamingRSIBasic:
    """Basic tests for StreamingRSI."""
    
    def test_initialization(self):
        """Test proper initialization."""
        rsi = StreamingRSI(timeperiod=14)
        assert rsi.timeperiod == 14
        assert not rsi.ready
    
    def test_rsi_bounds(self):
        """Test RSI stays within 0-100 range."""
        rsi = StreamingRSI(timeperiod=5)
        
        # Generate random walk
        np.random.seed(42)
        for _ in range(50):
            result = rsi.update(100 + np.random.randn() * 5)
            if result is not None:
                assert 0 <= result <= 100, f"RSI out of bounds: {result}"
    
    def test_uptrend_high_rsi(self):
        """Test that uptrend produces high RSI."""
        rsi = StreamingRSI(timeperiod=5)
        
        # Consistent upward movement
        for i in range(20):
            result = rsi.update(100 + i)
        
        assert result is not None
        assert result > 70, "RSI should be high in uptrend"
    
    def test_downtrend_low_rsi(self):
        """Test that downtrend produces low RSI."""
        rsi = StreamingRSI(timeperiod=5)
        
        # Consistent downward movement
        for i in range(20):
            result = rsi.update(200 - i)
        
        assert result is not None
        assert result < 30, "RSI should be low in downtrend"


class TestBatchEquivalence:
    """Test that streaming results match batch calculations."""
    
    def test_sma_matches_batch(self, sample_prices):
        """Test StreamingSMA matches batch SMA."""
        timeperiod = 10
        
        # Batch calculation
        batch_result = SMA(sample_prices, timeperiod=timeperiod)
        
        # Streaming calculation
        sma = StreamingSMA(timeperiod=timeperiod)
        streaming_result = []
        for price in sample_prices:
            result = sma.update(price)
            streaming_result.append(result if result is not None else np.nan)
        streaming_result = np.array(streaming_result)
        
        # Compare valid values
        valid_mask = ~np.isnan(batch_result)
        np.testing.assert_array_almost_equal(
            streaming_result[valid_mask],
            batch_result[valid_mask],
            decimal=10
        )
    
    def test_ema_matches_batch(self, sample_prices):
        """Test StreamingEMA matches batch EMA."""
        timeperiod = 10
        
        batch_result = EMA(sample_prices, timeperiod=timeperiod)
        
        ema = StreamingEMA(timeperiod=timeperiod)
        streaming_result = []
        for price in sample_prices:
            result = ema.update(price)
            streaming_result.append(result if result is not None else np.nan)
        streaming_result = np.array(streaming_result)
        
        valid_mask = ~np.isnan(batch_result)
        np.testing.assert_array_almost_equal(
            streaming_result[valid_mask],
            batch_result[valid_mask],
            decimal=10
        )
    
    def test_wma_matches_batch(self, sample_prices):
        """Test StreamingWMA matches batch WMA."""
        timeperiod = 10
        
        batch_result = WMA(sample_prices, timeperiod=timeperiod)
        
        wma = StreamingWMA(timeperiod=timeperiod)
        streaming_result = []
        for price in sample_prices:
            result = wma.update(price)
            streaming_result.append(result if result is not None else np.nan)
        streaming_result = np.array(streaming_result)
        
        valid_mask = ~np.isnan(batch_result)
        np.testing.assert_array_almost_equal(
            streaming_result[valid_mask],
            batch_result[valid_mask],
            decimal=10
        )
    
    def test_bbands_matches_batch(self, sample_prices):
        """Test StreamingBBANDS matches batch BBANDS."""
        timeperiod = 10
        
        batch_upper, batch_middle, batch_lower = BBANDS(sample_prices, timeperiod=timeperiod)
        
        bbands = StreamingBBANDS(timeperiod=timeperiod)
        streaming_upper = []
        streaming_middle = []
        streaming_lower = []
        
        for price in sample_prices:
            result = bbands.update(price)
            if result is not None:
                streaming_upper.append(result[0])
                streaming_middle.append(result[1])
                streaming_lower.append(result[2])
            else:
                streaming_upper.append(np.nan)
                streaming_middle.append(np.nan)
                streaming_lower.append(np.nan)
        
        streaming_middle = np.array(streaming_middle)
        
        valid_mask = ~np.isnan(batch_middle)
        np.testing.assert_array_almost_equal(
            streaming_middle[valid_mask],
            batch_middle[valid_mask],
            decimal=10
        )


class TestResetFunctionality:
    """Test reset functionality for streaming indicators."""
    
    def test_sma_reset(self):
        """Test StreamingSMA reset."""
        sma = StreamingSMA(timeperiod=5)
        
        # Fill with data
        for i in range(10):
            sma.update(100 + i)
        
        assert sma.ready
        original_value = sma.value
        
        # Reset
        sma.reset()
        
        assert not sma.ready
        assert sma.value is None
        
        # Fill again with different data
        for i in range(10):
            sma.update(200 + i)
        
        assert sma.ready
        assert sma.value != original_value
    
    def test_ema_reset(self):
        """Test StreamingEMA reset."""
        ema = StreamingEMA(timeperiod=5)
        
        for i in range(10):
            ema.update(100 + i)
        
        assert ema.ready
        
        ema.reset()
        
        assert not ema.ready
        assert ema.value is None
    
    def test_rsi_reset(self):
        """Test StreamingRSI reset."""
        rsi = StreamingRSI(timeperiod=5)
        
        for i in range(10):
            rsi.update(100 + i)
        
        assert rsi.ready
        
        rsi.reset()
        
        assert not rsi.ready
        assert rsi.value is None
    
    def test_all_indicators_reset(self):
        """Test reset for all streaming indicators."""
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
        
        # Fill all with data
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
            assert not ind.ready, f"{type(ind).__name__} should not be ready"
            assert ind.value is None


class TestEdgeCases:
    """Test edge cases for streaming indicators."""
    
    def test_nan_input(self):
        """Test handling of NaN input values."""
        sma = StreamingSMA(timeperiod=5)
        
        # Send some values then NaN
        for i in range(4):
            sma.update(100 + i)
        
        # NaN input
        result = sma.update(np.nan)
        # Should handle NaN gracefully (result might be NaN)
        
        # Continue with valid values
        for i in range(5):
            result = sma.update(100 + i)
    
    def test_inf_input(self):
        """Test handling of inf input values."""
        sma = StreamingSMA(timeperiod=5)
        
        for i in range(4):
            sma.update(100 + i)
        
        # Inf input
        result = sma.update(np.inf)
        
        # Result might be inf
        # Just verify it doesn't crash
    
    def test_very_large_values(self):
        """Test handling of very large values."""
        sma = StreamingSMA(timeperiod=5)
        
        for i in range(10):
            result = sma.update(1e100)
        
        assert result is not None
        np.testing.assert_almost_equal(result, 1e100)
    
    def test_very_small_values(self):
        """Test handling of very small values."""
        sma = StreamingSMA(timeperiod=5)
        
        for i in range(10):
            result = sma.update(1e-100)
        
        assert result is not None
        np.testing.assert_almost_equal(result, 1e-100)
    
    def test_zero_values(self):
        """Test handling of zero values."""
        sma = StreamingSMA(timeperiod=5)
        
        for i in range(10):
            result = sma.update(0.0)
        
        assert result is not None
        np.testing.assert_almost_equal(result, 0.0)
    
    def test_negative_values(self):
        """Test handling of negative values."""
        sma = StreamingSMA(timeperiod=5)
        
        values = [-100.0, -101.0, -102.0, -103.0, -104.0, -105.0, -106.0, -107.0, -108.0, -109.0]
        for val in values:
            result = sma.update(val)
        
        assert result is not None
        # SMA of last 5 values: -105, -106, -107, -108, -109 = -107
        np.testing.assert_almost_equal(result, -107.0)


class TestCircularBufferComprehensive:
    """Comprehensive tests for CircularBuffer."""
    
    def test_empty_operations(self):
        """Test operations on empty buffer."""
        buf = CircularBuffer(5)
        
        assert len(buf) == 0
        assert not buf.full
        assert buf.sum == 0.0
        assert len(buf.values) == 0
    
    def test_filling_buffer(self):
        """Test progressive filling of buffer."""
        buf = CircularBuffer(5)
        
        for i in range(5):
            buf.append(float(i + 1))
            assert len(buf) == i + 1
            assert not buf.full if i < 4 else buf.full
    
    def test_overflow_behavior(self):
        """Test behavior when buffer overflows."""
        buf = CircularBuffer(3)
        
        # Fill buffer
        buf.append(1.0)
        buf.append(2.0)
        buf.append(3.0)
        
        assert buf.full
        assert len(buf) == 3
        
        # Overflow
        buf.append(4.0)
        
        assert len(buf) == 3  # Still 3
        assert buf[0] == 2.0  # Oldest is now 2
        assert buf[2] == 4.0  # Newest is 4
    
    def test_sum_tracking(self):
        """Test sum is correctly maintained."""
        buf = CircularBuffer(3)
        
        buf.append(10.0)
        assert buf.sum == 10.0
        
        buf.append(20.0)
        assert buf.sum == 30.0
        
        buf.append(30.0)
        assert buf.sum == 60.0
        
        # Overflow removes 10, adds 40
        buf.append(40.0)
        assert buf.sum == 90.0  # 20 + 30 + 40
    
    def test_values_property(self):
        """Test values property returns correct array."""
        buf = CircularBuffer(4)
        
        buf.append(1.0)
        buf.append(2.0)
        np.testing.assert_array_equal(buf.values, [1.0, 2.0])
        
        buf.append(3.0)
        buf.append(4.0)
        np.testing.assert_array_equal(buf.values, [1.0, 2.0, 3.0, 4.0])
        
        buf.append(5.0)
        np.testing.assert_array_equal(buf.values, [2.0, 3.0, 4.0, 5.0])
    
    def test_negative_indexing(self):
        """Test negative indexing."""
        buf = CircularBuffer(3)
        
        buf.append(1.0)
        buf.append(2.0)
        buf.append(3.0)
        
        assert buf[-1] == 3.0  # Most recent
        assert buf[-2] == 2.0
        assert buf[-3] == 1.0  # Oldest
    
    def test_clear(self):
        """Test clearing buffer."""
        buf = CircularBuffer(5)
        
        for i in range(5):
            buf.append(float(i))
        
        assert buf.full
        
        buf.clear()
        
        assert len(buf) == 0
        assert not buf.full
        assert buf.sum == 0.0


class TestMomentumIndicators:
    """Test momentum streaming indicators."""
    
    def test_streaming_mom(self):
        """Test StreamingMOM."""
        mom = StreamingMOM(timeperiod=5)
        
        # Linearly increasing values
        for i in range(10):
            result = mom.update(100 + i)
        
        assert mom.ready
        # MOM = current - past = 109 - 104 = 5
        np.testing.assert_almost_equal(mom.value, 5.0)
    
    def test_streaming_roc(self):
        """Test StreamingROC."""
        roc = StreamingROC(timeperiod=5)
        
        # Double the price
        for i in range(6):
            roc.update(100.0)
        
        result = roc.update(200.0)
        
        # ROC = ((200 - 100) / 100) * 100 = 100%
        assert roc.ready
        np.testing.assert_almost_equal(roc.value, 100.0)


class TestVolatilityIndicators:
    """Test volatility streaming indicators."""
    
    def test_streaming_trange(self, sample_ohlcv):
        """Test StreamingTRANGE."""
        open_, high, low, close, _ = sample_ohlcv
        
        tr = StreamingTRANGE()
        
        for i in range(len(high)):
            result = tr.update_bar(open_[i], high[i], low[i], close[i])
        
        # Should have values after first bar
        assert tr.ready
        assert tr.value is not None
        assert tr.value >= 0  # True range is always non-negative
    
    def test_streaming_atr(self, sample_ohlcv):
        """Test StreamingATR."""
        _, high, low, close, _ = sample_ohlcv
        
        atr = StreamingATR(timeperiod=14)
        
        for i in range(len(high)):
            result = atr.update_bar(
                open_=high[i] - 0.5,
                high=high[i],
                low=low[i],
                close=close[i]
            )
        
        assert atr.ready
        assert atr.value is not None
        assert atr.value > 0  # ATR should be positive


class TestVolumeIndicators:
    """Test volume streaming indicators."""
    
    def test_streaming_obv(self, sample_ohlcv):
        """Test StreamingOBV."""
        _, high, low, close, volume = sample_ohlcv
        
        obv = StreamingOBV()
        
        for i in range(len(close)):
            result = obv.update_bar(
                open_=high[i] - 0.5,
                high=high[i],
                low=low[i],
                close=close[i],
                volume=volume[i]
            )
        
        # OBV should have a value
        assert obv.value is not None
    
    def test_obv_up_days(self):
        """Test OBV accumulates on up days."""
        obv = StreamingOBV()
        
        # First bar
        obv.update_bar(100, 105, 95, 100, 1000)
        assert obv.value == 0  # First bar is always 0
        
        # Up day - volume should be added
        obv.update_bar(100, 110, 98, 105, 2000)
        assert obv.value == 2000
        
        # Another up day
        obv.update_bar(105, 115, 103, 110, 1500)
        assert obv.value == 3500
    
    def test_obv_down_days(self):
        """Test OBV decrements on down days."""
        obv = StreamingOBV()
        
        # First bar
        obv.update_bar(100, 105, 95, 100, 1000)
        
        # Down day - volume should be subtracted
        obv.update_bar(100, 102, 92, 95, 2000)
        assert obv.value == -2000


class TestLongRunning:
    """Test streaming indicators over long periods."""
    
    def test_sma_long_running(self, large_sample_prices):
        """Test StreamingSMA over 1000 points."""
        timeperiod = 50
        
        batch_result = SMA(large_sample_prices, timeperiod=timeperiod)
        
        sma = StreamingSMA(timeperiod=timeperiod)
        streaming_result = []
        for price in large_sample_prices:
            result = sma.update(price)
            streaming_result.append(result if result is not None else np.nan)
        streaming_result = np.array(streaming_result)
        
        valid_mask = ~np.isnan(batch_result)
        np.testing.assert_array_almost_equal(
            streaming_result[valid_mask],
            batch_result[valid_mask],
            decimal=8
        )
    
    def test_ema_long_running(self, large_sample_prices):
        """Test StreamingEMA over 1000 points."""
        timeperiod = 50
        
        batch_result = EMA(large_sample_prices, timeperiod=timeperiod)
        
        ema = StreamingEMA(timeperiod=timeperiod)
        streaming_result = []
        for price in large_sample_prices:
            result = ema.update(price)
            streaming_result.append(result if result is not None else np.nan)
        streaming_result = np.array(streaming_result)
        
        valid_mask = ~np.isnan(batch_result)
        np.testing.assert_array_almost_equal(
            streaming_result[valid_mask],
            batch_result[valid_mask],
            decimal=8
        )
