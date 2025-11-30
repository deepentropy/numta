"""
Tests for pandas DataFrame extension accessor (.ta)
"""

import pytest
import numpy as np

# Check if pandas is available
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

if HAS_PANDAS:
    import numta


# Skip all tests if pandas is not available
pytestmark = pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")


@pytest.fixture
def sample_df():
    """Create a sample OHLCV DataFrame for testing."""
    np.random.seed(42)
    n = 100
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.5)
    low = close - np.abs(np.random.randn(n) * 0.5)
    open_ = close + np.random.randn(n) * 0.3
    volume = np.random.randint(1000, 10000, n).astype(float)
    
    return pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })


@pytest.fixture
def uppercase_df():
    """Create a DataFrame with uppercase column names."""
    return pd.DataFrame({
        'Open': [100, 101, 102, 103, 104],
        'High': [105, 106, 107, 108, 109],
        'Low': [99, 100, 101, 102, 103],
        'Close': [104, 105, 106, 107, 108],
        'Volume': [1000, 1100, 1200, 1300, 1400]
    })


@pytest.fixture
def short_names_df():
    """Create a DataFrame with short column names."""
    return pd.DataFrame({
        'o': [100, 101, 102, 103, 104],
        'h': [105, 106, 107, 108, 109],
        'l': [99, 100, 101, 102, 103],
        'c': [104, 105, 106, 107, 108],
        'v': [1000, 1100, 1200, 1300, 1400]
    })


class TestTAAccessorBasics:
    """Test basic accessor functionality."""
    
    def test_accessor_exists(self, sample_df):
        """Test that .ta accessor is available on DataFrame."""
        assert hasattr(sample_df, 'ta')
    
    def test_accessor_with_uppercase_columns(self, uppercase_df):
        """Test that accessor detects uppercase column names."""
        result = uppercase_df.ta.sma(timeperiod=2)
        assert isinstance(result, pd.Series)
        assert len(result) == len(uppercase_df)
    
    def test_accessor_with_short_names(self, short_names_df):
        """Test that accessor detects short column names."""
        result = short_names_df.ta.sma(timeperiod=2)
        assert isinstance(result, pd.Series)
    
    def test_custom_column(self, sample_df):
        """Test using a custom column for calculation."""
        sample_df['custom_price'] = sample_df['close'] * 1.1
        result = sample_df.ta.sma(timeperiod=10, column='custom_price')
        assert isinstance(result, pd.Series)
    
    def test_invalid_column_raises(self, sample_df):
        """Test that specifying invalid column raises error."""
        with pytest.raises(ValueError, match="not found"):
            sample_df.ta.sma(timeperiod=10, column='nonexistent')
    
    def test_missing_ohlcv_column_raises(self):
        """Test that missing required columns raise appropriate errors."""
        df = pd.DataFrame({'price': [1, 2, 3]})
        with pytest.raises(ValueError, match="auto-detect"):
            df.ta.sma(timeperiod=2)


class TestAppendBehavior:
    """Test append=True/False behavior."""
    
    def test_append_false_returns_series(self, sample_df):
        """Test that append=False returns a Series."""
        result = sample_df.ta.sma(timeperiod=10, append=False)
        assert isinstance(result, pd.Series)
        assert 'SMA_10' not in sample_df.columns
    
    def test_append_true_modifies_dataframe(self, sample_df):
        """Test that append=True adds column to DataFrame."""
        original_cols = len(sample_df.columns)
        result = sample_df.ta.sma(timeperiod=10, append=True)
        assert result is None
        assert 'SMA_10' in sample_df.columns
        assert len(sample_df.columns) == original_cols + 1
    
    def test_append_multiple_indicators(self, sample_df):
        """Test appending multiple indicators."""
        sample_df.ta.sma(timeperiod=10, append=True)
        sample_df.ta.ema(timeperiod=10, append=True)
        sample_df.ta.rsi(timeperiod=14, append=True)
        
        assert 'SMA_10' in sample_df.columns
        assert 'EMA_10' in sample_df.columns
        assert 'RSI_14' in sample_df.columns


class TestColumnNaming:
    """Test column naming conventions."""
    
    def test_sma_column_name(self, sample_df):
        """Test SMA column naming."""
        result = sample_df.ta.sma(timeperiod=20)
        assert result.name == 'SMA_20'
    
    def test_ema_column_name(self, sample_df):
        """Test EMA column naming."""
        result = sample_df.ta.ema(timeperiod=12)
        assert result.name == 'EMA_12'
    
    def test_rsi_column_name(self, sample_df):
        """Test RSI column naming."""
        result = sample_df.ta.rsi(timeperiod=14)
        assert result.name == 'RSI_14'
    
    def test_bbands_column_names(self, sample_df):
        """Test Bollinger Bands column naming."""
        result = sample_df.ta.bbands(timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        assert isinstance(result, pd.DataFrame)
        assert 'BBU_20_2.0' in result.columns
        assert 'BBM_20' in result.columns
        assert 'BBL_20_2.0' in result.columns
    
    def test_macd_column_names(self, sample_df):
        """Test MACD column naming."""
        result = sample_df.ta.macd(fastperiod=12, slowperiod=26, signalperiod=9)
        assert isinstance(result, pd.DataFrame)
        assert 'MACD_12_26_9' in result.columns
        assert 'MACDSignal_12_26_9' in result.columns
        assert 'MACDHist_12_26_9' in result.columns
    
    def test_pattern_column_name(self, sample_df):
        """Test pattern recognition column naming."""
        result = sample_df.ta.cdldoji()
        assert result.name == 'CDLDOJI'


class TestOverlapIndicators:
    """Test overlap studies indicators."""
    
    def test_sma(self, sample_df):
        """Test SMA calculation."""
        result = sample_df.ta.sma(timeperiod=10)
        direct = numta.SMA(sample_df['close'].values, timeperiod=10)
        np.testing.assert_array_almost_equal(result.values, direct)
    
    def test_ema(self, sample_df):
        """Test EMA calculation."""
        result = sample_df.ta.ema(timeperiod=10)
        direct = numta.EMA(sample_df['close'].values, timeperiod=10)
        np.testing.assert_array_almost_equal(result.values, direct)
    
    def test_bbands(self, sample_df):
        """Test Bollinger Bands calculation."""
        result = sample_df.ta.bbands(timeperiod=20)
        upper, middle, lower = numta.BBANDS(sample_df['close'].values, timeperiod=20)
        np.testing.assert_array_almost_equal(result['BBU_20_2.0'].values, upper)
        np.testing.assert_array_almost_equal(result['BBM_20'].values, middle)
        np.testing.assert_array_almost_equal(result['BBL_20_2.0'].values, lower)
    
    def test_dema(self, sample_df):
        """Test DEMA calculation."""
        result = sample_df.ta.dema(timeperiod=10)
        direct = numta.DEMA(sample_df['close'].values, timeperiod=10)
        np.testing.assert_array_almost_equal(result.values, direct)
    
    def test_tema(self, sample_df):
        """Test TEMA calculation."""
        result = sample_df.ta.tema(timeperiod=10)
        direct = numta.TEMA(sample_df['close'].values, timeperiod=10)
        np.testing.assert_array_almost_equal(result.values, direct)
    
    def test_wma(self, sample_df):
        """Test WMA calculation."""
        result = sample_df.ta.wma(timeperiod=10)
        direct = numta.WMA(sample_df['close'].values, timeperiod=10)
        np.testing.assert_array_almost_equal(result.values, direct)


class TestMomentumIndicators:
    """Test momentum indicators."""
    
    def test_rsi(self, sample_df):
        """Test RSI calculation."""
        result = sample_df.ta.rsi(timeperiod=14)
        direct = numta.RSI(sample_df['close'].values, timeperiod=14)
        np.testing.assert_array_almost_equal(result.values, direct)
    
    def test_macd(self, sample_df):
        """Test MACD calculation."""
        result = sample_df.ta.macd()
        macd, signal, hist = numta.MACD(sample_df['close'].values)
        np.testing.assert_array_almost_equal(result['MACD_12_26_9'].values, macd)
        np.testing.assert_array_almost_equal(result['MACDSignal_12_26_9'].values, signal)
        np.testing.assert_array_almost_equal(result['MACDHist_12_26_9'].values, hist)
    
    def test_adx(self, sample_df):
        """Test ADX calculation."""
        result = sample_df.ta.adx(timeperiod=14)
        direct = numta.ADX(sample_df['high'].values, sample_df['low'].values, 
                          sample_df['close'].values, timeperiod=14)
        np.testing.assert_array_almost_equal(result.values, direct)
    
    def test_stoch(self, sample_df):
        """Test Stochastic calculation."""
        result = sample_df.ta.stoch()
        slowk, slowd = numta.STOCH(sample_df['high'].values, sample_df['low'].values,
                                   sample_df['close'].values)
        np.testing.assert_array_almost_equal(result['STOCH_SLOWK_5_3_3'].values, slowk)
        np.testing.assert_array_almost_equal(result['STOCH_SLOWD_5_3_3'].values, slowd)
    
    def test_atr(self, sample_df):
        """Test ATR calculation."""
        result = sample_df.ta.atr(timeperiod=14)
        direct = numta.ATR(sample_df['high'].values, sample_df['low'].values,
                          sample_df['close'].values, timeperiod=14)
        np.testing.assert_array_almost_equal(result.values, direct)


class TestVolumeIndicators:
    """Test volume indicators."""
    
    def test_obv(self, sample_df):
        """Test OBV calculation."""
        result = sample_df.ta.obv()
        direct = numta.OBV(sample_df['close'].values, sample_df['volume'].values)
        np.testing.assert_array_almost_equal(result.values, direct)
    
    def test_ad(self, sample_df):
        """Test AD calculation."""
        result = sample_df.ta.ad()
        direct = numta.AD(sample_df['high'].values, sample_df['low'].values,
                         sample_df['close'].values, sample_df['volume'].values)
        np.testing.assert_array_almost_equal(result.values, direct)


class TestPatternRecognition:
    """Test pattern recognition functions."""
    
    def test_cdldoji(self, sample_df):
        """Test CDLDOJI pattern."""
        result = sample_df.ta.cdldoji()
        direct = numta.CDLDOJI(sample_df['open'].values, sample_df['high'].values,
                               sample_df['low'].values, sample_df['close'].values)
        np.testing.assert_array_equal(result.values, direct)
    
    def test_cdlengulfing(self, sample_df):
        """Test CDLENGULFING pattern."""
        result = sample_df.ta.cdlengulfing()
        direct = numta.CDLENGULFING(sample_df['open'].values, sample_df['high'].values,
                                     sample_df['low'].values, sample_df['close'].values)
        np.testing.assert_array_equal(result.values, direct)
    
    def test_cdlhammer(self, sample_df):
        """Test CDLHAMMER pattern."""
        result = sample_df.ta.cdlhammer()
        direct = numta.CDLHAMMER(sample_df['open'].values, sample_df['high'].values,
                                  sample_df['low'].values, sample_df['close'].values)
        np.testing.assert_array_equal(result.values, direct)


class TestStatisticFunctions:
    """Test statistic functions."""
    
    def test_linearreg(self, sample_df):
        """Test Linear Regression."""
        result = sample_df.ta.linearreg(timeperiod=14)
        direct = numta.LINEARREG(sample_df['close'].values, timeperiod=14)
        np.testing.assert_array_almost_equal(result.values, direct)
    
    def test_stddev(self, sample_df):
        """Test Standard Deviation."""
        result = sample_df.ta.stddev(timeperiod=5)
        direct = numta.STDDEV(sample_df['close'].values, timeperiod=5)
        np.testing.assert_array_almost_equal(result.values, direct)


class TestPriceTransform:
    """Test price transform functions."""
    
    def test_typprice(self, sample_df):
        """Test Typical Price."""
        result = sample_df.ta.typprice()
        direct = numta.TYPPRICE(sample_df['high'].values, sample_df['low'].values,
                                sample_df['close'].values)
        np.testing.assert_array_almost_equal(result.values, direct)
    
    def test_medprice(self, sample_df):
        """Test Median Price."""
        result = sample_df.ta.medprice()
        direct = numta.MEDPRICE(sample_df['high'].values, sample_df['low'].values)
        np.testing.assert_array_almost_equal(result.values, direct)


class TestEdgeCases:
    """Test edge cases."""
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame({'open': [], 'high': [], 'low': [], 'close': [], 'volume': []})
        result = df.ta.sma(timeperiod=10)
        assert len(result) == 0
    
    def test_single_row_dataframe(self):
        """Test with single row DataFrame."""
        df = pd.DataFrame({
            'open': [100.0],
            'high': [105.0],
            'low': [99.0],
            'close': [104.0],
            'volume': [1000.0]
        })
        result = df.ta.sma(timeperiod=2)
        assert len(result) == 1
        assert np.isnan(result.iloc[0])
    
    def test_preserves_index(self, sample_df):
        """Test that index is preserved."""
        sample_df.index = pd.date_range('2020-01-01', periods=len(sample_df))
        result = sample_df.ta.sma(timeperiod=10)
        pd.testing.assert_index_equal(result.index, sample_df.index)
