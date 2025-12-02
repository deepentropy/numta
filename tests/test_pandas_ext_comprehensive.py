"""
Comprehensive tests for pandas DataFrame extension accessor (.ta).

This module provides extensive testing for the pandas integration,
including edge cases, index preservation, and all indicator types.
"""

import pytest
import numpy as np

# Check if pandas is available
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

if HAS_PANDAS:
    import numta


# Skip all tests if pandas is not available
pytestmark = pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")


# =====================================================================
# Fixtures
# =====================================================================

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
def sample_series():
    """Create a sample price Series."""
    np.random.seed(42)
    return pd.Series(100 + np.cumsum(np.random.randn(100) * 0.5), name='close')


@pytest.fixture
def datetime_df():
    """Create a DataFrame with DatetimeIndex."""
    np.random.seed(42)
    n = 100
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.5)
    low = close - np.abs(np.random.randn(n) * 0.5)
    open_ = close + np.random.randn(n) * 0.3
    volume = np.random.randint(1000, 10000, n).astype(float)
    
    index = pd.date_range('2020-01-01', periods=n, freq='D')
    
    return pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=index)


@pytest.fixture
def empty_df():
    """Create an empty DataFrame."""
    return pd.DataFrame({
        'open': pd.Series([], dtype=float),
        'high': pd.Series([], dtype=float),
        'low': pd.Series([], dtype=float),
        'close': pd.Series([], dtype=float),
        'volume': pd.Series([], dtype=float)
    })


@pytest.fixture
def nan_df():
    """Create a DataFrame with NaN values."""
    np.random.seed(42)
    n = 20
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    close[5] = np.nan
    close[10] = np.nan
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


# =====================================================================
# Test Classes
# =====================================================================

class TestAccessorRegistration:
    """Test that accessor is properly registered."""
    
    def test_ta_accessor_exists_on_dataframe(self, sample_df):
        """Test that .ta accessor is available on DataFrame."""
        assert hasattr(sample_df, 'ta')
    
    def test_ta_accessor_callable(self, sample_df):
        """Test that accessor methods are callable."""
        assert callable(getattr(sample_df.ta, 'sma', None))
        assert callable(getattr(sample_df.ta, 'ema', None))
        assert callable(getattr(sample_df.ta, 'rsi', None))
    
    def test_accessor_with_different_dtypes(self):
        """Test accessor works with different column dtypes."""
        df = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [99, 100, 101, 102, 103],
            'close': [104, 105, 106, 107, 108],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        # Should work with integer columns
        assert hasattr(df, 'ta')
        result = df.ta.sma(timeperiod=2)
        assert isinstance(result, pd.Series)


class TestSeriesIndicators:
    """Test indicators on Series objects."""
    
    def test_sma_on_series(self, sample_df):
        """Test SMA on Series via DataFrame accessor."""
        result = sample_df.ta.sma(timeperiod=10)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_df)
        assert result.name == 'SMA_10'
    
    def test_ema_on_series(self, sample_df):
        """Test EMA on Series via DataFrame accessor."""
        result = sample_df.ta.ema(timeperiod=10)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_df)
        assert result.name == 'EMA_10'
    
    def test_rsi_on_series(self, sample_df):
        """Test RSI calculation."""
        result = sample_df.ta.rsi(timeperiod=14)
        assert isinstance(result, pd.Series)
        
        # RSI should be bounded between 0 and 100
        valid_values = result.dropna()
        assert (valid_values >= 0).all() and (valid_values <= 100).all()


class TestOHLCIndicators:
    """Test indicators requiring OHLC data."""
    
    def test_atr_calculation(self, sample_df):
        """Test ATR calculation using high, low, close."""
        result = sample_df.ta.atr(timeperiod=14)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_df)
        
        # ATR should be positive
        valid_values = result.dropna()
        assert (valid_values >= 0).all()
    
    def test_adx_calculation(self, sample_df):
        """Test ADX calculation."""
        result = sample_df.ta.adx(timeperiod=14)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_df)
    
    def test_stoch_calculation(self, sample_df):
        """Test Stochastic calculation."""
        result = sample_df.ta.stoch()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_df)
        assert 'STOCH_SLOWK_5_3_3' in result.columns
        assert 'STOCH_SLOWD_5_3_3' in result.columns
    
    def test_bbands_calculation(self, sample_df):
        """Test Bollinger Bands calculation."""
        result = sample_df.ta.bbands(timeperiod=20)
        assert isinstance(result, pd.DataFrame)
        assert 'BBU_20_2.0' in result.columns
        assert 'BBM_20' in result.columns
        assert 'BBL_20_2.0' in result.columns
        
        # Upper should be above middle, middle above lower
        valid_mask = ~result.isna().any(axis=1)
        assert (result.loc[valid_mask, 'BBU_20_2.0'] >= result.loc[valid_mask, 'BBM_20']).all()
        assert (result.loc[valid_mask, 'BBM_20'] >= result.loc[valid_mask, 'BBL_20_2.0']).all()


class TestIndexPreservation:
    """Test that index is preserved in results."""
    
    def test_datetime_index_preserved(self, datetime_df):
        """Test that DatetimeIndex is preserved."""
        result = datetime_df.ta.sma(timeperiod=10)
        pd.testing.assert_index_equal(result.index, datetime_df.index)
    
    def test_custom_integer_index_preserved(self, sample_df):
        """Test that custom integer index is preserved."""
        sample_df.index = range(100, 200)
        result = sample_df.ta.sma(timeperiod=10)
        pd.testing.assert_index_equal(result.index, sample_df.index)
    
    def test_string_index_preserved(self, sample_df):
        """Test that string index is preserved."""
        sample_df.index = [f'row_{i}' for i in range(len(sample_df))]
        result = sample_df.ta.sma(timeperiod=10)
        pd.testing.assert_index_equal(result.index, sample_df.index)
    
    def test_multi_output_index_preserved(self, datetime_df):
        """Test that multi-output functions preserve index."""
        result = datetime_df.ta.bbands(timeperiod=10)
        pd.testing.assert_index_equal(result.index, datetime_df.index)


class TestEdgeCases:
    """Test edge cases for pandas extension."""
    
    def test_empty_dataframe(self, empty_df):
        """Test handling of empty DataFrame."""
        result = empty_df.ta.sma(timeperiod=10)
        assert len(result) == 0
        assert isinstance(result, pd.Series)
    
    def test_nan_handling(self, nan_df):
        """Test handling of NaN values in data."""
        result = nan_df.ta.sma(timeperiod=5)
        assert isinstance(result, pd.Series)
        assert len(result) == len(nan_df)
    
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
        assert pd.isna(result.iloc[0])
    
    def test_timeperiod_larger_than_data(self, sample_df):
        """Test when timeperiod is larger than data length."""
        # Use a smaller subset to avoid potential JIT issues
        small_df = sample_df.head(50)
        result = small_df.ta.sma(timeperiod=100)
        assert len(result) == len(small_df)
        # All values should be NaN when timeperiod > data length
        assert result.isna().all()
    
    def test_missing_columns_error(self):
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
        sample_df.ta.sma(timeperiod=10, append=True)
        assert 'SMA_10' in sample_df.columns
        assert len(sample_df.columns) == original_cols + 1
    
    def test_append_returns_none(self, sample_df):
        """Test that append=True returns None."""
        result = sample_df.ta.sma(timeperiod=10, append=True)
        assert result is None
    
    def test_multiple_appends(self, sample_df):
        """Test multiple indicator appends."""
        sample_df.ta.sma(timeperiod=10, append=True)
        sample_df.ta.ema(timeperiod=10, append=True)
        sample_df.ta.rsi(timeperiod=14, append=True)
        
        assert 'SMA_10' in sample_df.columns
        assert 'EMA_10' in sample_df.columns
        assert 'RSI_14' in sample_df.columns
    
    def test_append_multi_output(self, sample_df):
        """Test appending multi-output indicators."""
        sample_df.ta.bbands(timeperiod=20, append=True)
        assert 'BBU_20_2.0' in sample_df.columns
        assert 'BBM_20' in sample_df.columns
        assert 'BBL_20_2.0' in sample_df.columns


class TestCustomColumn:
    """Test using custom columns."""
    
    def test_custom_column_name(self, sample_df):
        """Test using a custom column for calculation."""
        sample_df['custom_price'] = sample_df['close'] * 1.1
        result = sample_df.ta.sma(timeperiod=10, column='custom_price')
        
        assert isinstance(result, pd.Series)
        # Result should be different from close-based SMA
        close_sma = sample_df.ta.sma(timeperiod=10)
        assert not np.allclose(result.dropna().values, close_sma.dropna().values)
    
    def test_invalid_column_raises(self, sample_df):
        """Test that specifying invalid column raises error."""
        with pytest.raises(ValueError, match="not found"):
            sample_df.ta.sma(timeperiod=10, column='nonexistent')


class TestVolumeIndicators:
    """Test volume-based indicators."""
    
    def test_obv_calculation(self, sample_df):
        """Test OBV calculation."""
        result = sample_df.ta.obv()
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_df)
    
    def test_ad_calculation(self, sample_df):
        """Test AD calculation."""
        result = sample_df.ta.ad()
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_df)
    
    def test_mfi_calculation(self, sample_df):
        """Test MFI calculation."""
        result = sample_df.ta.mfi(timeperiod=14)
        assert isinstance(result, pd.Series)
        
        # MFI should be bounded between 0 and 100
        valid_values = result.dropna()
        assert (valid_values >= 0).all() and (valid_values <= 100).all()


class TestCandlestickPatterns:
    """Test candlestick pattern recognition."""
    
    def test_cdldoji_pattern(self, sample_df):
        """Test CDLDOJI pattern detection."""
        result = sample_df.ta.cdldoji()
        assert isinstance(result, pd.Series)
        assert result.name == 'CDLDOJI'
        
        # Pattern should return -100, 0, or 100
        unique_values = set(result.dropna().unique().astype(int))
        assert unique_values.issubset({-100, 0, 100})
    
    def test_cdlengulfing_pattern(self, sample_df):
        """Test CDLENGULFING pattern detection."""
        result = sample_df.ta.cdlengulfing()
        assert isinstance(result, pd.Series)
        assert result.name == 'CDLENGULFING'


class TestResultsMatchNumta:
    """Test that results match direct numta calls."""
    
    def test_sma_matches_numta(self, sample_df):
        """Test SMA matches direct numta call."""
        accessor_result = sample_df.ta.sma(timeperiod=10)
        direct_result = numta.SMA(sample_df['close'].values, timeperiod=10)
        np.testing.assert_array_almost_equal(accessor_result.values, direct_result)
    
    def test_ema_matches_numta(self, sample_df):
        """Test EMA matches direct numta call."""
        accessor_result = sample_df.ta.ema(timeperiod=10)
        direct_result = numta.EMA(sample_df['close'].values, timeperiod=10)
        np.testing.assert_array_almost_equal(accessor_result.values, direct_result)
    
    def test_rsi_matches_numta(self, sample_df):
        """Test RSI matches direct numta call."""
        accessor_result = sample_df.ta.rsi(timeperiod=14)
        direct_result = numta.RSI(sample_df['close'].values, timeperiod=14)
        np.testing.assert_array_almost_equal(accessor_result.values, direct_result)
    
    def test_atr_matches_numta(self, sample_df):
        """Test ATR matches direct numta call."""
        accessor_result = sample_df.ta.atr(timeperiod=14)
        direct_result = numta.ATR(
            sample_df['high'].values,
            sample_df['low'].values,
            sample_df['close'].values,
            timeperiod=14
        )
        np.testing.assert_array_almost_equal(accessor_result.values, direct_result)
    
    def test_bbands_matches_numta(self, sample_df):
        """Test BBANDS matches direct numta call."""
        accessor_result = sample_df.ta.bbands(timeperiod=20)
        upper, middle, lower = numta.BBANDS(sample_df['close'].values, timeperiod=20)
        
        np.testing.assert_array_almost_equal(accessor_result['BBU_20_2.0'].values, upper)
        np.testing.assert_array_almost_equal(accessor_result['BBM_20'].values, middle)
        np.testing.assert_array_almost_equal(accessor_result['BBL_20_2.0'].values, lower)
