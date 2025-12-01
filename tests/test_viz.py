"""
Tests for visualization module.
"""

import pytest
import numpy as np

from numta.viz import (
    HAS_LWCHARTS,
    plot_chart,
    plot_pattern,
    plot_harmonic,
    create_point_marker,
    create_trendline,
    create_horizontal_line,
    create_zone,
    create_text_annotation,
    create_pattern_markers,
    create_fibonacci_markers,
)


# Module-level random seed for reproducibility
RANDOM_SEED = 42


def _create_sample_ohlcv_data(n: int = 100):
    """Create sample OHLCV data for testing."""
    np.random.seed(RANDOM_SEED)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.5)
    low = close - np.abs(np.random.randn(n) * 0.5)
    open_ = close + np.random.randn(n) * 0.3
    volume = np.random.randint(1000, 10000, n).astype(float)
    return open_, high, low, close, volume


class TestPatternMarkers:
    """Test pattern marker creation functions."""
    
    def test_create_point_marker(self):
        """Test point marker creation."""
        marker = create_point_marker(10, 100.5, 'H', 'red', 'circle', 12)
        
        assert marker['type'] == 'marker'
        assert marker['index'] == 10
        assert marker['price'] == 100.5
        assert marker['label'] == 'H'
        assert marker['color'] == 'red'
        assert marker['shape'] == 'circle'
        assert marker['size'] == 12
    
    def test_create_point_marker_defaults(self):
        """Test point marker with default values."""
        marker = create_point_marker(5, 50.0, 'A')
        
        assert marker['color'] == 'blue'
        assert marker['shape'] == 'circle'
        assert marker['size'] == 8
    
    def test_create_trendline(self):
        """Test trendline creation."""
        line = create_trendline(0, 100.0, 50, 150.0, 'blue', 2, 'dashed', True)
        
        assert line['type'] == 'trendline'
        assert line['start_index'] == 0
        assert line['start_price'] == 100.0
        assert line['end_index'] == 50
        assert line['end_price'] == 150.0
        assert line['color'] == 'blue'
        assert line['width'] == 2
        assert line['style'] == 'dashed'
        assert line['extend'] == True
    
    def test_create_trendline_defaults(self):
        """Test trendline with default values."""
        line = create_trendline(0, 100.0, 50, 150.0)
        
        assert line['color'] == 'blue'
        assert line['width'] == 1
        assert line['style'] == 'solid'
        assert line['extend'] == False
    
    def test_create_horizontal_line(self):
        """Test horizontal line creation."""
        line = create_horizontal_line(105.0, 10, 50, 'red', 2, 'solid', 'Resistance')
        
        assert line['type'] == 'horizontal_line'
        assert line['price'] == 105.0
        assert line['start_index'] == 10
        assert line['end_index'] == 50
        assert line['color'] == 'red'
        assert line['width'] == 2
        assert line['style'] == 'solid'
        assert line['label'] == 'Resistance'
    
    def test_create_horizontal_line_full_width(self):
        """Test horizontal line with full width (no indices)."""
        line = create_horizontal_line(100.0)
        
        assert line['start_index'] is None
        assert line['end_index'] is None
    
    def test_create_zone(self):
        """Test zone/rectangle creation."""
        zone = create_zone(10, 50, 110.0, 100.0, 'rgba(0, 255, 0, 0.2)', 'green', 'PRZ')
        
        assert zone['type'] == 'zone'
        assert zone['start_index'] == 10
        assert zone['end_index'] == 50
        assert zone['upper_price'] == 110.0
        assert zone['lower_price'] == 100.0
        assert zone['color'] == 'rgba(0, 255, 0, 0.2)'
        assert zone['border_color'] == 'green'
        assert zone['label'] == 'PRZ'
    
    def test_create_text_annotation(self):
        """Test text annotation creation."""
        text = create_text_annotation(25, 115.0, 'Head', 'black', 14, 'white', 'above')
        
        assert text['type'] == 'text'
        assert text['index'] == 25
        assert text['price'] == 115.0
        assert text['text'] == 'Head'
        assert text['color'] == 'black'
        assert text['font_size'] == 14
        assert text['background'] == 'white'
        assert text['position'] == 'above'


class TestFibonacciMarkers:
    """Test Fibonacci level marker creation."""
    
    def test_create_fibonacci_markers(self):
        """Test Fibonacci marker creation."""
        levels = {
            '0%': 100.0,
            '23.6%': 123.6,
            '38.2%': 138.2,
            '50%': 150.0,
            '61.8%': 161.8,
            '78.6%': 178.6,
            '100%': 200.0
        }
        
        markers = create_fibonacci_markers(levels, 0, 100)
        
        assert isinstance(markers, list)
        assert len(markers) == 7
        
        for marker in markers:
            assert marker['type'] == 'horizontal_line'
            assert marker['style'] == 'dotted'


class TestPatternMarkersCreation:
    """Test pattern markers creation from pattern objects."""
    
    def test_create_pattern_markers_head_shoulders(self):
        """Test creating markers for head and shoulders pattern."""
        from numta.patterns import HeadShouldersPattern
        
        pattern = HeadShouldersPattern(
            pattern_type='head_shoulders',
            left_shoulder=(10, 110.0),
            head=(20, 120.0),
            right_shoulder=(30, 110.0),
            neckline=(0.1, 100.0),
            neckline_points=[(15, 105.0), (25, 106.0)],
            breakout_level=103.0,
            start_index=10,
            end_index=30,
            confidence=0.85
        )
        
        markers = create_pattern_markers(pattern, 'head_shoulders')
        
        assert isinstance(markers, list)
        assert len(markers) > 0
        
        # Check for expected marker types
        marker_types = [m['type'] for m in markers]
        assert 'marker' in marker_types or 'trendline' in marker_types
    
    def test_create_pattern_markers_double_top(self):
        """Test creating markers for double top pattern."""
        from numta.patterns import DoublePattern
        
        pattern = DoublePattern(
            pattern_type='double_top',
            first_peak=(10, 115.0),
            second_peak=(30, 115.0),
            middle_point=(20, 105.0),
            neckline=105.0,
            breakout_level=105.0,
            start_index=10,
            end_index=30,
            confidence=0.80
        )
        
        markers = create_pattern_markers(pattern, 'double_top')
        
        assert isinstance(markers, list)


class TestPlotFunctions:
    """Test plotting functions."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        try:
            import pandas as pd
            open_, high, low, close, volume = _create_sample_ohlcv_data()
            
            return pd.DataFrame({
                'open': open_,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        except ImportError:
            pytest.skip("pandas not installed")
    
    def test_plot_chart_without_lwcharts(self, sample_df):
        """Test plot_chart returns None gracefully when lwcharts not installed."""
        if HAS_LWCHARTS:
            pytest.skip("lwcharts is installed, skipping graceful degradation test")
        
        result = plot_chart(sample_df)
        assert result is None
    
    def test_plot_pattern_without_lwcharts(self, sample_df):
        """Test plot_pattern returns None gracefully when lwcharts not installed."""
        if HAS_LWCHARTS:
            pytest.skip("lwcharts is installed, skipping graceful degradation test")
        
        result = plot_pattern(sample_df, [])
        assert result is None
    
    def test_plot_harmonic_without_lwcharts(self, sample_df):
        """Test plot_harmonic returns None gracefully when lwcharts not installed."""
        if HAS_LWCHARTS:
            pytest.skip("lwcharts is installed, skipping graceful degradation test")
        
        result = plot_harmonic(sample_df, [])
        assert result is None


class TestHasLwcharts:
    """Test lwcharts availability flag."""
    
    def test_has_lwcharts_is_boolean(self):
        """Test HAS_LWCHARTS is a boolean."""
        assert isinstance(HAS_LWCHARTS, bool)


# Tests for pandas accessor integration
class TestPandasAccessorPatterns:
    """Test pattern detection via pandas accessor."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        try:
            import pandas as pd
            import numta  # noqa: F401 - registers accessor
            
            open_, high, low, close, volume = _create_sample_ohlcv_data()
            
            return pd.DataFrame({
                'open': open_,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        except ImportError:
            pytest.skip("pandas not installed")
    
    def test_find_patterns_accessor(self, sample_df):
        """Test df.ta.find_patterns() method."""
        patterns = sample_df.ta.find_patterns(pattern_type='all', order=3)
        
        assert isinstance(patterns, list)
    
    def test_find_patterns_specific_type(self, sample_df):
        """Test df.ta.find_patterns() with specific pattern type."""
        patterns = sample_df.ta.find_patterns(pattern_type='double', order=3)
        
        assert isinstance(patterns, list)
    
    def test_find_patterns_invalid_type(self, sample_df):
        """Test df.ta.find_patterns() with invalid pattern type."""
        with pytest.raises(ValueError):
            sample_df.ta.find_patterns(pattern_type='invalid_pattern')
    
    def test_find_harmonic_patterns_accessor(self, sample_df):
        """Test df.ta.find_harmonic_patterns() method."""
        patterns = sample_df.ta.find_harmonic_patterns(order=3, tolerance=0.05)
        
        assert isinstance(patterns, list)
    
    def test_find_harmonic_patterns_specific(self, sample_df):
        """Test df.ta.find_harmonic_patterns() with specific patterns."""
        patterns = sample_df.ta.find_harmonic_patterns(
            patterns=['gartley', 'bat'],
            order=3, tolerance=0.05
        )
        
        assert isinstance(patterns, list)
    
    def test_plot_accessor(self, sample_df):
        """Test df.ta.plot() method."""
        # This should work even without lwcharts (returns None)
        result = sample_df.ta.plot()
        
        if not HAS_LWCHARTS:
            assert result is None
    
    def test_plot_accessor_with_indicators(self, sample_df):
        """Test df.ta.plot() with indicators."""
        sma = sample_df.ta.sma(timeperiod=10)
        result = sample_df.ta.plot(indicators={'SMA_10': sma})
        
        if not HAS_LWCHARTS:
            assert result is None
