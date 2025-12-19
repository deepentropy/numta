"""
Tests for pattern recognition module.
"""

import pytest
import numpy as np

import numta
from numta.patterns import (
    # Swing detection
    find_swing_highs,
    find_swing_lows,
    find_swing_points,
    get_swing_high_indices,
    get_swing_low_indices,
    # Utilities
    fibonacci_retracement,
    fibonacci_extension,
    fit_trendline,
    price_within_tolerance,
    calculate_pattern_confidence,
    # Chart patterns
    detect_head_shoulders,
    detect_inverse_head_shoulders,
    detect_double_top,
    detect_double_bottom,
    detect_triple_top,
    detect_triple_bottom,
    detect_triangle,
    detect_wedge,
    detect_flag,
    detect_vcp,
    # Harmonic patterns
    detect_gartley,
    detect_butterfly,
    detect_bat,
    detect_crab,
    detect_harmonic_patterns,
    # Dataclasses
    HeadShouldersPattern,
    DoublePattern,
    TriplePattern,
    TrianglePattern,
    WedgePattern,
    FlagPattern,
    VCPPattern,
    HarmonicPattern,
)


class TestSwingDetection:
    """Test swing point detection functions."""
    
    def test_find_swing_highs_basic(self):
        """Test basic swing high detection."""
        # Create data with clear swing highs at indices 2 and 6
        high = np.array([10, 12, 15, 14, 11, 13, 16, 14, 12])
        swing_highs = find_swing_highs(high, order=2)
        
        assert isinstance(swing_highs, np.ndarray)
        assert swing_highs.dtype == bool
        assert swing_highs[2] == True  # Peak at 15
        assert swing_highs[6] == True  # Peak at 16
        assert swing_highs[0] == False
    
    def test_find_swing_lows_basic(self):
        """Test basic swing low detection."""
        # Create data with clear swing lows at indices 2 and 6
        low = np.array([12, 10, 8, 9, 11, 9, 7, 8, 10])
        swing_lows = find_swing_lows(low, order=2)
        
        assert isinstance(swing_lows, np.ndarray)
        assert swing_lows.dtype == bool
        assert swing_lows[2] == True  # Trough at 8
        assert swing_lows[6] == True  # Trough at 7
    
    def test_find_swing_points(self):
        """Test combined swing point detection."""
        high = np.array([11, 13, 16, 15, 12, 14, 17, 15, 13])
        low = np.array([10, 12, 15, 14, 11, 13, 16, 14, 12])
        
        swing_highs, swing_lows = find_swing_points(high, low, order=2)
        
        assert isinstance(swing_highs, np.ndarray)
        assert isinstance(swing_lows, np.ndarray)
        assert len(swing_highs) == len(high)
        assert len(swing_lows) == len(low)
    
    def test_get_swing_indices(self):
        """Test getting swing indices."""
        high = np.array([10, 12, 15, 14, 11, 13, 16, 14, 12])
        indices = get_swing_high_indices(high, order=2)
        
        assert isinstance(indices, np.ndarray)
        assert 2 in indices
        assert 6 in indices
    
    def test_swing_detection_insufficient_data(self):
        """Test swing detection with insufficient data."""
        high = np.array([10, 12])
        swing_highs = find_swing_highs(high, order=5)
        
        assert isinstance(swing_highs, np.ndarray)
        assert not np.any(swing_highs)


class TestFibonacciUtils:
    """Test Fibonacci utility functions."""
    
    def test_fibonacci_retracement(self):
        """Test Fibonacci retracement calculation."""
        levels = fibonacci_retracement(100.0, 200.0)
        
        assert '0%' in levels
        assert '23.6%' in levels
        assert '38.2%' in levels
        assert '50%' in levels
        assert '61.8%' in levels
        assert '78.6%' in levels
        assert '100%' in levels
        
        assert levels['0%'] == 100.0
        assert levels['100%'] == 200.0
        assert abs(levels['50%'] - 150.0) < 0.01
        assert abs(levels['61.8%'] - 161.8) < 0.01
    
    def test_fibonacci_extension(self):
        """Test Fibonacci extension calculation."""
        levels = fibonacci_extension(100.0, 150.0, 130.0)
        
        assert '100%' in levels
        assert '127.2%' in levels
        assert '161.8%' in levels
        assert '200%' in levels
        assert '261.8%' in levels
        
        # Check 100% extension: retracement + 1.0 * move
        expected_100 = 130.0 + 1.0 * 50.0  # 180.0
        assert abs(levels['100%'] - expected_100) < 0.01


class TestTrendline:
    """Test trendline fitting functions."""
    
    def test_fit_trendline_basic(self):
        """Test basic trendline fitting."""
        points = [(0, 100), (1, 110), (2, 120), (3, 130)]
        slope, intercept = fit_trendline(points)
        
        assert abs(slope - 10.0) < 0.01
        assert abs(intercept - 100.0) < 0.01
    
    def test_fit_trendline_insufficient_points(self):
        """Test trendline fitting with insufficient points."""
        with pytest.raises(ValueError):
            fit_trendline([(0, 100)])
    
    def test_price_within_tolerance(self):
        """Test price tolerance check."""
        assert price_within_tolerance(100.0, 102.0, 0.03) == True
        assert price_within_tolerance(100.0, 110.0, 0.03) == False
        assert price_within_tolerance(100.0, 100.0, 0.01) == True


class TestPatternConfidence:
    """Test pattern confidence calculation."""
    
    def test_calculate_pattern_confidence(self):
        """Test pattern confidence calculation."""
        pattern_data = {
            'symmetry_score': 0.9,
            'volume_confirmation': 0.7,
            'trendline_fit': 0.8,
            'fibonacci_alignment': 0.6
        }
        
        confidence = calculate_pattern_confidence(pattern_data)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be fairly confident with good scores
    
    def test_calculate_pattern_confidence_empty(self):
        """Test pattern confidence with empty data."""
        confidence = calculate_pattern_confidence({})
        
        assert 0.0 <= confidence <= 1.0
        assert confidence == 0.5  # Default values


class TestChartPatternDetection:
    """Test chart pattern detection functions."""
    
    @pytest.fixture
    def sample_ohlc_data(self):
        """Create sample OHLC data with patterns."""
        np.random.seed(42)
        n = 100
        
        # Generate trending data with some patterns
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.5)
        low = close - np.abs(np.random.randn(n) * 0.5)
        
        return high, low, close
    
    @pytest.fixture
    def head_shoulders_data(self):
        """Create data with a clear head and shoulders pattern."""
        # Create a clear H&S pattern
        n = 50
        close = np.zeros(n)
        
        # Left shoulder: peak around index 10
        close[0:5] = np.linspace(100, 105, 5)
        close[5:10] = np.linspace(105, 110, 5)  # Left shoulder peak
        close[10:15] = np.linspace(110, 105, 5)
        
        # Head: higher peak around index 25
        close[15:20] = np.linspace(105, 110, 5)
        close[20:25] = np.linspace(110, 120, 5)  # Head peak
        close[25:30] = np.linspace(120, 110, 5)
        
        # Right shoulder: similar to left
        close[30:35] = np.linspace(110, 115, 5)
        close[35:40] = np.linspace(115, 110, 5)  # Right shoulder peak
        close[40:45] = np.linspace(110, 105, 5)
        close[45:50] = np.linspace(105, 100, 5)
        
        high = close + 1
        low = close - 1
        
        return high, low, close
    
    @pytest.fixture
    def double_top_data(self):
        """Create data with a double top pattern."""
        n = 40
        close = np.zeros(n)
        
        # First top at index 10
        close[0:5] = np.linspace(100, 105, 5)
        close[5:10] = np.linspace(105, 115, 5)  # First top
        close[10:15] = np.linspace(115, 105, 5)
        
        # Middle valley
        close[15:20] = np.linspace(105, 100, 5)
        
        # Second top at index 30
        close[20:25] = np.linspace(100, 105, 5)
        close[25:30] = np.linspace(105, 115, 5)  # Second top
        close[30:35] = np.linspace(115, 105, 5)
        close[35:40] = np.linspace(105, 95, 5)
        
        high = close + 1
        low = close - 1
        
        return high, low, close
    
    def test_detect_head_shoulders(self, sample_ohlc_data):
        """Test head and shoulders detection."""
        high, low, close = sample_ohlc_data
        patterns = detect_head_shoulders(high, low, close, order=3)
        
        assert isinstance(patterns, list)
        for p in patterns:
            assert isinstance(p, HeadShouldersPattern)
            assert p.pattern_type == 'head_shoulders'
            assert 0.0 <= p.confidence <= 1.0
    
    def test_detect_inverse_head_shoulders(self, sample_ohlc_data):
        """Test inverse head and shoulders detection."""
        high, low, close = sample_ohlc_data
        patterns = detect_inverse_head_shoulders(high, low, close, order=3)
        
        assert isinstance(patterns, list)
        for p in patterns:
            assert isinstance(p, HeadShouldersPattern)
            assert p.pattern_type == 'inverse_head_shoulders'
    
    def test_detect_double_top(self, sample_ohlc_data):
        """Test double top detection."""
        high, low, close = sample_ohlc_data
        patterns = detect_double_top(high, low, close, order=3)
        
        assert isinstance(patterns, list)
        for p in patterns:
            assert isinstance(p, DoublePattern)
            assert p.pattern_type == 'double_top'
    
    def test_detect_double_bottom(self, sample_ohlc_data):
        """Test double bottom detection."""
        high, low, close = sample_ohlc_data
        patterns = detect_double_bottom(high, low, close, order=3)
        
        assert isinstance(patterns, list)
        for p in patterns:
            assert isinstance(p, DoublePattern)
            assert p.pattern_type == 'double_bottom'
    
    def test_detect_triple_top(self, sample_ohlc_data):
        """Test triple top detection."""
        high, low, close = sample_ohlc_data
        patterns = detect_triple_top(high, low, close, order=3)
        
        assert isinstance(patterns, list)
        for p in patterns:
            assert isinstance(p, TriplePattern)
            assert p.pattern_type == 'triple_top'
    
    def test_detect_triple_bottom(self, sample_ohlc_data):
        """Test triple bottom detection."""
        high, low, close = sample_ohlc_data
        patterns = detect_triple_bottom(high, low, close, order=3)
        
        assert isinstance(patterns, list)
        for p in patterns:
            assert isinstance(p, TriplePattern)
            assert p.pattern_type == 'triple_bottom'
    
    def test_detect_triangle(self, sample_ohlc_data):
        """Test triangle pattern detection."""
        high, low, close = sample_ohlc_data
        patterns = detect_triangle(high, low, close, order=3)
        
        assert isinstance(patterns, list)
        for p in patterns:
            assert isinstance(p, TrianglePattern)
            assert p.pattern_type in ('ascending', 'descending', 'symmetrical')
    
    def test_detect_wedge(self, sample_ohlc_data):
        """Test wedge pattern detection."""
        high, low, close = sample_ohlc_data
        patterns = detect_wedge(high, low, close, order=3)
        
        assert isinstance(patterns, list)
        for p in patterns:
            assert isinstance(p, WedgePattern)
            assert p.pattern_type in ('rising', 'falling')
    
    def test_detect_flag(self, sample_ohlc_data):
        """Test flag pattern detection."""
        high, low, close = sample_ohlc_data
        patterns = detect_flag(high, low, close, order=3)
        
        assert isinstance(patterns, list)
        for p in patterns:
            assert isinstance(p, FlagPattern)
            assert p.pattern_type in ('bull_flag', 'bear_flag', 'pennant')
    
    def test_detect_vcp(self, sample_ohlc_data):
        """Test VCP pattern detection."""
        high, low, close = sample_ohlc_data
        patterns = detect_vcp(high, low, close, order=3)
        
        assert isinstance(patterns, list)
        for p in patterns:
            assert isinstance(p, VCPPattern)
            assert len(p.contractions) >= 2


class TestHarmonicPatternDetection:
    """Test harmonic pattern detection functions."""
    
    @pytest.fixture
    def sample_ohlc_data(self):
        """Create sample OHLC data."""
        np.random.seed(42)
        n = 200
        
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.5)
        low = close - np.abs(np.random.randn(n) * 0.5)
        
        return high, low, close
    
    def test_detect_gartley(self, sample_ohlc_data):
        """Test Gartley pattern detection."""
        high, low, close = sample_ohlc_data
        patterns = detect_gartley(high, low, close, order=3, tolerance=0.05)
        
        assert isinstance(patterns, list)
        for p in patterns:
            assert isinstance(p, HarmonicPattern)
            assert p.pattern_type == 'gartley'
            assert p.direction in ('bullish', 'bearish')
    
    def test_detect_butterfly(self, sample_ohlc_data):
        """Test Butterfly pattern detection."""
        high, low, close = sample_ohlc_data
        patterns = detect_butterfly(high, low, close, order=3, tolerance=0.05)
        
        assert isinstance(patterns, list)
        for p in patterns:
            assert isinstance(p, HarmonicPattern)
            assert p.pattern_type == 'butterfly'
    
    def test_detect_bat(self, sample_ohlc_data):
        """Test Bat pattern detection."""
        high, low, close = sample_ohlc_data
        patterns = detect_bat(high, low, close, order=3, tolerance=0.05)
        
        assert isinstance(patterns, list)
        for p in patterns:
            assert isinstance(p, HarmonicPattern)
            assert p.pattern_type == 'bat'
    
    def test_detect_crab(self, sample_ohlc_data):
        """Test Crab pattern detection."""
        high, low, close = sample_ohlc_data
        patterns = detect_crab(high, low, close, order=3, tolerance=0.05)
        
        assert isinstance(patterns, list)
        for p in patterns:
            assert isinstance(p, HarmonicPattern)
            assert p.pattern_type == 'crab'
    
    def test_detect_harmonic_patterns_all(self, sample_ohlc_data):
        """Test detecting all harmonic patterns."""
        high, low, close = sample_ohlc_data
        patterns = detect_harmonic_patterns(high, low, close, order=3, tolerance=0.05)
        
        assert isinstance(patterns, list)
        # Patterns should be sorted by confidence
        if len(patterns) > 1:
            confidences = [p.confidence for p in patterns]
            assert confidences == sorted(confidences, reverse=True)
    
    def test_detect_harmonic_patterns_specific(self, sample_ohlc_data):
        """Test detecting specific harmonic patterns."""
        high, low, close = sample_ohlc_data
        patterns = detect_harmonic_patterns(
            high, low, close,
            patterns=['gartley', 'bat'],
            order=3, tolerance=0.05
        )
        
        assert isinstance(patterns, list)
        for p in patterns:
            assert p.pattern_type in ('gartley', 'bat')


class TestPatternDataclasses:
    """Test pattern dataclass structures."""
    
    def test_head_shoulders_pattern(self):
        """Test HeadShouldersPattern dataclass."""
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
        
        assert pattern.pattern_type == 'head_shoulders'
        assert pattern.head[1] > pattern.left_shoulder[1]
        assert pattern.head[1] > pattern.right_shoulder[1]
    
    def test_double_pattern(self):
        """Test DoublePattern dataclass."""
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
        
        assert pattern.pattern_type == 'double_top'
        assert abs(pattern.first_peak[1] - pattern.second_peak[1]) < 1.0
    
    def test_harmonic_pattern(self):
        """Test HarmonicPattern dataclass."""
        pattern = HarmonicPattern(
            pattern_type='gartley',
            X=(0, 100.0),
            A=(10, 120.0),
            B=(20, 107.6),  # 61.8% retracement
            C=(30, 115.0),
            D=(40, 104.3),  # 78.6% of XA
            direction='bullish',
            XA_retracement=0.618,
            AB_retracement=0.618,
            BC_retracement=1.272,
            CD_retracement=0.786,
            prz=(103.0, 105.0),
            confidence=0.75
        )
        
        assert pattern.pattern_type == 'gartley'
        assert pattern.direction == 'bullish'
        assert 0.0 <= pattern.confidence <= 1.0


class TestModuleExports:
    """Test that all expected symbols are exported from numta."""
    
    def test_swing_functions_exported(self):
        """Test swing detection functions are exported."""
        assert hasattr(numta, 'find_swing_highs')
        assert hasattr(numta, 'find_swing_lows')
        assert hasattr(numta, 'find_swing_points')
    
    def test_utility_functions_exported(self):
        """Test utility functions are exported."""
        assert hasattr(numta, 'fibonacci_retracement')
        assert hasattr(numta, 'fibonacci_extension')
        assert hasattr(numta, 'fit_trendline')
    
    def test_chart_pattern_functions_exported(self):
        """Test chart pattern detection functions are exported."""
        assert hasattr(numta, 'detect_head_shoulders')
        assert hasattr(numta, 'detect_inverse_head_shoulders')
        assert hasattr(numta, 'detect_double_top')
        assert hasattr(numta, 'detect_double_bottom')
        assert hasattr(numta, 'detect_triple_top')
        assert hasattr(numta, 'detect_triple_bottom')
        assert hasattr(numta, 'detect_triangle')
        assert hasattr(numta, 'detect_wedge')
        assert hasattr(numta, 'detect_flag')
        assert hasattr(numta, 'detect_vcp')
    
    def test_harmonic_pattern_functions_exported(self):
        """Test harmonic pattern detection functions are exported."""
        assert hasattr(numta, 'detect_gartley')
        assert hasattr(numta, 'detect_butterfly')
        assert hasattr(numta, 'detect_bat')
        assert hasattr(numta, 'detect_crab')
        assert hasattr(numta, 'detect_harmonic_patterns')
    
    def test_pattern_dataclasses_exported(self):
        """Test pattern dataclasses are exported."""
        assert hasattr(numta, 'HeadShouldersPattern')
        assert hasattr(numta, 'DoublePattern')
        assert hasattr(numta, 'TriplePattern')
        assert hasattr(numta, 'TrianglePattern')
        assert hasattr(numta, 'WedgePattern')
        assert hasattr(numta, 'FlagPattern')
        assert hasattr(numta, 'VCPPattern')
        assert hasattr(numta, 'HarmonicPattern')
