"""
Pattern Recognition module for numta.

This package provides chart pattern detection and harmonic pattern analysis.
"""

from .swing import (
    find_swing_highs,
    find_swing_lows,
    find_swing_points,
    get_swing_high_indices,
    get_swing_low_indices,
    get_swing_points_with_prices,
    find_higher_highs,
    find_lower_lows,
    find_higher_lows,
    find_lower_highs,
)

from .utils import (
    fibonacci_retracement,
    fibonacci_extension,
    fit_trendline,
    calculate_trendline_value,
    calculate_pattern_confidence,
    price_within_tolerance,
    calculate_retracement_ratio,
)

from .chart_patterns import (
    HeadShouldersPattern,
    DoublePattern,
    TriplePattern,
    TrianglePattern,
    WedgePattern,
    FlagPattern,
    VCPPattern,
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
)

from .harmonic_patterns import (
    HarmonicPattern,
    detect_gartley,
    detect_butterfly,
    detect_bat,
    detect_crab,
    detect_harmonic_patterns,
)

__all__ = [
    # Swing detection
    'find_swing_highs',
    'find_swing_lows',
    'find_swing_points',
    'get_swing_high_indices',
    'get_swing_low_indices',
    'get_swing_points_with_prices',
    'find_higher_highs',
    'find_lower_lows',
    'find_higher_lows',
    'find_lower_highs',
    # Utilities
    'fibonacci_retracement',
    'fibonacci_extension',
    'fit_trendline',
    'calculate_trendline_value',
    'calculate_pattern_confidence',
    'price_within_tolerance',
    'calculate_retracement_ratio',
    # Chart pattern dataclasses
    'HeadShouldersPattern',
    'DoublePattern',
    'TriplePattern',
    'TrianglePattern',
    'WedgePattern',
    'FlagPattern',
    'VCPPattern',
    # Chart pattern detection
    'detect_head_shoulders',
    'detect_inverse_head_shoulders',
    'detect_double_top',
    'detect_double_bottom',
    'detect_triple_top',
    'detect_triple_bottom',
    'detect_triangle',
    'detect_wedge',
    'detect_flag',
    'detect_vcp',
    # Harmonic patterns
    'HarmonicPattern',
    'detect_gartley',
    'detect_butterfly',
    'detect_bat',
    'detect_crab',
    'detect_harmonic_patterns',
]
