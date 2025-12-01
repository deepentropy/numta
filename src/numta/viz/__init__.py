"""
Visualization module for numta.

This package provides visualization capabilities for technical analysis,
including chart plotting and pattern visualization.
"""

from .lwcharts_adapter import (
    HAS_LWCHARTS,
    plot_chart,
    plot_pattern,
    plot_harmonic,
)

from .pattern_markers import (
    create_point_marker,
    create_trendline,
    create_horizontal_line,
    create_zone,
    create_text_annotation,
    create_pattern_markers,
    create_fibonacci_markers,
)

__all__ = [
    # lwcharts adapter
    'HAS_LWCHARTS',
    'plot_chart',
    'plot_pattern',
    'plot_harmonic',
    # Pattern markers
    'create_point_marker',
    'create_trendline',
    'create_horizontal_line',
    'create_zone',
    'create_text_annotation',
    'create_pattern_markers',
    'create_fibonacci_markers',
]
