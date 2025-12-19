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

from .chart import (
    plot_ohlc,
    plot_line,
    plot_with_indicators,
)

from .streaming_chart import (
    StreamingChart,
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
    # Chart functions
    'plot_ohlc',
    'plot_line',
    'plot_with_indicators',
    # Streaming chart
    'StreamingChart',
    # Pattern markers
    'create_point_marker',
    'create_trendline',
    'create_horizontal_line',
    'create_zone',
    'create_text_annotation',
    'create_pattern_markers',
    'create_fibonacci_markers',
]
