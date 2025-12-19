"""
Pattern annotation and marker helpers for visualization.

This module provides helper functions for creating pattern annotations
and markers that can be used with various charting libraries.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def create_point_marker(
    index: int,
    price: float,
    label: str,
    color: str = 'blue',
    shape: str = 'circle',
    size: int = 8
) -> Dict[str, Any]:
    """
    Create a point marker for pattern visualization.
    
    Parameters
    ----------
    index : int
        Bar index for the marker
    price : float
        Price level for the marker
    label : str
        Text label for the marker
    color : str, optional
        Marker color (default: 'blue')
    shape : str, optional
        Marker shape: 'circle', 'square', 'triangle', 'diamond' (default: 'circle')
    size : int, optional
        Marker size in pixels (default: 8)
    
    Returns
    -------
    dict
        Marker configuration dictionary
    """
    return {
        'type': 'marker',
        'index': index,
        'price': price,
        'label': label,
        'color': color,
        'shape': shape,
        'size': size
    }


def create_trendline(
    start_index: int,
    start_price: float,
    end_index: int,
    end_price: float,
    color: str = 'blue',
    width: int = 1,
    style: str = 'solid',
    extend: bool = False
) -> Dict[str, Any]:
    """
    Create a trendline for pattern visualization.
    
    Parameters
    ----------
    start_index : int
        Starting bar index
    start_price : float
        Starting price
    end_index : int
        Ending bar index
    end_price : float
        Ending price
    color : str, optional
        Line color (default: 'blue')
    width : int, optional
        Line width in pixels (default: 1)
    style : str, optional
        Line style: 'solid', 'dashed', 'dotted' (default: 'solid')
    extend : bool, optional
        Whether to extend line beyond endpoints (default: False)
    
    Returns
    -------
    dict
        Trendline configuration dictionary
    """
    return {
        'type': 'trendline',
        'start_index': start_index,
        'start_price': start_price,
        'end_index': end_index,
        'end_price': end_price,
        'color': color,
        'width': width,
        'style': style,
        'extend': extend
    }


def create_horizontal_line(
    price: float,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
    color: str = 'gray',
    width: int = 1,
    style: str = 'dashed',
    label: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a horizontal line for support/resistance levels.
    
    Parameters
    ----------
    price : float
        Price level for the horizontal line
    start_index : int, optional
        Starting bar index (None for full width)
    end_index : int, optional
        Ending bar index (None for full width)
    color : str, optional
        Line color (default: 'gray')
    width : int, optional
        Line width in pixels (default: 1)
    style : str, optional
        Line style: 'solid', 'dashed', 'dotted' (default: 'dashed')
    label : str, optional
        Optional label for the line
    
    Returns
    -------
    dict
        Horizontal line configuration dictionary
    """
    return {
        'type': 'horizontal_line',
        'price': price,
        'start_index': start_index,
        'end_index': end_index,
        'color': color,
        'width': width,
        'style': style,
        'label': label
    }


def create_zone(
    start_index: int,
    end_index: int,
    upper_price: float,
    lower_price: float,
    color: str = 'rgba(0, 100, 255, 0.1)',
    border_color: Optional[str] = None,
    label: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a price zone/rectangle for pattern visualization.
    
    Parameters
    ----------
    start_index : int
        Starting bar index
    end_index : int
        Ending bar index
    upper_price : float
        Upper price boundary
    lower_price : float
        Lower price boundary
    color : str, optional
        Fill color with opacity (default: 'rgba(0, 100, 255, 0.1)')
    border_color : str, optional
        Border color (None for no border)
    label : str, optional
        Optional label for the zone
    
    Returns
    -------
    dict
        Zone configuration dictionary
    """
    return {
        'type': 'zone',
        'start_index': start_index,
        'end_index': end_index,
        'upper_price': upper_price,
        'lower_price': lower_price,
        'color': color,
        'border_color': border_color,
        'label': label
    }


def create_text_annotation(
    index: int,
    price: float,
    text: str,
    color: str = 'black',
    font_size: int = 12,
    background: Optional[str] = None,
    position: str = 'above'
) -> Dict[str, Any]:
    """
    Create a text annotation for pattern visualization.
    
    Parameters
    ----------
    index : int
        Bar index for the annotation
    price : float
        Price level for the annotation
    text : str
        Text content
    color : str, optional
        Text color (default: 'black')
    font_size : int, optional
        Font size in pixels (default: 12)
    background : str, optional
        Background color (None for transparent)
    position : str, optional
        Position relative to price: 'above', 'below', 'center' (default: 'above')
    
    Returns
    -------
    dict
        Text annotation configuration dictionary
    """
    return {
        'type': 'text',
        'index': index,
        'price': price,
        'text': text,
        'color': color,
        'font_size': font_size,
        'background': background,
        'position': position
    }


def create_pattern_markers(pattern: Any, pattern_type: str) -> List[Dict[str, Any]]:
    """
    Create markers and annotations for a detected pattern.
    
    Parameters
    ----------
    pattern : Any
        Pattern dataclass object (HeadShouldersPattern, DoublePattern, etc.)
    pattern_type : str
        Type of pattern for styling
    
    Returns
    -------
    list
        List of marker/annotation configuration dictionaries
    """
    markers = []
    
    if pattern_type in ('head_shoulders', 'inverse_head_shoulders'):
        # Head and Shoulders pattern markers
        color = 'red' if pattern_type == 'head_shoulders' else 'green'
        
        markers.append(create_point_marker(
            pattern.left_shoulder[0], pattern.left_shoulder[1],
            'LS', color, 'circle', 10
        ))
        markers.append(create_point_marker(
            pattern.head[0], pattern.head[1],
            'H', color, 'circle', 12
        ))
        markers.append(create_point_marker(
            pattern.right_shoulder[0], pattern.right_shoulder[1],
            'RS', color, 'circle', 10
        ))
        
        # Neckline
        slope, intercept = pattern.neckline
        start_idx = pattern.start_index
        end_idx = pattern.end_index
        markers.append(create_trendline(
            start_idx, slope * start_idx + intercept,
            end_idx, slope * end_idx + intercept,
            color, 2, 'dashed', True
        ))
        
        # Breakout level
        markers.append(create_horizontal_line(
            pattern.breakout_level, pattern.start_index, pattern.end_index,
            color, 1, 'dotted', 'Breakout'
        ))
    
    elif pattern_type in ('double_top', 'double_bottom'):
        color = 'red' if pattern_type == 'double_top' else 'green'
        
        markers.append(create_point_marker(
            pattern.first_peak[0], pattern.first_peak[1],
            '1', color, 'circle', 10
        ))
        markers.append(create_point_marker(
            pattern.second_peak[0], pattern.second_peak[1],
            '2', color, 'circle', 10
        ))
        markers.append(create_point_marker(
            pattern.middle_point[0], pattern.middle_point[1],
            'M', 'gray', 'square', 8
        ))
        
        markers.append(create_horizontal_line(
            pattern.neckline, pattern.start_index, pattern.end_index,
            color, 2, 'dashed', 'Neckline'
        ))
    
    elif pattern_type in ('gartley', 'butterfly', 'bat', 'crab'):
        # Harmonic pattern markers
        color = 'green' if pattern.direction == 'bullish' else 'red'
        
        for label, point in [('X', pattern.X), ('A', pattern.A),
                              ('B', pattern.B), ('C', pattern.C), ('D', pattern.D)]:
            markers.append(create_point_marker(
                point[0], point[1], label, color, 'circle', 10
            ))
        
        # Connect the points
        points = [pattern.X, pattern.A, pattern.B, pattern.C, pattern.D]
        for i in range(len(points) - 1):
            markers.append(create_trendline(
                points[i][0], points[i][1],
                points[i + 1][0], points[i + 1][1],
                color, 1, 'solid'
            ))
        
        # PRZ zone - determine fill color based on pattern direction
        prz_fill_rgb = '0, 255, 0' if color == 'green' else '255, 0, 0'
        prz_fill_color = f'rgba({prz_fill_rgb}, 0.1)'
        
        markers.append(create_zone(
            pattern.D[0] - 5, pattern.D[0] + 10,
            pattern.prz[1], pattern.prz[0],
            prz_fill_color,
            color, 'PRZ'
        ))
    
    return markers


def create_fibonacci_markers(
    levels: Dict[str, float],
    start_index: int,
    end_index: int,
    color: str = 'purple'
) -> List[Dict[str, Any]]:
    """
    Create markers for Fibonacci levels.
    
    Parameters
    ----------
    levels : dict
        Fibonacci levels dictionary from fibonacci_retracement()
    start_index : int
        Starting bar index
    end_index : int
        Ending bar index
    color : str, optional
        Color for the lines (default: 'purple')
    
    Returns
    -------
    list
        List of horizontal line configuration dictionaries
    """
    markers = []
    
    level_colors = {
        '0%': 'gray',
        '23.6%': 'green',
        '38.2%': 'blue',
        '50%': 'purple',
        '61.8%': 'orange',
        '78.6%': 'red',
        '100%': 'gray'
    }
    
    for level_name, price in levels.items():
        line_color = level_colors.get(level_name, color)
        markers.append(create_horizontal_line(
            price, start_index, end_index,
            line_color, 1, 'dotted', level_name
        ))
    
    return markers
