"""
Utility functions for pattern detection.

This module provides helper functions for pattern detection including:
- Fibonacci retracement and extension calculations
- Trendline fitting
- Pattern confidence scoring
"""

from typing import Dict, List, Tuple
import numpy as np


def fibonacci_retracement(start: float, end: float) -> Dict[str, float]:
    """
    Calculate Fibonacci retracement levels between two price points.
    
    Parameters
    ----------
    start : float
        Starting price (swing high or low)
    end : float
        Ending price (swing low or high)
    
    Returns
    -------
    dict
        Dictionary with Fibonacci levels as keys and prices as values
        Keys: '0%', '23.6%', '38.2%', '50%', '61.8%', '78.6%', '100%'
    """
    diff = end - start
    return {
        '0%': start,
        '23.6%': start + diff * 0.236,
        '38.2%': start + diff * 0.382,
        '50%': start + diff * 0.5,
        '61.8%': start + diff * 0.618,
        '78.6%': start + diff * 0.786,
        '100%': end
    }


def fibonacci_extension(start: float, end: float, retracement: float) -> Dict[str, float]:
    """
    Calculate Fibonacci extension levels from a retracement point.
    
    Parameters
    ----------
    start : float
        Starting price (point A)
    end : float
        Ending price (point B)
    retracement : float
        Retracement price (point C)
    
    Returns
    -------
    dict
        Dictionary with Fibonacci extension levels as keys and prices as values
        Keys: '100%', '127.2%', '161.8%', '200%', '261.8%'
    """
    # The move from start to end
    ab_diff = end - start
    # Direction of extension (same as original move)
    direction = 1 if ab_diff > 0 else -1
    ab_length = abs(ab_diff)
    
    return {
        '100%': retracement + direction * ab_length * 1.0,
        '127.2%': retracement + direction * ab_length * 1.272,
        '161.8%': retracement + direction * ab_length * 1.618,
        '200%': retracement + direction * ab_length * 2.0,
        '261.8%': retracement + direction * ab_length * 2.618
    }


def fit_trendline(points: List[Tuple[int, float]]) -> Tuple[float, float]:
    """
    Fit a linear trendline through a set of points using least squares.
    
    Parameters
    ----------
    points : list of tuple
        List of (index, price) tuples
    
    Returns
    -------
    tuple
        (slope, intercept) of the fitted line
        The line equation is: price = slope * index + intercept
    """
    if len(points) < 2:
        raise ValueError("At least 2 points required to fit a trendline")
    
    x = np.array([p[0] for p in points], dtype=np.float64)
    y = np.array([p[1] for p in points], dtype=np.float64)
    
    # Least squares: y = slope * x + intercept
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xx = np.sum(x * x)
    sum_xy = np.sum(x * y)
    
    denominator = n * sum_xx - sum_x * sum_x
    if denominator == 0:
        # Vertical line case - all x values are the same
        return 0.0, np.mean(y)
    
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n
    
    return slope, intercept


def calculate_trendline_value(slope: float, intercept: float, index: int) -> float:
    """
    Calculate the price value on a trendline at a given index.
    
    Parameters
    ----------
    slope : float
        Slope of the trendline
    intercept : float
        Intercept of the trendline
    index : int
        Index at which to calculate the price
    
    Returns
    -------
    float
        Price value at the given index
    """
    return slope * index + intercept


def calculate_pattern_confidence(
    pattern_data: dict,
    price_tolerance: float = 0.02,
    symmetry_weight: float = 0.3,
    volume_weight: float = 0.2,
    trendline_weight: float = 0.3,
    fibonacci_weight: float = 0.2
) -> float:
    """
    Calculate a confidence score for a detected pattern.
    
    Parameters
    ----------
    pattern_data : dict
        Dictionary containing pattern characteristics:
        - 'symmetry_score': How symmetric the pattern is (0-1)
        - 'volume_confirmation': Volume confirmation score (0-1)
        - 'trendline_fit': R-squared of trendline fits (0-1)
        - 'fibonacci_alignment': How well points align to Fibonacci (0-1)
    price_tolerance : float
        Price tolerance for matching (not directly used but available)
    symmetry_weight : float
        Weight for symmetry score
    volume_weight : float
        Weight for volume confirmation
    trendline_weight : float
        Weight for trendline fit
    fibonacci_weight : float
        Weight for Fibonacci alignment
    
    Returns
    -------
    float
        Confidence score between 0 and 1
    """
    symmetry = pattern_data.get('symmetry_score', 0.5)
    volume = pattern_data.get('volume_confirmation', 0.5)
    trendline = pattern_data.get('trendline_fit', 0.5)
    fibonacci = pattern_data.get('fibonacci_alignment', 0.5)
    
    # Weighted average
    confidence = (
        symmetry * symmetry_weight +
        volume * volume_weight +
        trendline * trendline_weight +
        fibonacci * fibonacci_weight
    )
    
    # Normalize to ensure total weights sum to 1
    total_weight = symmetry_weight + volume_weight + trendline_weight + fibonacci_weight
    if total_weight > 0:
        confidence = confidence / total_weight
    
    return min(1.0, max(0.0, confidence))


def price_within_tolerance(price1: float, price2: float, tolerance: float = 0.02) -> bool:
    """
    Check if two prices are within a given tolerance percentage.
    
    Parameters
    ----------
    price1 : float
        First price
    price2 : float
        Second price
    tolerance : float
        Tolerance as a decimal (e.g., 0.02 = 2%)
    
    Returns
    -------
    bool
        True if prices are within tolerance
    """
    epsilon = 1e-10
    if abs(price1) < epsilon:
        return abs(price2) < epsilon
    return abs(price1 - price2) / abs(price1) <= tolerance


def calculate_retracement_ratio(
    start_price: float,
    end_price: float,
    current_price: float
) -> float:
    """
    Calculate the retracement ratio from start to end.
    
    Parameters
    ----------
    start_price : float
        Starting price of the move
    end_price : float
        Ending price of the move
    current_price : float
        Current price to measure retracement
    
    Returns
    -------
    float
        Retracement ratio (0 = no retracement, 1 = full retracement)
    """
    move = end_price - start_price
    if move == 0:
        return 0.0
    retracement = current_price - end_price
    return abs(retracement / move)
