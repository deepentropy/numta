"""
Chart pattern detection module.

This module provides detection functions for common chart patterns including:
- Head and Shoulders (regular and inverse)
- Double Top/Bottom
- Triple Top/Bottom
- Triangles (ascending, descending, symmetrical)
- Wedges (rising, falling)
- Flags (bull flag, bear flag, pennant)
- VCP (Volatility Contraction Pattern)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from .swing import get_swing_high_indices, get_swing_low_indices
from .utils import fit_trendline, price_within_tolerance, calculate_pattern_confidence


@dataclass
class HeadShouldersPattern:
    """
    Head and Shoulders pattern data structure.
    
    Attributes
    ----------
    pattern_type : str
        'head_shoulders' for bearish, 'inverse_head_shoulders' for bullish
    left_shoulder : tuple
        (index, price) of left shoulder
    head : tuple
        (index, price) of head
    right_shoulder : tuple
        (index, price) of right shoulder
    neckline : tuple
        (slope, intercept) of neckline
    neckline_points : list
        List of (index, price) points used for neckline
    breakout_level : float
        Price level where breakout occurs
    start_index : int
        Starting index of pattern
    end_index : int
        Ending index of pattern
    confidence : float
        Confidence score (0-1)
    """
    pattern_type: str
    left_shoulder: Tuple[int, float]
    head: Tuple[int, float]
    right_shoulder: Tuple[int, float]
    neckline: Tuple[float, float]
    neckline_points: List[Tuple[int, float]]
    breakout_level: float
    start_index: int
    end_index: int
    confidence: float


@dataclass
class DoublePattern:
    """
    Double Top/Bottom pattern data structure.
    
    Attributes
    ----------
    pattern_type : str
        'double_top' or 'double_bottom'
    first_peak : tuple
        (index, price) of first peak/trough
    second_peak : tuple
        (index, price) of second peak/trough
    middle_point : tuple
        (index, price) of middle point between peaks
    neckline : float
        Neckline price level
    breakout_level : float
        Price level where breakout occurs
    start_index : int
        Starting index of pattern
    end_index : int
        Ending index of pattern
    confidence : float
        Confidence score (0-1)
    """
    pattern_type: str
    first_peak: Tuple[int, float]
    second_peak: Tuple[int, float]
    middle_point: Tuple[int, float]
    neckline: float
    breakout_level: float
    start_index: int
    end_index: int
    confidence: float


@dataclass
class TriplePattern:
    """
    Triple Top/Bottom pattern data structure.
    
    Attributes
    ----------
    pattern_type : str
        'triple_top' or 'triple_bottom'
    first_peak : tuple
        (index, price) of first peak/trough
    second_peak : tuple
        (index, price) of second peak/trough
    third_peak : tuple
        (index, price) of third peak/trough
    support_resistance : float
        Support/resistance level
    breakout_level : float
        Price level where breakout occurs
    start_index : int
        Starting index of pattern
    end_index : int
        Ending index of pattern
    confidence : float
        Confidence score (0-1)
    """
    pattern_type: str
    first_peak: Tuple[int, float]
    second_peak: Tuple[int, float]
    third_peak: Tuple[int, float]
    support_resistance: float
    breakout_level: float
    start_index: int
    end_index: int
    confidence: float


@dataclass
class TrianglePattern:
    """
    Triangle pattern data structure.
    
    Attributes
    ----------
    pattern_type : str
        'ascending', 'descending', or 'symmetrical'
    upper_trendline : tuple
        (slope, intercept) of upper trendline
    lower_trendline : tuple
        (slope, intercept) of lower trendline
    upper_points : list
        List of (index, price) points on upper trendline
    lower_points : list
        List of (index, price) points on lower trendline
    apex_index : int
        Index where trendlines would converge
    breakout_level : float
        Expected breakout price level
    start_index : int
        Starting index of pattern
    end_index : int
        Ending index of pattern
    confidence : float
        Confidence score (0-1)
    """
    pattern_type: str
    upper_trendline: Tuple[float, float]
    lower_trendline: Tuple[float, float]
    upper_points: List[Tuple[int, float]]
    lower_points: List[Tuple[int, float]]
    apex_index: int
    breakout_level: float
    start_index: int
    end_index: int
    confidence: float


@dataclass
class WedgePattern:
    """
    Wedge pattern data structure.
    
    Attributes
    ----------
    pattern_type : str
        'rising' or 'falling'
    upper_trendline : tuple
        (slope, intercept) of upper trendline
    lower_trendline : tuple
        (slope, intercept) of lower trendline
    upper_points : list
        List of (index, price) points on upper trendline
    lower_points : list
        List of (index, price) points on lower trendline
    breakout_level : float
        Expected breakout price level
    start_index : int
        Starting index of pattern
    end_index : int
        Ending index of pattern
    confidence : float
        Confidence score (0-1)
    """
    pattern_type: str
    upper_trendline: Tuple[float, float]
    lower_trendline: Tuple[float, float]
    upper_points: List[Tuple[int, float]]
    lower_points: List[Tuple[int, float]]
    breakout_level: float
    start_index: int
    end_index: int
    confidence: float


@dataclass
class FlagPattern:
    """
    Flag/Pennant pattern data structure.
    
    Attributes
    ----------
    pattern_type : str
        'bull_flag', 'bear_flag', or 'pennant'
    pole_start : tuple
        (index, price) of flagpole start
    pole_end : tuple
        (index, price) of flagpole end
    flag_upper : tuple
        (slope, intercept) of upper flag boundary
    flag_lower : tuple
        (slope, intercept) of lower flag boundary
    breakout_level : float
        Expected breakout price level
    target : float
        Price target based on pole length
    start_index : int
        Starting index of pattern
    end_index : int
        Ending index of pattern
    confidence : float
        Confidence score (0-1)
    """
    pattern_type: str
    pole_start: Tuple[int, float]
    pole_end: Tuple[int, float]
    flag_upper: Tuple[float, float]
    flag_lower: Tuple[float, float]
    breakout_level: float
    target: float
    start_index: int
    end_index: int
    confidence: float


@dataclass
class VCPPattern:
    """
    Volatility Contraction Pattern (VCP) data structure.
    
    Attributes
    ----------
    contractions : list
        List of (high_index, low_index, range) tuples for each contraction
    pivot_point : tuple
        (index, price) of the pivot/buy point
    breakout_level : float
        Price level where breakout occurs
    depth_percentages : list
        List of depth percentages for each contraction
    start_index : int
        Starting index of pattern
    end_index : int
        Ending index of pattern
    confidence : float
        Confidence score (0-1)
    """
    contractions: List[Tuple[int, int, float]]
    pivot_point: Tuple[int, float]
    breakout_level: float
    depth_percentages: List[float]
    start_index: int
    end_index: int
    confidence: float


def detect_head_shoulders(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    order: int = 5,
    tolerance: float = 0.03
) -> List[HeadShouldersPattern]:
    """
    Detect Head and Shoulders patterns (bearish reversal).
    
    Parameters
    ----------
    high : np.ndarray
        Array of high prices
    low : np.ndarray
        Array of low prices
    close : np.ndarray
        Array of close prices
    order : int, optional
        Swing detection order (default: 5)
    tolerance : float, optional
        Price tolerance for shoulder matching (default: 0.03)
    
    Returns
    -------
    list
        List of HeadShouldersPattern objects
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    patterns = []
    
    swing_high_indices = get_swing_high_indices(high, order)
    swing_low_indices = get_swing_low_indices(low, order)
    
    if len(swing_high_indices) < 3 or len(swing_low_indices) < 2:
        return patterns
    
    # Look for pattern: left_shoulder < head > right_shoulder
    for i in range(len(swing_high_indices) - 2):
        ls_idx = swing_high_indices[i]
        head_idx = swing_high_indices[i + 1]
        rs_idx = swing_high_indices[i + 2]
        
        ls_price = high[ls_idx]
        head_price = high[head_idx]
        rs_price = high[rs_idx]
        
        # Head must be higher than both shoulders
        if head_price <= ls_price or head_price <= rs_price:
            continue
        
        # Shoulders should be roughly equal
        if not price_within_tolerance(ls_price, rs_price, tolerance):
            continue
        
        # Find neckline points (lows between shoulders and head)
        neckline_points = []
        for low_idx in swing_low_indices:
            if ls_idx < low_idx < head_idx:
                neckline_points.append((low_idx, low[low_idx]))
            elif head_idx < low_idx < rs_idx:
                neckline_points.append((low_idx, low[low_idx]))
        
        if len(neckline_points) < 2:
            continue
        
        # Fit neckline
        try:
            slope, intercept = fit_trendline(neckline_points)
        except ValueError:
            continue
        
        # Calculate breakout level at right shoulder position
        breakout_level = slope * rs_idx + intercept
        
        # Calculate confidence
        shoulder_symmetry = 1 - abs(ls_price - rs_price) / head_price
        pattern_data = {
            'symmetry_score': shoulder_symmetry,
            'trendline_fit': 0.7,  # Placeholder
            'volume_confirmation': 0.5,
            'fibonacci_alignment': 0.5
        }
        confidence = calculate_pattern_confidence(pattern_data)
        
        pattern = HeadShouldersPattern(
            pattern_type='head_shoulders',
            left_shoulder=(ls_idx, ls_price),
            head=(head_idx, head_price),
            right_shoulder=(rs_idx, rs_price),
            neckline=(slope, intercept),
            neckline_points=neckline_points,
            breakout_level=breakout_level,
            start_index=ls_idx,
            end_index=rs_idx,
            confidence=confidence
        )
        patterns.append(pattern)
    
    return patterns


def detect_inverse_head_shoulders(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    order: int = 5,
    tolerance: float = 0.03
) -> List[HeadShouldersPattern]:
    """
    Detect Inverse Head and Shoulders patterns (bullish reversal).
    
    Parameters
    ----------
    high : np.ndarray
        Array of high prices
    low : np.ndarray
        Array of low prices
    close : np.ndarray
        Array of close prices
    order : int, optional
        Swing detection order (default: 5)
    tolerance : float, optional
        Price tolerance for shoulder matching (default: 0.03)
    
    Returns
    -------
    list
        List of HeadShouldersPattern objects
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    patterns = []
    
    swing_high_indices = get_swing_high_indices(high, order)
    swing_low_indices = get_swing_low_indices(low, order)
    
    if len(swing_low_indices) < 3 or len(swing_high_indices) < 2:
        return patterns
    
    # Look for pattern: left_shoulder > head < right_shoulder (inverted)
    for i in range(len(swing_low_indices) - 2):
        ls_idx = swing_low_indices[i]
        head_idx = swing_low_indices[i + 1]
        rs_idx = swing_low_indices[i + 2]
        
        ls_price = low[ls_idx]
        head_price = low[head_idx]
        rs_price = low[rs_idx]
        
        # Head must be lower than both shoulders
        if head_price >= ls_price or head_price >= rs_price:
            continue
        
        # Shoulders should be roughly equal
        if not price_within_tolerance(ls_price, rs_price, tolerance):
            continue
        
        # Find neckline points (highs between shoulders and head)
        neckline_points = []
        for high_idx in swing_high_indices:
            if ls_idx < high_idx < head_idx:
                neckline_points.append((high_idx, high[high_idx]))
            elif head_idx < high_idx < rs_idx:
                neckline_points.append((high_idx, high[high_idx]))
        
        if len(neckline_points) < 2:
            continue
        
        # Fit neckline
        try:
            slope, intercept = fit_trendline(neckline_points)
        except ValueError:
            continue
        
        # Calculate breakout level at right shoulder position
        breakout_level = slope * rs_idx + intercept
        
        # Calculate confidence
        shoulder_symmetry = 1 - abs(ls_price - rs_price) / abs(head_price) if head_price != 0 else 0
        pattern_data = {
            'symmetry_score': min(1.0, shoulder_symmetry),
            'trendline_fit': 0.7,
            'volume_confirmation': 0.5,
            'fibonacci_alignment': 0.5
        }
        confidence = calculate_pattern_confidence(pattern_data)
        
        pattern = HeadShouldersPattern(
            pattern_type='inverse_head_shoulders',
            left_shoulder=(ls_idx, ls_price),
            head=(head_idx, head_price),
            right_shoulder=(rs_idx, rs_price),
            neckline=(slope, intercept),
            neckline_points=neckline_points,
            breakout_level=breakout_level,
            start_index=ls_idx,
            end_index=rs_idx,
            confidence=confidence
        )
        patterns.append(pattern)
    
    return patterns


def detect_double_top(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    order: int = 5,
    tolerance: float = 0.02
) -> List[DoublePattern]:
    """
    Detect Double Top patterns (bearish reversal).
    
    Parameters
    ----------
    high : np.ndarray
        Array of high prices
    low : np.ndarray
        Array of low prices
    close : np.ndarray
        Array of close prices
    order : int, optional
        Swing detection order (default: 5)
    tolerance : float, optional
        Price tolerance for peak matching (default: 0.02)
    
    Returns
    -------
    list
        List of DoublePattern objects
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    patterns = []
    
    swing_high_indices = get_swing_high_indices(high, order)
    swing_low_indices = get_swing_low_indices(low, order)
    
    if len(swing_high_indices) < 2:
        return patterns
    
    for i in range(len(swing_high_indices) - 1):
        first_idx = swing_high_indices[i]
        second_idx = swing_high_indices[i + 1]
        
        first_price = high[first_idx]
        second_price = high[second_idx]
        
        # Peaks should be roughly equal
        if not price_within_tolerance(first_price, second_price, tolerance):
            continue
        
        # Find the middle low point
        middle_lows = [idx for idx in swing_low_indices 
                       if first_idx < idx < second_idx]
        
        if not middle_lows:
            continue
        
        # Use the lowest point as the middle
        middle_idx = min(middle_lows, key=lambda x: low[x])
        middle_price = low[middle_idx]
        neckline = middle_price
        
        # Calculate confidence based on peak similarity
        peak_diff = abs(first_price - second_price) / first_price
        confidence = max(0.0, min(1.0, 1 - peak_diff * 10))
        
        pattern = DoublePattern(
            pattern_type='double_top',
            first_peak=(first_idx, first_price),
            second_peak=(second_idx, second_price),
            middle_point=(middle_idx, middle_price),
            neckline=neckline,
            breakout_level=neckline,
            start_index=first_idx,
            end_index=second_idx,
            confidence=confidence
        )
        patterns.append(pattern)
    
    return patterns


def detect_double_bottom(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    order: int = 5,
    tolerance: float = 0.02
) -> List[DoublePattern]:
    """
    Detect Double Bottom patterns (bullish reversal).
    
    Parameters
    ----------
    high : np.ndarray
        Array of high prices
    low : np.ndarray
        Array of low prices
    close : np.ndarray
        Array of close prices
    order : int, optional
        Swing detection order (default: 5)
    tolerance : float, optional
        Price tolerance for trough matching (default: 0.02)
    
    Returns
    -------
    list
        List of DoublePattern objects
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    patterns = []
    
    swing_high_indices = get_swing_high_indices(high, order)
    swing_low_indices = get_swing_low_indices(low, order)
    
    if len(swing_low_indices) < 2:
        return patterns
    
    for i in range(len(swing_low_indices) - 1):
        first_idx = swing_low_indices[i]
        second_idx = swing_low_indices[i + 1]
        
        first_price = low[first_idx]
        second_price = low[second_idx]
        
        # Troughs should be roughly equal
        if not price_within_tolerance(first_price, second_price, tolerance):
            continue
        
        # Find the middle high point
        middle_highs = [idx for idx in swing_high_indices 
                        if first_idx < idx < second_idx]
        
        if not middle_highs:
            continue
        
        # Use the highest point as the middle
        middle_idx = max(middle_highs, key=lambda x: high[x])
        middle_price = high[middle_idx]
        neckline = middle_price
        
        # Calculate confidence based on trough similarity
        trough_diff = abs(first_price - second_price) / first_price if first_price != 0 else 0
        confidence = max(0.0, min(1.0, 1 - trough_diff * 10))
        
        pattern = DoublePattern(
            pattern_type='double_bottom',
            first_peak=(first_idx, first_price),
            second_peak=(second_idx, second_price),
            middle_point=(middle_idx, middle_price),
            neckline=neckline,
            breakout_level=neckline,
            start_index=first_idx,
            end_index=second_idx,
            confidence=confidence
        )
        patterns.append(pattern)
    
    return patterns


def detect_triple_top(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    order: int = 5,
    tolerance: float = 0.02
) -> List[TriplePattern]:
    """
    Detect Triple Top patterns (bearish reversal).
    
    Parameters
    ----------
    high : np.ndarray
        Array of high prices
    low : np.ndarray
        Array of low prices
    close : np.ndarray
        Array of close prices
    order : int, optional
        Swing detection order (default: 5)
    tolerance : float, optional
        Price tolerance for peak matching (default: 0.02)
    
    Returns
    -------
    list
        List of TriplePattern objects
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    patterns = []
    
    swing_high_indices = get_swing_high_indices(high, order)
    swing_low_indices = get_swing_low_indices(low, order)
    
    if len(swing_high_indices) < 3:
        return patterns
    
    for i in range(len(swing_high_indices) - 2):
        first_idx = swing_high_indices[i]
        second_idx = swing_high_indices[i + 1]
        third_idx = swing_high_indices[i + 2]
        
        first_price = high[first_idx]
        second_price = high[second_idx]
        third_price = high[third_idx]
        
        # All three peaks should be roughly equal
        avg_price = (first_price + second_price + third_price) / 3
        if not (price_within_tolerance(first_price, avg_price, tolerance) and
                price_within_tolerance(second_price, avg_price, tolerance) and
                price_within_tolerance(third_price, avg_price, tolerance)):
            continue
        
        # Find the lowest low between first and third peak for support
        lows_between = [low[idx] for idx in swing_low_indices 
                        if first_idx < idx < third_idx]
        
        if not lows_between:
            continue
        
        support_level = min(lows_between)
        
        # Calculate confidence
        price_variance = np.std([first_price, second_price, third_price]) / avg_price
        confidence = max(0.0, min(1.0, 1 - price_variance * 10))
        
        pattern = TriplePattern(
            pattern_type='triple_top',
            first_peak=(first_idx, first_price),
            second_peak=(second_idx, second_price),
            third_peak=(third_idx, third_price),
            support_resistance=avg_price,
            breakout_level=support_level,
            start_index=first_idx,
            end_index=third_idx,
            confidence=confidence
        )
        patterns.append(pattern)
    
    return patterns


def detect_triple_bottom(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    order: int = 5,
    tolerance: float = 0.02
) -> List[TriplePattern]:
    """
    Detect Triple Bottom patterns (bullish reversal).
    
    Parameters
    ----------
    high : np.ndarray
        Array of high prices
    low : np.ndarray
        Array of low prices
    close : np.ndarray
        Array of close prices
    order : int, optional
        Swing detection order (default: 5)
    tolerance : float, optional
        Price tolerance for trough matching (default: 0.02)
    
    Returns
    -------
    list
        List of TriplePattern objects
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    patterns = []
    
    swing_high_indices = get_swing_high_indices(high, order)
    swing_low_indices = get_swing_low_indices(low, order)
    
    if len(swing_low_indices) < 3:
        return patterns
    
    for i in range(len(swing_low_indices) - 2):
        first_idx = swing_low_indices[i]
        second_idx = swing_low_indices[i + 1]
        third_idx = swing_low_indices[i + 2]
        
        first_price = low[first_idx]
        second_price = low[second_idx]
        third_price = low[third_idx]
        
        # All three troughs should be roughly equal
        avg_price = (first_price + second_price + third_price) / 3
        if not (price_within_tolerance(first_price, avg_price, tolerance) and
                price_within_tolerance(second_price, avg_price, tolerance) and
                price_within_tolerance(third_price, avg_price, tolerance)):
            continue
        
        # Find the highest high between first and third trough for resistance
        highs_between = [high[idx] for idx in swing_high_indices 
                         if first_idx < idx < third_idx]
        
        if not highs_between:
            continue
        
        resistance_level = max(highs_between)
        
        # Calculate confidence
        price_variance = np.std([first_price, second_price, third_price]) / avg_price if avg_price != 0 else 0
        confidence = max(0.0, min(1.0, 1 - price_variance * 10))
        
        pattern = TriplePattern(
            pattern_type='triple_bottom',
            first_peak=(first_idx, first_price),
            second_peak=(second_idx, second_price),
            third_peak=(third_idx, third_price),
            support_resistance=avg_price,
            breakout_level=resistance_level,
            start_index=first_idx,
            end_index=third_idx,
            confidence=confidence
        )
        patterns.append(pattern)
    
    return patterns


def detect_triangle(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    order: int = 5,
    min_points: int = 2
) -> List[TrianglePattern]:
    """
    Detect Triangle patterns (ascending, descending, symmetrical).
    
    Parameters
    ----------
    high : np.ndarray
        Array of high prices
    low : np.ndarray
        Array of low prices
    close : np.ndarray
        Array of close prices
    order : int, optional
        Swing detection order (default: 5)
    min_points : int, optional
        Minimum points for each trendline (default: 2)
    
    Returns
    -------
    list
        List of TrianglePattern objects
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    patterns = []
    
    swing_high_indices = get_swing_high_indices(high, order)
    swing_low_indices = get_swing_low_indices(low, order)
    
    if len(swing_high_indices) < min_points or len(swing_low_indices) < min_points:
        return patterns
    
    # Create points for trendlines
    upper_points = [(int(idx), float(high[idx])) for idx in swing_high_indices]
    lower_points = [(int(idx), float(low[idx])) for idx in swing_low_indices]
    
    if len(upper_points) < min_points or len(lower_points) < min_points:
        return patterns
    
    try:
        upper_slope, upper_intercept = fit_trendline(upper_points[-min_points:])
        lower_slope, lower_intercept = fit_trendline(lower_points[-min_points:])
    except ValueError:
        return patterns
    
    # Determine triangle type based on slopes
    slope_threshold = 0.001
    
    if abs(upper_slope) < slope_threshold and lower_slope > slope_threshold:
        # Ascending triangle: flat top, rising bottom
        pattern_type = 'ascending'
    elif upper_slope < -slope_threshold and abs(lower_slope) < slope_threshold:
        # Descending triangle: falling top, flat bottom
        pattern_type = 'descending'
    elif upper_slope < -slope_threshold and lower_slope > slope_threshold:
        # Symmetrical triangle: converging trendlines
        pattern_type = 'symmetrical'
    else:
        return patterns
    
    # Calculate apex (where trendlines meet)
    if abs(upper_slope - lower_slope) > 0.0001:
        apex_index = int((lower_intercept - upper_intercept) / (upper_slope - lower_slope))
    else:
        apex_index = swing_high_indices[-1] + 50  # Default if parallel
    
    start_index = min(swing_high_indices[0], swing_low_indices[0])
    end_index = max(swing_high_indices[-1], swing_low_indices[-1])
    
    # Breakout level at current position
    breakout_level = upper_slope * end_index + upper_intercept
    
    # Calculate confidence based on trendline fit
    confidence = 0.6  # Base confidence
    
    pattern = TrianglePattern(
        pattern_type=pattern_type,
        upper_trendline=(upper_slope, upper_intercept),
        lower_trendline=(lower_slope, lower_intercept),
        upper_points=upper_points,
        lower_points=lower_points,
        apex_index=apex_index,
        breakout_level=breakout_level,
        start_index=start_index,
        end_index=end_index,
        confidence=confidence
    )
    patterns.append(pattern)
    
    return patterns


def detect_wedge(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    order: int = 5,
    min_points: int = 2
) -> List[WedgePattern]:
    """
    Detect Wedge patterns (rising, falling).
    
    Parameters
    ----------
    high : np.ndarray
        Array of high prices
    low : np.ndarray
        Array of low prices
    close : np.ndarray
        Array of close prices
    order : int, optional
        Swing detection order (default: 5)
    min_points : int, optional
        Minimum points for each trendline (default: 2)
    
    Returns
    -------
    list
        List of WedgePattern objects
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    patterns = []
    
    swing_high_indices = get_swing_high_indices(high, order)
    swing_low_indices = get_swing_low_indices(low, order)
    
    if len(swing_high_indices) < min_points or len(swing_low_indices) < min_points:
        return patterns
    
    # Create points for trendlines
    upper_points = [(int(idx), float(high[idx])) for idx in swing_high_indices]
    lower_points = [(int(idx), float(low[idx])) for idx in swing_low_indices]
    
    try:
        upper_slope, upper_intercept = fit_trendline(upper_points[-min_points:])
        lower_slope, lower_intercept = fit_trendline(lower_points[-min_points:])
    except ValueError:
        return patterns
    
    slope_threshold = 0.001
    
    # Rising wedge: both trendlines sloping up, converging
    if upper_slope > slope_threshold and lower_slope > slope_threshold:
        if upper_slope < lower_slope:  # Converging
            pattern_type = 'rising'
        else:
            return patterns
    # Falling wedge: both trendlines sloping down, converging
    elif upper_slope < -slope_threshold and lower_slope < -slope_threshold:
        if upper_slope > lower_slope:  # Converging
            pattern_type = 'falling'
        else:
            return patterns
    else:
        return patterns
    
    start_index = min(swing_high_indices[0], swing_low_indices[0])
    end_index = max(swing_high_indices[-1], swing_low_indices[-1])
    
    # Breakout level at current position
    if pattern_type == 'rising':
        breakout_level = lower_slope * end_index + lower_intercept
    else:
        breakout_level = upper_slope * end_index + upper_intercept
    
    pattern = WedgePattern(
        pattern_type=pattern_type,
        upper_trendline=(upper_slope, upper_intercept),
        lower_trendline=(lower_slope, lower_intercept),
        upper_points=upper_points,
        lower_points=lower_points,
        breakout_level=breakout_level,
        start_index=start_index,
        end_index=end_index,
        confidence=0.6
    )
    patterns.append(pattern)
    
    return patterns


def detect_flag(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    order: int = 5,
    min_pole_bars: int = 5,
    max_flag_bars: int = 20
) -> List[FlagPattern]:
    """
    Detect Flag and Pennant patterns.
    
    Parameters
    ----------
    high : np.ndarray
        Array of high prices
    low : np.ndarray
        Array of low prices
    close : np.ndarray
        Array of close prices
    order : int, optional
        Swing detection order (default: 5)
    min_pole_bars : int, optional
        Minimum bars for the flagpole (default: 5)
    max_flag_bars : int, optional
        Maximum bars for the flag consolidation (default: 20)
    
    Returns
    -------
    list
        List of FlagPattern objects
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    patterns = []
    
    n = len(close)
    if n < min_pole_bars + order:
        return patterns
    
    swing_high_indices = get_swing_high_indices(high, order)
    swing_low_indices = get_swing_low_indices(low, order)
    
    # Look for bull flags (strong up move followed by consolidation)
    for i in range(len(swing_high_indices)):
        pole_end_idx = swing_high_indices[i]
        
        if pole_end_idx < min_pole_bars:
            continue
        
        # Find pole start (significant low before the high)
        pole_start_candidates = [idx for idx in swing_low_indices 
                                  if pole_end_idx - idx >= min_pole_bars and idx < pole_end_idx]
        
        if not pole_start_candidates:
            continue
        
        pole_start_idx = max(pole_start_candidates, key=lambda x: high[pole_end_idx] - low[x])
        pole_start_price = low[pole_start_idx]
        pole_end_price = high[pole_end_idx]
        
        # Check if pole is strong enough (at least 5% move)
        if (pole_end_price - pole_start_price) / pole_start_price < 0.05:
            continue
        
        # Look for flag consolidation
        flag_end_idx = min(pole_end_idx + max_flag_bars, n - 1)
        flag_highs = [(j, high[j]) for j in range(pole_end_idx, flag_end_idx)]
        flag_lows = [(j, low[j]) for j in range(pole_end_idx, flag_end_idx)]
        
        if len(flag_highs) < 2:
            continue
        
        try:
            upper_slope, upper_intercept = fit_trendline(flag_highs)
            lower_slope, lower_intercept = fit_trendline(flag_lows)
        except ValueError:
            continue
        
        # Bull flag: slight downward or sideways consolidation
        if upper_slope > 0.001 or lower_slope > 0.001:
            continue
        
        # Determine pattern type
        if abs(upper_slope - lower_slope) < 0.001:
            pattern_type = 'bull_flag'
        else:
            pattern_type = 'pennant'
        
        pole_length = pole_end_price - pole_start_price
        breakout_level = pole_end_price
        target = breakout_level + pole_length
        
        pattern = FlagPattern(
            pattern_type=pattern_type,
            pole_start=(pole_start_idx, pole_start_price),
            pole_end=(pole_end_idx, pole_end_price),
            flag_upper=(upper_slope, upper_intercept),
            flag_lower=(lower_slope, lower_intercept),
            breakout_level=breakout_level,
            target=target,
            start_index=pole_start_idx,
            end_index=flag_end_idx,
            confidence=0.6
        )
        patterns.append(pattern)
    
    # Look for bear flags (strong down move followed by consolidation)
    for i in range(len(swing_low_indices)):
        pole_end_idx = swing_low_indices[i]
        
        if pole_end_idx < min_pole_bars:
            continue
        
        # Find pole start (significant high before the low)
        pole_start_candidates = [idx for idx in swing_high_indices 
                                  if pole_end_idx - idx >= min_pole_bars and idx < pole_end_idx]
        
        if not pole_start_candidates:
            continue
        
        pole_start_idx = max(pole_start_candidates, key=lambda x: high[x] - low[pole_end_idx])
        pole_start_price = high[pole_start_idx]
        pole_end_price = low[pole_end_idx]
        
        # Check if pole is strong enough
        if (pole_start_price - pole_end_price) / pole_start_price < 0.05:
            continue
        
        # Look for flag consolidation
        flag_end_idx = min(pole_end_idx + max_flag_bars, n - 1)
        flag_highs = [(j, high[j]) for j in range(pole_end_idx, flag_end_idx)]
        flag_lows = [(j, low[j]) for j in range(pole_end_idx, flag_end_idx)]
        
        if len(flag_highs) < 2:
            continue
        
        try:
            upper_slope, upper_intercept = fit_trendline(flag_highs)
            lower_slope, lower_intercept = fit_trendline(flag_lows)
        except ValueError:
            continue
        
        # Bear flag: slight upward or sideways consolidation
        if upper_slope < -0.001 or lower_slope < -0.001:
            continue
        
        pole_length = pole_start_price - pole_end_price
        breakout_level = pole_end_price
        target = breakout_level - pole_length
        
        pattern = FlagPattern(
            pattern_type='bear_flag',
            pole_start=(pole_start_idx, pole_start_price),
            pole_end=(pole_end_idx, pole_end_price),
            flag_upper=(upper_slope, upper_intercept),
            flag_lower=(lower_slope, lower_intercept),
            breakout_level=breakout_level,
            target=target,
            start_index=pole_start_idx,
            end_index=flag_end_idx,
            confidence=0.6
        )
        patterns.append(pattern)
    
    return patterns


def detect_vcp(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    order: int = 5,
    min_contractions: int = 2,
    max_contractions: int = 5
) -> List[VCPPattern]:
    """
    Detect Volatility Contraction Pattern (VCP).
    
    A VCP consists of a series of price contractions (each smaller than the previous)
    as the stock consolidates before a potential breakout.
    
    Parameters
    ----------
    high : np.ndarray
        Array of high prices
    low : np.ndarray
        Array of low prices
    close : np.ndarray
        Array of close prices
    order : int, optional
        Swing detection order (default: 5)
    min_contractions : int, optional
        Minimum number of contractions required (default: 2)
    max_contractions : int, optional
        Maximum contractions to look for (default: 5)
    
    Returns
    -------
    list
        List of VCPPattern objects
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    patterns = []
    
    swing_high_indices = get_swing_high_indices(high, order)
    swing_low_indices = get_swing_low_indices(low, order)
    
    if len(swing_high_indices) < min_contractions or len(swing_low_indices) < min_contractions:
        return patterns
    
    # Look for decreasing volatility contractions
    contractions = []
    depth_percentages = []
    
    # Start from the first significant high
    if len(swing_high_indices) == 0:
        return patterns
    
    first_high_idx = swing_high_indices[0]
    first_high_price = high[first_high_idx]
    
    # Look for subsequent swing lows and highs with decreasing range
    prev_range = float('inf')
    
    for i in range(min(len(swing_high_indices), max_contractions)):
        high_idx = swing_high_indices[i]
        high_price = high[high_idx]
        
        # Find corresponding low
        corresponding_lows = [idx for idx in swing_low_indices 
                               if high_idx - order * 2 < idx < high_idx + order * 2]
        
        if not corresponding_lows:
            continue
        
        low_idx = min(corresponding_lows, key=lambda x: low[x])
        low_price = low[low_idx]
        
        current_range = high_price - low_price
        depth_pct = (first_high_price - low_price) / first_high_price * 100 if first_high_price > 0 else 0
        
        if current_range < prev_range:
            contractions.append((high_idx, low_idx, current_range))
            depth_percentages.append(depth_pct)
            prev_range = current_range
        else:
            break
    
    if len(contractions) >= min_contractions:
        # Determine pivot point (the last swing high)
        last_high_idx = contractions[-1][0]
        pivot_price = high[last_high_idx]
        
        pattern = VCPPattern(
            contractions=contractions,
            pivot_point=(last_high_idx, pivot_price),
            breakout_level=pivot_price,
            depth_percentages=depth_percentages,
            start_index=contractions[0][0],
            end_index=contractions[-1][0],
            confidence=min(1.0, len(contractions) / max_contractions)
        )
        patterns.append(pattern)
    
    return patterns
