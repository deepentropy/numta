"""
Swing point detection module.

This module provides functions for detecting swing highs and lows in price data.
Swing points are local extrema that are commonly used as the foundation for
identifying chart patterns and key support/resistance levels.
"""

from typing import Tuple
import numpy as np


def find_swing_highs(high: np.ndarray, order: int = 5) -> np.ndarray:
    """
    Find swing high points in price data.
    
    A swing high is a local maximum where the high at index i is greater than
    the highs at indices [i-order, ..., i-1, i+1, ..., i+order].
    
    Parameters
    ----------
    high : np.ndarray
        Array of high prices
    order : int, optional
        Number of bars on each side to compare for swing detection (default: 5)
        A higher order means fewer but more significant swing points.
    
    Returns
    -------
    np.ndarray
        Boolean array where True indicates a swing high at that index
    
    Examples
    --------
    >>> import numpy as np
    >>> high = np.array([10, 12, 15, 14, 11, 13, 16, 14, 12])
    >>> swing_highs = find_swing_highs(high, order=2)
    >>> np.where(swing_highs)[0]  # Indices of swing highs
    array([2, 6])
    """
    high = np.asarray(high, dtype=np.float64)
    n = len(high)
    swing_highs = np.zeros(n, dtype=bool)
    
    if n < 2 * order + 1:
        return swing_highs
    
    for i in range(order, n - order):
        is_swing_high = True
        for j in range(1, order + 1):
            if high[i] <= high[i - j] or high[i] <= high[i + j]:
                is_swing_high = False
                break
        if is_swing_high:
            swing_highs[i] = True
    
    return swing_highs


def find_swing_lows(low: np.ndarray, order: int = 5) -> np.ndarray:
    """
    Find swing low points in price data.
    
    A swing low is a local minimum where the low at index i is less than
    the lows at indices [i-order, ..., i-1, i+1, ..., i+order].
    
    Parameters
    ----------
    low : np.ndarray
        Array of low prices
    order : int, optional
        Number of bars on each side to compare for swing detection (default: 5)
        A higher order means fewer but more significant swing points.
    
    Returns
    -------
    np.ndarray
        Boolean array where True indicates a swing low at that index
    
    Examples
    --------
    >>> import numpy as np
    >>> low = np.array([12, 10, 8, 9, 11, 9, 7, 8, 10])
    >>> swing_lows = find_swing_lows(low, order=2)
    >>> np.where(swing_lows)[0]  # Indices of swing lows
    array([2, 6])
    """
    low = np.asarray(low, dtype=np.float64)
    n = len(low)
    swing_lows = np.zeros(n, dtype=bool)
    
    if n < 2 * order + 1:
        return swing_lows
    
    for i in range(order, n - order):
        is_swing_low = True
        for j in range(1, order + 1):
            if low[i] >= low[i - j] or low[i] >= low[i + j]:
                is_swing_low = False
                break
        if is_swing_low:
            swing_lows[i] = True
    
    return swing_lows


def find_swing_points(
    high: np.ndarray,
    low: np.ndarray,
    order: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find both swing high and swing low points in price data.
    
    Parameters
    ----------
    high : np.ndarray
        Array of high prices
    low : np.ndarray
        Array of low prices
    order : int, optional
        Number of bars on each side to compare for swing detection (default: 5)
    
    Returns
    -------
    tuple
        (swing_highs, swing_lows) - Two boolean arrays
        
    Examples
    --------
    >>> import numpy as np
    >>> high = np.array([11, 13, 16, 15, 12, 14, 17, 15, 13])
    >>> low = np.array([10, 12, 15, 14, 11, 13, 16, 14, 12])
    >>> swing_highs, swing_lows = find_swing_points(high, low, order=2)
    """
    swing_highs = find_swing_highs(high, order)
    swing_lows = find_swing_lows(low, order)
    return swing_highs, swing_lows


def get_swing_high_indices(high: np.ndarray, order: int = 5) -> np.ndarray:
    """
    Get indices of swing high points.
    
    Parameters
    ----------
    high : np.ndarray
        Array of high prices
    order : int, optional
        Number of bars on each side to compare for swing detection (default: 5)
    
    Returns
    -------
    np.ndarray
        Array of indices where swing highs occur
    """
    swing_highs = find_swing_highs(high, order)
    return np.where(swing_highs)[0]


def get_swing_low_indices(low: np.ndarray, order: int = 5) -> np.ndarray:
    """
    Get indices of swing low points.
    
    Parameters
    ----------
    low : np.ndarray
        Array of low prices
    order : int, optional
        Number of bars on each side to compare for swing detection (default: 5)
    
    Returns
    -------
    np.ndarray
        Array of indices where swing lows occur
    """
    swing_lows = find_swing_lows(low, order)
    return np.where(swing_lows)[0]


def get_swing_points_with_prices(
    high: np.ndarray,
    low: np.ndarray,
    order: int = 5
) -> Tuple[list, list]:
    """
    Get swing points as lists of (index, price) tuples.
    
    Parameters
    ----------
    high : np.ndarray
        Array of high prices
    low : np.ndarray
        Array of low prices
    order : int, optional
        Number of bars on each side to compare for swing detection (default: 5)
    
    Returns
    -------
    tuple
        (swing_highs_list, swing_lows_list) where each list contains
        tuples of (index, price)
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    
    swing_high_indices = get_swing_high_indices(high, order)
    swing_low_indices = get_swing_low_indices(low, order)
    
    swing_highs_list = [(int(i), float(high[i])) for i in swing_high_indices]
    swing_lows_list = [(int(i), float(low[i])) for i in swing_low_indices]
    
    return swing_highs_list, swing_lows_list


def find_higher_highs(high: np.ndarray, order: int = 5) -> np.ndarray:
    """
    Find swing highs that are higher than the previous swing high.
    
    Parameters
    ----------
    high : np.ndarray
        Array of high prices
    order : int, optional
        Number of bars on each side to compare for swing detection (default: 5)
    
    Returns
    -------
    np.ndarray
        Boolean array where True indicates a higher high
    """
    swing_high_indices = get_swing_high_indices(high, order)
    higher_highs = np.zeros(len(high), dtype=bool)
    
    if len(swing_high_indices) < 2:
        return higher_highs
    
    for i in range(1, len(swing_high_indices)):
        current_idx = swing_high_indices[i]
        prev_idx = swing_high_indices[i - 1]
        if high[current_idx] > high[prev_idx]:
            higher_highs[current_idx] = True
    
    return higher_highs


def find_lower_lows(low: np.ndarray, order: int = 5) -> np.ndarray:
    """
    Find swing lows that are lower than the previous swing low.
    
    Parameters
    ----------
    low : np.ndarray
        Array of low prices
    order : int, optional
        Number of bars on each side to compare for swing detection (default: 5)
    
    Returns
    -------
    np.ndarray
        Boolean array where True indicates a lower low
    """
    swing_low_indices = get_swing_low_indices(low, order)
    lower_lows = np.zeros(len(low), dtype=bool)
    
    if len(swing_low_indices) < 2:
        return lower_lows
    
    for i in range(1, len(swing_low_indices)):
        current_idx = swing_low_indices[i]
        prev_idx = swing_low_indices[i - 1]
        if low[current_idx] < low[prev_idx]:
            lower_lows[current_idx] = True
    
    return lower_lows


def find_higher_lows(low: np.ndarray, order: int = 5) -> np.ndarray:
    """
    Find swing lows that are higher than the previous swing low (uptrend).
    
    Parameters
    ----------
    low : np.ndarray
        Array of low prices
    order : int, optional
        Number of bars on each side to compare for swing detection (default: 5)
    
    Returns
    -------
    np.ndarray
        Boolean array where True indicates a higher low
    """
    swing_low_indices = get_swing_low_indices(low, order)
    higher_lows = np.zeros(len(low), dtype=bool)
    
    if len(swing_low_indices) < 2:
        return higher_lows
    
    for i in range(1, len(swing_low_indices)):
        current_idx = swing_low_indices[i]
        prev_idx = swing_low_indices[i - 1]
        if low[current_idx] > low[prev_idx]:
            higher_lows[current_idx] = True
    
    return higher_lows


def find_lower_highs(high: np.ndarray, order: int = 5) -> np.ndarray:
    """
    Find swing highs that are lower than the previous swing high (downtrend).
    
    Parameters
    ----------
    high : np.ndarray
        Array of high prices
    order : int, optional
        Number of bars on each side to compare for swing detection (default: 5)
    
    Returns
    -------
    np.ndarray
        Boolean array where True indicates a lower high
    """
    swing_high_indices = get_swing_high_indices(high, order)
    lower_highs = np.zeros(len(high), dtype=bool)
    
    if len(swing_high_indices) < 2:
        return lower_highs
    
    for i in range(1, len(swing_high_indices)):
        current_idx = swing_high_indices[i]
        prev_idx = swing_high_indices[i - 1]
        if high[current_idx] < high[prev_idx]:
            lower_highs[current_idx] = True
    
    return lower_highs
