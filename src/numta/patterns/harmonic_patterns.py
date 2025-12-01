"""
Harmonic pattern detection module.

This module provides detection functions for harmonic patterns including:
- Gartley pattern
- Butterfly pattern
- Bat pattern
- Crab pattern

Harmonic patterns are price patterns that use Fibonacci numbers to define
precise turning points. They provide high probability trading setups with
clearly defined risk/reward ratios.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from .swing import get_swing_high_indices, get_swing_low_indices


@dataclass
class HarmonicPattern:
    """
    Harmonic pattern data structure.
    
    Attributes
    ----------
    pattern_type : str
        Type of pattern: 'gartley', 'butterfly', 'bat', 'crab'
    X : tuple
        (index, price) of point X
    A : tuple
        (index, price) of point A
    B : tuple
        (index, price) of point B
    C : tuple
        (index, price) of point C
    D : tuple
        (index, price) of point D (Potential Reversal Zone)
    direction : str
        'bullish' or 'bearish'
    XA_retracement : float
        AB retracement of XA leg
    AB_retracement : float
        BC retracement of AB leg
    BC_retracement : float
        CD retracement of BC leg
    CD_retracement : float
        CD extension/retracement
    prz : tuple
        (lower_price, upper_price) Potential Reversal Zone
    confidence : float
        Confidence score (0-1) based on Fibonacci ratio accuracy
    """
    pattern_type: str
    X: Tuple[int, float]
    A: Tuple[int, float]
    B: Tuple[int, float]
    C: Tuple[int, float]
    D: Tuple[int, float]
    direction: str
    XA_retracement: float
    AB_retracement: float
    BC_retracement: float
    CD_retracement: float
    prz: Tuple[float, float]
    confidence: float


# Standard Fibonacci ratios for harmonic patterns
GARTLEY_RATIOS = {
    'XA_retracement': 0.618,  # B should be at 61.8% of XA
    'AB_retracement': (0.382, 0.886),  # C should be 38.2-88.6% of AB
    'BC_extension': (1.272, 1.618),  # CD should be 127.2-161.8% of BC
    'XA_retracement_D': 0.786,  # D should be at 78.6% of XA
}

BUTTERFLY_RATIOS = {
    'XA_retracement': 0.786,  # B should be at 78.6% of XA
    'AB_retracement': (0.382, 0.886),  # C should be 38.2-88.6% of AB
    'BC_extension': (1.618, 2.618),  # CD should be 161.8-261.8% of BC
    'XA_extension': (1.272, 1.618),  # D should be 127.2-161.8% extension of XA
}

BAT_RATIOS = {
    'XA_retracement': (0.382, 0.50),  # B should be at 38.2-50% of XA
    'AB_retracement': (0.382, 0.886),  # C should be 38.2-88.6% of AB
    'BC_extension': (1.618, 2.618),  # CD should be 161.8-261.8% of BC
    'XA_retracement_D': 0.886,  # D should be at 88.6% of XA
}

CRAB_RATIOS = {
    'XA_retracement': (0.382, 0.618),  # B should be at 38.2-61.8% of XA
    'AB_retracement': (0.382, 0.886),  # C should be 38.2-88.6% of AB
    'BC_extension': (2.618, 3.618),  # CD should be 261.8-361.8% of BC
    'XA_extension': 1.618,  # D should be at 161.8% extension of XA
}


def _calculate_retracement(start: float, end: float, current: float) -> float:
    """Calculate retracement ratio."""
    move = end - start
    if abs(move) < 1e-10:
        return 0.0
    return abs(current - end) / abs(move)


def _is_within_tolerance(value: float, target: float, tolerance: float) -> bool:
    """Check if value is within tolerance of target."""
    return abs(value - target) <= tolerance


def _is_within_range(value: float, range_tuple: Tuple[float, float], tolerance: float) -> bool:
    """Check if value is within a range with tolerance."""
    return (range_tuple[0] - tolerance) <= value <= (range_tuple[1] + tolerance)


def _calculate_confidence(ratios: dict, targets: dict, tolerance: float) -> float:
    """Calculate confidence score based on how closely ratios match targets."""
    scores = []
    
    for key, actual in ratios.items():
        if key not in targets:
            continue
        
        target = targets[key]
        if isinstance(target, tuple):
            # Range target - score based on distance from center of range
            center = (target[0] + target[1]) / 2
            range_half = (target[1] - target[0]) / 2
            if range_half > 0:
                distance = abs(actual - center) / range_half
                score = max(0, 1 - distance)
            else:
                score = 1.0 if actual == center else 0.0
        else:
            # Single target value
            distance = abs(actual - target) / max(abs(target), 0.001)
            score = max(0, 1 - distance / tolerance * 0.5)
        
        scores.append(score)
    
    if not scores:
        return 0.5
    
    return sum(scores) / len(scores)


def _find_zigzag_points(
    high: np.ndarray,
    low: np.ndarray,
    order: int = 5,
    num_points: int = 5
) -> List[Tuple[int, float, str]]:
    """
    Find zigzag points (alternating swing highs and lows).
    
    Returns list of (index, price, type) where type is 'high' or 'low'.
    """
    swing_high_indices = get_swing_high_indices(high, order)
    swing_low_indices = get_swing_low_indices(low, order)
    
    # Combine and sort all swing points
    all_swings = []
    for idx in swing_high_indices:
        all_swings.append((int(idx), float(high[idx]), 'high'))
    for idx in swing_low_indices:
        all_swings.append((int(idx), float(low[idx]), 'low'))
    
    all_swings.sort(key=lambda x: x[0])
    
    # Filter to get alternating highs and lows
    if not all_swings:
        return []
    
    zigzag = [all_swings[0]]
    for swing in all_swings[1:]:
        if swing[2] != zigzag[-1][2]:  # Different type
            zigzag.append(swing)
    
    return zigzag[-num_points:] if len(zigzag) >= num_points else zigzag


def detect_gartley(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    order: int = 5,
    tolerance: float = 0.02
) -> List[HarmonicPattern]:
    """
    Detect Gartley harmonic patterns.
    
    The Gartley pattern is characterized by:
    - B retraces 61.8% of XA
    - C retraces 38.2-88.6% of AB
    - D retraces 78.6% of XA
    - CD is 127.2-161.8% of BC
    
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
        Tolerance for Fibonacci ratio matching (default: 0.02)
    
    Returns
    -------
    list
        List of HarmonicPattern objects
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    patterns = []
    
    zigzag = _find_zigzag_points(high, low, order, num_points=5)
    
    if len(zigzag) < 5:
        return patterns
    
    # Try to find Gartley patterns in the zigzag points
    for i in range(len(zigzag) - 4):
        X = zigzag[i]
        A = zigzag[i + 1]
        B = zigzag[i + 2]
        C = zigzag[i + 3]
        D = zigzag[i + 4]
        
        # Determine direction
        if X[2] == 'low' and A[2] == 'high':
            direction = 'bullish'
        elif X[2] == 'high' and A[2] == 'low':
            direction = 'bearish'
        else:
            continue
        
        # Calculate retracements
        XA_move = A[1] - X[1]
        AB_move = B[1] - A[1]
        BC_move = C[1] - B[1]
        
        if abs(XA_move) < 1e-10 or abs(AB_move) < 1e-10 or abs(BC_move) < 1e-10:
            continue
        
        # B retracement of XA
        XA_ret = abs(AB_move) / abs(XA_move)
        
        # C retracement of AB
        AB_ret = abs(BC_move) / abs(AB_move)
        
        # D retracement of XA
        XA_ret_D = abs(D[1] - A[1]) / abs(XA_move) if direction == 'bullish' else abs(A[1] - D[1]) / abs(XA_move)
        
        # CD extension of BC
        CD_move = D[1] - C[1]
        BC_ext = abs(CD_move) / abs(BC_move)
        
        # Check Gartley ratios
        if not _is_within_tolerance(XA_ret, GARTLEY_RATIOS['XA_retracement'], tolerance):
            continue
        
        if not _is_within_range(AB_ret, GARTLEY_RATIOS['AB_retracement'], tolerance):
            continue
        
        if not _is_within_tolerance(XA_ret_D, GARTLEY_RATIOS['XA_retracement_D'], tolerance):
            continue
        
        if not _is_within_range(BC_ext, GARTLEY_RATIOS['BC_extension'], tolerance):
            continue
        
        # Calculate PRZ (Potential Reversal Zone)
        prz_center = D[1]
        prz_range = abs(XA_move) * 0.02
        prz = (prz_center - prz_range, prz_center + prz_range)
        
        # Calculate confidence
        ratios = {
            'XA_retracement': XA_ret,
            'AB_retracement': AB_ret,
            'XA_retracement_D': XA_ret_D,
            'BC_extension': BC_ext
        }
        confidence = _calculate_confidence(ratios, GARTLEY_RATIOS, tolerance)
        
        pattern = HarmonicPattern(
            pattern_type='gartley',
            X=(X[0], X[1]),
            A=(A[0], A[1]),
            B=(B[0], B[1]),
            C=(C[0], C[1]),
            D=(D[0], D[1]),
            direction=direction,
            XA_retracement=XA_ret,
            AB_retracement=AB_ret,
            BC_retracement=BC_ext,  # CD retracement of BC
            CD_retracement=XA_ret_D,
            prz=prz,
            confidence=confidence
        )
        patterns.append(pattern)
    
    return patterns


def detect_butterfly(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    order: int = 5,
    tolerance: float = 0.02
) -> List[HarmonicPattern]:
    """
    Detect Butterfly harmonic patterns.
    
    The Butterfly pattern is characterized by:
    - B retraces 78.6% of XA
    - C retraces 38.2-88.6% of AB
    - D extends 127.2-161.8% beyond X
    - CD is 161.8-261.8% of BC
    
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
        Tolerance for Fibonacci ratio matching (default: 0.02)
    
    Returns
    -------
    list
        List of HarmonicPattern objects
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    patterns = []
    
    zigzag = _find_zigzag_points(high, low, order, num_points=5)
    
    if len(zigzag) < 5:
        return patterns
    
    for i in range(len(zigzag) - 4):
        X = zigzag[i]
        A = zigzag[i + 1]
        B = zigzag[i + 2]
        C = zigzag[i + 3]
        D = zigzag[i + 4]
        
        # Determine direction
        if X[2] == 'low' and A[2] == 'high':
            direction = 'bullish'
        elif X[2] == 'high' and A[2] == 'low':
            direction = 'bearish'
        else:
            continue
        
        # Calculate retracements
        XA_move = A[1] - X[1]
        AB_move = B[1] - A[1]
        BC_move = C[1] - B[1]
        
        if abs(XA_move) < 1e-10 or abs(AB_move) < 1e-10 or abs(BC_move) < 1e-10:
            continue
        
        XA_ret = abs(AB_move) / abs(XA_move)
        AB_ret = abs(BC_move) / abs(AB_move)
        CD_move = D[1] - C[1]
        BC_ext = abs(CD_move) / abs(BC_move)
        
        # D extension beyond X
        XA_ext = abs(D[1] - X[1]) / abs(XA_move)
        
        # Check Butterfly ratios
        if not _is_within_tolerance(XA_ret, BUTTERFLY_RATIOS['XA_retracement'], tolerance):
            continue
        
        if not _is_within_range(AB_ret, BUTTERFLY_RATIOS['AB_retracement'], tolerance):
            continue
        
        if not _is_within_range(BC_ext, BUTTERFLY_RATIOS['BC_extension'], tolerance):
            continue
        
        if not _is_within_range(XA_ext, BUTTERFLY_RATIOS['XA_extension'], tolerance):
            continue
        
        prz_center = D[1]
        prz_range = abs(XA_move) * 0.02
        prz = (prz_center - prz_range, prz_center + prz_range)
        
        ratios = {
            'XA_retracement': XA_ret,
            'AB_retracement': AB_ret,
            'BC_extension': BC_ext,
            'XA_extension': XA_ext
        }
        confidence = _calculate_confidence(ratios, BUTTERFLY_RATIOS, tolerance)
        
        pattern = HarmonicPattern(
            pattern_type='butterfly',
            X=(X[0], X[1]),
            A=(A[0], A[1]),
            B=(B[0], B[1]),
            C=(C[0], C[1]),
            D=(D[0], D[1]),
            direction=direction,
            XA_retracement=XA_ret,
            AB_retracement=AB_ret,
            BC_retracement=BC_ext,
            CD_retracement=XA_ext,
            prz=prz,
            confidence=confidence
        )
        patterns.append(pattern)
    
    return patterns


def detect_bat(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    order: int = 5,
    tolerance: float = 0.02
) -> List[HarmonicPattern]:
    """
    Detect Bat harmonic patterns.
    
    The Bat pattern is characterized by:
    - B retraces 38.2-50% of XA
    - C retraces 38.2-88.6% of AB
    - D retraces 88.6% of XA
    - CD is 161.8-261.8% of BC
    
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
        Tolerance for Fibonacci ratio matching (default: 0.02)
    
    Returns
    -------
    list
        List of HarmonicPattern objects
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    patterns = []
    
    zigzag = _find_zigzag_points(high, low, order, num_points=5)
    
    if len(zigzag) < 5:
        return patterns
    
    for i in range(len(zigzag) - 4):
        X = zigzag[i]
        A = zigzag[i + 1]
        B = zigzag[i + 2]
        C = zigzag[i + 3]
        D = zigzag[i + 4]
        
        if X[2] == 'low' and A[2] == 'high':
            direction = 'bullish'
        elif X[2] == 'high' and A[2] == 'low':
            direction = 'bearish'
        else:
            continue
        
        XA_move = A[1] - X[1]
        AB_move = B[1] - A[1]
        BC_move = C[1] - B[1]
        
        if abs(XA_move) < 1e-10 or abs(AB_move) < 1e-10 or abs(BC_move) < 1e-10:
            continue
        
        XA_ret = abs(AB_move) / abs(XA_move)
        AB_ret = abs(BC_move) / abs(AB_move)
        CD_move = D[1] - C[1]
        BC_ext = abs(CD_move) / abs(BC_move)
        XA_ret_D = abs(D[1] - A[1]) / abs(XA_move) if direction == 'bullish' else abs(A[1] - D[1]) / abs(XA_move)
        
        if not _is_within_range(XA_ret, BAT_RATIOS['XA_retracement'], tolerance):
            continue
        
        if not _is_within_range(AB_ret, BAT_RATIOS['AB_retracement'], tolerance):
            continue
        
        if not _is_within_tolerance(XA_ret_D, BAT_RATIOS['XA_retracement_D'], tolerance):
            continue
        
        if not _is_within_range(BC_ext, BAT_RATIOS['BC_extension'], tolerance):
            continue
        
        prz_center = D[1]
        prz_range = abs(XA_move) * 0.02
        prz = (prz_center - prz_range, prz_center + prz_range)
        
        ratios = {
            'XA_retracement': XA_ret,
            'AB_retracement': AB_ret,
            'XA_retracement_D': XA_ret_D,
            'BC_extension': BC_ext
        }
        confidence = _calculate_confidence(ratios, BAT_RATIOS, tolerance)
        
        pattern = HarmonicPattern(
            pattern_type='bat',
            X=(X[0], X[1]),
            A=(A[0], A[1]),
            B=(B[0], B[1]),
            C=(C[0], C[1]),
            D=(D[0], D[1]),
            direction=direction,
            XA_retracement=XA_ret,
            AB_retracement=AB_ret,
            BC_retracement=BC_ext,
            CD_retracement=XA_ret_D,
            prz=prz,
            confidence=confidence
        )
        patterns.append(pattern)
    
    return patterns


def detect_crab(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    order: int = 5,
    tolerance: float = 0.02
) -> List[HarmonicPattern]:
    """
    Detect Crab harmonic patterns.
    
    The Crab pattern is characterized by:
    - B retraces 38.2-61.8% of XA
    - C retraces 38.2-88.6% of AB
    - D extends 161.8% beyond X
    - CD is 261.8-361.8% of BC
    
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
        Tolerance for Fibonacci ratio matching (default: 0.02)
    
    Returns
    -------
    list
        List of HarmonicPattern objects
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    patterns = []
    
    zigzag = _find_zigzag_points(high, low, order, num_points=5)
    
    if len(zigzag) < 5:
        return patterns
    
    for i in range(len(zigzag) - 4):
        X = zigzag[i]
        A = zigzag[i + 1]
        B = zigzag[i + 2]
        C = zigzag[i + 3]
        D = zigzag[i + 4]
        
        if X[2] == 'low' and A[2] == 'high':
            direction = 'bullish'
        elif X[2] == 'high' and A[2] == 'low':
            direction = 'bearish'
        else:
            continue
        
        XA_move = A[1] - X[1]
        AB_move = B[1] - A[1]
        BC_move = C[1] - B[1]
        
        if abs(XA_move) < 1e-10 or abs(AB_move) < 1e-10 or abs(BC_move) < 1e-10:
            continue
        
        XA_ret = abs(AB_move) / abs(XA_move)
        AB_ret = abs(BC_move) / abs(AB_move)
        CD_move = D[1] - C[1]
        BC_ext = abs(CD_move) / abs(BC_move)
        XA_ext = abs(D[1] - X[1]) / abs(XA_move)
        
        if not _is_within_range(XA_ret, CRAB_RATIOS['XA_retracement'], tolerance):
            continue
        
        if not _is_within_range(AB_ret, CRAB_RATIOS['AB_retracement'], tolerance):
            continue
        
        if not _is_within_range(BC_ext, CRAB_RATIOS['BC_extension'], tolerance):
            continue
        
        if not _is_within_tolerance(XA_ext, CRAB_RATIOS['XA_extension'], tolerance):
            continue
        
        prz_center = D[1]
        prz_range = abs(XA_move) * 0.02
        prz = (prz_center - prz_range, prz_center + prz_range)
        
        ratios = {
            'XA_retracement': XA_ret,
            'AB_retracement': AB_ret,
            'BC_extension': BC_ext,
            'XA_extension': XA_ext
        }
        confidence = _calculate_confidence(ratios, CRAB_RATIOS, tolerance)
        
        pattern = HarmonicPattern(
            pattern_type='crab',
            X=(X[0], X[1]),
            A=(A[0], A[1]),
            B=(B[0], B[1]),
            C=(C[0], C[1]),
            D=(D[0], D[1]),
            direction=direction,
            XA_retracement=XA_ret,
            AB_retracement=AB_ret,
            BC_retracement=BC_ext,
            CD_retracement=XA_ext,
            prz=prz,
            confidence=confidence
        )
        patterns.append(pattern)
    
    return patterns


def detect_harmonic_patterns(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    patterns: Optional[List[str]] = None,
    order: int = 5,
    tolerance: float = 0.02
) -> List[HarmonicPattern]:
    """
    Detect multiple types of harmonic patterns.
    
    Parameters
    ----------
    high : np.ndarray
        Array of high prices
    low : np.ndarray
        Array of low prices
    close : np.ndarray
        Array of close prices
    patterns : list, optional
        List of pattern types to detect. Options: 'gartley', 'butterfly', 'bat', 'crab'.
        If None, all patterns are detected.
    order : int, optional
        Swing detection order (default: 5)
    tolerance : float, optional
        Tolerance for Fibonacci ratio matching (default: 0.02)
    
    Returns
    -------
    list
        List of HarmonicPattern objects
    """
    if patterns is None:
        patterns = ['gartley', 'butterfly', 'bat', 'crab']
    
    all_patterns = []
    
    pattern_detectors = {
        'gartley': detect_gartley,
        'butterfly': detect_butterfly,
        'bat': detect_bat,
        'crab': detect_crab
    }
    
    for pattern_type in patterns:
        if pattern_type in pattern_detectors:
            detected = pattern_detectors[pattern_type](high, low, close, order, tolerance)
            all_patterns.extend(detected)
    
    # Sort by confidence
    all_patterns.sort(key=lambda x: x.confidence, reverse=True)
    
    return all_patterns
