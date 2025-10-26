"""
Cycle Indicators - CPU/Numba implementations

This module contains Numba JIT-compiled implementations for cycle indicators.
"""

import numpy as np
from numba import jit


__all__ = [
    "_ht_dcperiod_numba",
    "_ht_dcphase_numba",
    "_ht_phasor_numba",
    "_ht_sine_numba",
    "_ht_trendline_numba",
    "_ht_trendmode_numba",
]


@jit(nopython=True, cache=True)
def _ht_dcperiod_numba(close: np.ndarray, output: np.ndarray) -> None:
    """
    Numba-compiled HT_DCPERIOD calculation

    Hilbert Transform - Dominant Cycle Period
    Uses Hilbert Transform to determine the dominant cycle period
    """
    n = len(close)

    # Constants
    a = 0.0962
    b = 0.5769

    # Minimum lookback for Hilbert Transform
    lookback = 32

    # Fill lookback period with NaN
    for i in range(lookback):
        output[i] = np.nan

    # Initialize arrays
    smooth = np.zeros(n, dtype=np.float64)
    detrender = np.zeros(n, dtype=np.float64)
    i1 = np.zeros(n, dtype=np.float64)
    q1 = np.zeros(n, dtype=np.float64)
    ji = np.zeros(n, dtype=np.float64)
    jq = np.zeros(n, dtype=np.float64)
    i2 = np.zeros(n, dtype=np.float64)
    q2 = np.zeros(n, dtype=np.float64)
    re = np.zeros(n, dtype=np.float64)
    im = np.zeros(n, dtype=np.float64)
    period = np.zeros(n, dtype=np.float64)
    smooth_period = np.zeros(n, dtype=np.float64)

    for i in range(lookback, n):
        # Smooth price
        if i >= 3:
            smooth[i] = (4.0 * close[i] + 3.0 * close[i-1] + 2.0 * close[i-2] + close[i-3]) / 10.0
        else:
            smooth[i] = close[i]

        # Detrend
        if i >= 6:
            detrender[i] = (0.0962 * smooth[i] + 0.5769 * smooth[i-2] -
                           0.5769 * smooth[i-4] - 0.0962 * smooth[i-6]) * (0.075 * period[i-1] + 0.54)

        # Compute InPhase and Quadrature components
        if i >= 7:
            q1[i] = (0.0962 * detrender[i] + 0.5769 * detrender[i-2] -
                    0.5769 * detrender[i-4] - 0.0962 * detrender[i-6]) * (0.075 * period[i-1] + 0.54)
            i1[i] = detrender[i-3]

        # Advance the phase of I1 and Q1 by 90 degrees
        if i >= 9:
            ji[i] = (0.0962 * i1[i] + 0.5769 * i1[i-2] -
                    0.5769 * i1[i-4] - 0.0962 * i1[i-6]) * (0.075 * period[i-1] + 0.54)
            jq[i] = (0.0962 * q1[i] + 0.5769 * q1[i-2] -
                    0.5769 * q1[i-4] - 0.0962 * q1[i-6]) * (0.075 * period[i-1] + 0.54)

        # Phasor addition for 3-bar averaging
        if i >= 10:
            i2[i] = i1[i] - jq[i]
            q2[i] = q1[i] + ji[i]

            # Smooth I and Q
            i2[i] = 0.2 * i2[i] + 0.8 * i2[i-1]
            q2[i] = 0.2 * q2[i] + 0.8 * q2[i-1]

        # Homodyne Discriminator
        if i >= 11:
            re[i] = i2[i] * i2[i-1] + q2[i] * q2[i-1]
            im[i] = i2[i] * q2[i-1] - q2[i] * i2[i-1]

            re[i] = 0.2 * re[i] + 0.8 * re[i-1]
            im[i] = 0.2 * im[i] + 0.8 * im[i-1]

            # Compute period
            if im[i] != 0.0 and re[i] != 0.0:
                period[i] = 360.0 / (np.arctan(im[i] / re[i]) * 180.0 / np.pi)

            # Constrain period
            if period[i] > 1.5 * period[i-1]:
                period[i] = 1.5 * period[i-1]
            if period[i] < 0.67 * period[i-1]:
                period[i] = 0.67 * period[i-1]
            if period[i] < 6:
                period[i] = 6
            if period[i] > 50:
                period[i] = 50

            # Smooth period
            period[i] = 0.2 * period[i] + 0.8 * period[i-1]
            smooth_period[i] = 0.33 * period[i] + 0.67 * smooth_period[i-1]
        else:
            period[i] = 0.0
            smooth_period[i] = 0.0

        output[i] = smooth_period[i]


@jit(nopython=True, cache=True)
def _ht_dcphase_numba(close: np.ndarray, output: np.ndarray) -> None:
    """
    Numba-compiled HT_DCPHASE calculation

    Hilbert Transform - Dominant Cycle Phase
    """
    n = len(close)
    lookback = 32

    # Fill lookback with NaN
    for i in range(lookback):
        output[i] = np.nan

    # Initialize arrays (similar to DCPERIOD)
    smooth = np.zeros(n, dtype=np.float64)
    detrender = np.zeros(n, dtype=np.float64)
    i1 = np.zeros(n, dtype=np.float64)
    q1 = np.zeros(n, dtype=np.float64)
    ji = np.zeros(n, dtype=np.float64)
    jq = np.zeros(n, dtype=np.float64)
    i2 = np.zeros(n, dtype=np.float64)
    q2 = np.zeros(n, dtype=np.float64)
    period = np.zeros(n, dtype=np.float64)

    for i in range(lookback, n):
        # Smooth price
        if i >= 3:
            smooth[i] = (4.0 * close[i] + 3.0 * close[i-1] + 2.0 * close[i-2] + close[i-3]) / 10.0
        else:
            smooth[i] = close[i]

        # Detrend
        if i >= 6:
            detrender[i] = (0.0962 * smooth[i] + 0.5769 * smooth[i-2] -
                           0.5769 * smooth[i-4] - 0.0962 * smooth[i-6]) * (0.075 * period[i-1] + 0.54)

        # Compute InPhase and Quadrature
        if i >= 7:
            q1[i] = (0.0962 * detrender[i] + 0.5769 * detrender[i-2] -
                    0.5769 * detrender[i-4] - 0.0962 * detrender[i-6]) * (0.075 * period[i-1] + 0.54)
            i1[i] = detrender[i-3]

        # Advance phase by 90 degrees
        if i >= 9:
            ji[i] = (0.0962 * i1[i] + 0.5769 * i1[i-2] -
                    0.5769 * i1[i-4] - 0.0962 * i1[i-6]) * (0.075 * period[i-1] + 0.54)
            jq[i] = (0.0962 * q1[i] + 0.5769 * q1[i-2] -
                    0.5769 * q1[i-4] - 0.0962 * q1[i-6]) * (0.075 * period[i-1] + 0.54)

        # Phasor addition
        if i >= 10:
            i2[i] = i1[i] - jq[i]
            q2[i] = q1[i] + ji[i]

            i2[i] = 0.2 * i2[i] + 0.8 * i2[i-1]
            q2[i] = 0.2 * q2[i] + 0.8 * q2[i-1]

        # Compute DC Phase
        if i >= 11:
            if i2[i] != 0.0:
                dc_phase = np.arctan(q2[i] / i2[i]) * 180.0 / np.pi
            else:
                dc_phase = 0.0

            # Adjust for quadrant
            if i2[i] < 0.0:
                dc_phase += 180.0
            if dc_phase < 0.0:
                dc_phase += 360.0

            output[i] = dc_phase
        else:
            output[i] = 0.0


@jit(nopython=True, cache=True)
def _ht_phasor_numba(close: np.ndarray, inphase: np.ndarray, quadrature: np.ndarray) -> None:
    """
    Numba-compiled HT_PHASOR calculation

    Returns InPhase and Quadrature components
    """
    n = len(close)
    lookback = 32

    # Fill lookback with NaN
    for i in range(lookback):
        inphase[i] = np.nan
        quadrature[i] = np.nan

    # Initialize arrays
    smooth = np.zeros(n, dtype=np.float64)
    detrender = np.zeros(n, dtype=np.float64)
    i1 = np.zeros(n, dtype=np.float64)
    q1 = np.zeros(n, dtype=np.float64)
    ji = np.zeros(n, dtype=np.float64)
    jq = np.zeros(n, dtype=np.float64)
    i2 = np.zeros(n, dtype=np.float64)
    q2 = np.zeros(n, dtype=np.float64)
    period = np.zeros(n, dtype=np.float64)

    for i in range(lookback, n):
        # Smooth price
        if i >= 3:
            smooth[i] = (4.0 * close[i] + 3.0 * close[i-1] + 2.0 * close[i-2] + close[i-3]) / 10.0
        else:
            smooth[i] = close[i]

        # Detrend
        if i >= 6:
            detrender[i] = (0.0962 * smooth[i] + 0.5769 * smooth[i-2] -
                           0.5769 * smooth[i-4] - 0.0962 * smooth[i-6]) * (0.075 * period[i-1] + 0.54)

        # Compute InPhase and Quadrature
        if i >= 7:
            q1[i] = (0.0962 * detrender[i] + 0.5769 * detrender[i-2] -
                    0.5769 * detrender[i-4] - 0.0962 * detrender[i-6]) * (0.075 * period[i-1] + 0.54)
            i1[i] = detrender[i-3]

        # Advance phase
        if i >= 9:
            ji[i] = (0.0962 * i1[i] + 0.5769 * i1[i-2] -
                    0.5769 * i1[i-4] - 0.0962 * i1[i-6]) * (0.075 * period[i-1] + 0.54)
            jq[i] = (0.0962 * q1[i] + 0.5769 * q1[i-2] -
                    0.5769 * q1[i-4] - 0.0962 * q1[i-6]) * (0.075 * period[i-1] + 0.54)

        # Phasor addition
        if i >= 10:
            i2[i] = i1[i] - jq[i]
            q2[i] = q1[i] + ji[i]

            # Smooth
            i2[i] = 0.2 * i2[i] + 0.8 * i2[i-1]
            q2[i] = 0.2 * q2[i] + 0.8 * q2[i-1]

            inphase[i] = i2[i]
            quadrature[i] = q2[i]
        else:
            inphase[i] = 0.0
            quadrature[i] = 0.0


@jit(nopython=True, cache=True)
def _ht_sine_numba(close: np.ndarray, sine: np.ndarray, leadsine: np.ndarray) -> None:
    """Numba-compiled HT_SINE calculation"""
    n = len(close)
    lookback = 32

    for i in range(lookback):
        sine[i] = np.nan
        leadsine[i] = np.nan

    smooth = np.zeros(n, dtype=np.float64)
    detrender = np.zeros(n, dtype=np.float64)
    i1 = np.zeros(n, dtype=np.float64)
    q1 = np.zeros(n, dtype=np.float64)
    ji = np.zeros(n, dtype=np.float64)
    jq = np.zeros(n, dtype=np.float64)
    i2 = np.zeros(n, dtype=np.float64)
    q2 = np.zeros(n, dtype=np.float64)
    re = np.zeros(n, dtype=np.float64)
    im = np.zeros(n, dtype=np.float64)
    period = np.zeros(n, dtype=np.float64)
    smooth_period = np.zeros(n, dtype=np.float64)

    for i in range(lookback, n):
        if i >= 3:
            smooth[i] = (4.0 * close[i] + 3.0 * close[i-1] + 2.0 * close[i-2] + close[i-3]) / 10.0
        else:
            smooth[i] = close[i]

        if i >= 6:
            detrender[i] = (0.0962 * smooth[i] + 0.5769 * smooth[i-2] -
                           0.5769 * smooth[i-4] - 0.0962 * smooth[i-6]) * (0.075 * period[i-1] + 0.54)

        if i >= 7:
            q1[i] = (0.0962 * detrender[i] + 0.5769 * detrender[i-2] -
                    0.5769 * detrender[i-4] - 0.0962 * detrender[i-6]) * (0.075 * period[i-1] + 0.54)
            i1[i] = detrender[i-3]

        if i >= 9:
            ji[i] = (0.0962 * i1[i] + 0.5769 * i1[i-2] -
                    0.5769 * i1[i-4] - 0.0962 * i1[i-6]) * (0.075 * period[i-1] + 0.54)
            jq[i] = (0.0962 * q1[i] + 0.5769 * q1[i-2] -
                    0.5769 * q1[i-4] - 0.0962 * q1[i-6]) * (0.075 * period[i-1] + 0.54)

        if i >= 10:
            i2[i] = i1[i] - jq[i]
            q2[i] = q1[i] + ji[i]
            i2[i] = 0.2 * i2[i] + 0.8 * i2[i-1]
            q2[i] = 0.2 * q2[i] + 0.8 * q2[i-1]

        if i >= 11:
            re[i] = i2[i] * i2[i-1] + q2[i] * q2[i-1]
            im[i] = i2[i] * q2[i-1] - q2[i] * i2[i-1]
            re[i] = 0.2 * re[i] + 0.8 * re[i-1]
            im[i] = 0.2 * im[i] + 0.8 * im[i-1]

            if im[i] != 0.0 and re[i] != 0.0:
                period[i] = 360.0 / (np.arctan(im[i] / re[i]) * 180.0 / np.pi)

            if period[i] > 1.5 * period[i-1]:
                period[i] = 1.5 * period[i-1]
            if period[i] < 0.67 * period[i-1]:
                period[i] = 0.67 * period[i-1]
            if period[i] < 6:
                period[i] = 6
            if period[i] > 50:
                period[i] = 50

            period[i] = 0.2 * period[i] + 0.8 * period[i-1]
            smooth_period[i] = 0.33 * period[i] + 0.67 * smooth_period[i-1]

            if i2[i] != 0.0:
                dc_phase = np.arctan(q2[i] / i2[i]) * 180.0 / np.pi
            else:
                dc_phase = 0.0

            if i2[i] < 0.0:
                dc_phase += 180.0
            if dc_phase < 0.0:
                dc_phase += 360.0

            dc_phase_rad = dc_phase * np.pi / 180.0
            sine[i] = np.sin(dc_phase_rad)
            leadsine[i] = np.sin(dc_phase_rad + 45.0 * np.pi / 180.0)


@jit(nopython=True, cache=True)
def _ht_trendline_numba(close: np.ndarray, output: np.ndarray) -> None:
    """Numba-compiled HT_TRENDLINE calculation"""
    n = len(close)
    lookback = 32

    for i in range(lookback):
        output[i] = np.nan

    for i in range(lookback, n):
        if i >= 3:
            output[i] = (4.0 * close[i] + 3.0 * close[i-1] + 2.0 * close[i-2] + close[i-3]) / 10.0
        else:
            output[i] = close[i]


@jit(nopython=True, cache=True)
def _ht_trendmode_numba(close: np.ndarray, output: np.ndarray) -> None:
    """Numba-compiled HT_TRENDMODE calculation"""
    n = len(close)
    lookback = 63

    for i in range(lookback):
        output[i] = np.nan

    smooth = np.zeros(n, dtype=np.float64)
    detrender = np.zeros(n, dtype=np.float64)
    period = np.zeros(n, dtype=np.float64)
    dc_period = np.zeros(n, dtype=np.float64)
    trend = np.zeros(n, dtype=np.float64)

    for i in range(lookback, n):
        if i >= 3:
            smooth[i] = (4.0 * close[i] + 3.0 * close[i-1] + 2.0 * close[i-2] + close[i-3]) / 10.0
        else:
            smooth[i] = close[i]

        if i >= 12:
            dc_period[i] = 15.0  # Simplified: use average period
            deviation = abs(close[i] - smooth[i])
            avg_deviation = 0.0
            count = 0
            for j in range(max(0, i-19), i+1):
                avg_deviation += abs(close[j] - smooth[j])
                count += 1
            avg_deviation = avg_deviation / count if count > 0 else 1.0

            if dc_period[i] > 30 or deviation > 2.0 * avg_deviation:
                trend[i] = 1.0
            else:
                trend[i] = 0.0

            if i > lookback:
                trend[i] = 0.8 * trend[i-1] + 0.2 * trend[i]

            output[i] = 1 if trend[i] > 0.5 else 0
        else:
            output[i] = 0
