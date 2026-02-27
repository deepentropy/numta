"""
Cycle Indicators - GPU batch CUDA kernels.

Each kernel processes one ticker per CUDA thread.
Hilbert Transform kernels require pre-allocated workspace arrays from host.
"""

import math
from numba import cuda


@cuda.jit
def _ht_trendline_batch_cuda(close_2d, output_2d):
    tid = cuda.grid(1)
    if tid >= close_2d.shape[0]:
        return
    n = close_2d.shape[1]
    lookback = 32

    for i in range(lookback):
        output_2d[tid, i] = math.nan

    for i in range(lookback, n):
        if i >= 3:
            output_2d[tid, i] = (4.0 * close_2d[tid, i] + 3.0 * close_2d[tid, i-1] +
                                  2.0 * close_2d[tid, i-2] + close_2d[tid, i-3]) / 10.0
        else:
            output_2d[tid, i] = close_2d[tid, i]


@cuda.jit
def _ht_trendmode_batch_cuda(close_2d, output_2d):
    tid = cuda.grid(1)
    if tid >= close_2d.shape[0]:
        return
    n = close_2d.shape[1]
    lookback = 63

    for i in range(lookback):
        output_2d[tid, i] = math.nan

    smooth_val = 0.0
    trend = 0.0

    for i in range(lookback, n):
        if i >= 3:
            smooth_val = (4.0 * close_2d[tid, i] + 3.0 * close_2d[tid, i-1] +
                          2.0 * close_2d[tid, i-2] + close_2d[tid, i-3]) / 10.0
        else:
            smooth_val = close_2d[tid, i]

        if i >= 12:
            deviation = abs(close_2d[tid, i] - smooth_val)
            avg_dev = 0.0
            count = 0
            start = i - 19
            if start < 0:
                start = 0
            for j in range(start, i + 1):
                # CPU stores smooth in array initialized to 0; only fills from lookback
                if j >= lookback and j >= 3:
                    sm = (4.0 * close_2d[tid, j] + 3.0 * close_2d[tid, j-1] +
                          2.0 * close_2d[tid, j-2] + close_2d[tid, j-3]) / 10.0
                else:
                    sm = 0.0
                avg_dev += abs(close_2d[tid, j] - sm)
                count += 1
            if count > 0:
                avg_dev = avg_dev / count
            else:
                avg_dev = 1.0

            if deviation > 2.0 * avg_dev:
                new_trend = 1.0
            else:
                new_trend = 0.0

            if i > lookback:
                trend = 0.8 * trend + 0.2 * new_trend
            else:
                trend = new_trend

            if trend > 0.5:
                output_2d[tid, i] = 1.0
            else:
                output_2d[tid, i] = 0.0
        else:
            output_2d[tid, i] = 0.0


@cuda.jit
def _ht_dcperiod_batch_cuda(close_2d, smooth_w, detrender_w, i1_w, q1_w,
                             ji_w, jq_w, i2_w, q2_w, re_w, im_w,
                             period_w, smooth_period_w, output_2d):
    """Hilbert Transform Dominant Cycle Period. Workspace arrays: (num_tickers, num_bars)."""
    tid = cuda.grid(1)
    if tid >= close_2d.shape[0]:
        return
    n = close_2d.shape[1]
    lookback = 32

    for i in range(min(lookback, n)):
        output_2d[tid, i] = math.nan

    for i in range(lookback, n):
        if i >= 3:
            smooth_w[tid, i] = (4.0 * close_2d[tid, i] + 3.0 * close_2d[tid, i-1] +
                                 2.0 * close_2d[tid, i-2] + close_2d[tid, i-3]) / 10.0
        else:
            smooth_w[tid, i] = close_2d[tid, i]

        if i >= 6:
            detrender_w[tid, i] = (0.0962 * smooth_w[tid, i] + 0.5769 * smooth_w[tid, i-2] -
                                    0.5769 * smooth_w[tid, i-4] - 0.0962 * smooth_w[tid, i-6]) * \
                                   (0.075 * period_w[tid, i-1] + 0.54)

        if i >= 7:
            q1_w[tid, i] = (0.0962 * detrender_w[tid, i] + 0.5769 * detrender_w[tid, i-2] -
                             0.5769 * detrender_w[tid, i-4] - 0.0962 * detrender_w[tid, i-6]) * \
                            (0.075 * period_w[tid, i-1] + 0.54)
            i1_w[tid, i] = detrender_w[tid, i-3]

        if i >= 9:
            ji_w[tid, i] = (0.0962 * i1_w[tid, i] + 0.5769 * i1_w[tid, i-2] -
                             0.5769 * i1_w[tid, i-4] - 0.0962 * i1_w[tid, i-6]) * \
                            (0.075 * period_w[tid, i-1] + 0.54)
            jq_w[tid, i] = (0.0962 * q1_w[tid, i] + 0.5769 * q1_w[tid, i-2] -
                             0.5769 * q1_w[tid, i-4] - 0.0962 * q1_w[tid, i-6]) * \
                            (0.075 * period_w[tid, i-1] + 0.54)

        if i >= 10:
            i2_w[tid, i] = i1_w[tid, i] - jq_w[tid, i]
            q2_w[tid, i] = q1_w[tid, i] + ji_w[tid, i]
            i2_w[tid, i] = 0.2 * i2_w[tid, i] + 0.8 * i2_w[tid, i-1]
            q2_w[tid, i] = 0.2 * q2_w[tid, i] + 0.8 * q2_w[tid, i-1]

        if i >= 11:
            re_w[tid, i] = i2_w[tid, i] * i2_w[tid, i-1] + q2_w[tid, i] * q2_w[tid, i-1]
            im_w[tid, i] = i2_w[tid, i] * q2_w[tid, i-1] - q2_w[tid, i] * i2_w[tid, i-1]
            re_w[tid, i] = 0.2 * re_w[tid, i] + 0.8 * re_w[tid, i-1]
            im_w[tid, i] = 0.2 * im_w[tid, i] + 0.8 * im_w[tid, i-1]

            if im_w[tid, i] != 0.0 and re_w[tid, i] != 0.0:
                period_w[tid, i] = 360.0 / (math.atan(im_w[tid, i] / re_w[tid, i]) * 180.0 / math.pi)

            if period_w[tid, i] > 1.5 * period_w[tid, i-1]:
                period_w[tid, i] = 1.5 * period_w[tid, i-1]
            if period_w[tid, i] < 0.67 * period_w[tid, i-1]:
                period_w[tid, i] = 0.67 * period_w[tid, i-1]
            if period_w[tid, i] < 6:
                period_w[tid, i] = 6.0
            if period_w[tid, i] > 50:
                period_w[tid, i] = 50.0

            period_w[tid, i] = 0.2 * period_w[tid, i] + 0.8 * period_w[tid, i-1]
            smooth_period_w[tid, i] = 0.33 * period_w[tid, i] + 0.67 * smooth_period_w[tid, i-1]
        else:
            period_w[tid, i] = 0.0
            smooth_period_w[tid, i] = 0.0

        output_2d[tid, i] = smooth_period_w[tid, i]


@cuda.jit
def _ht_dcphase_batch_cuda(close_2d, smooth_w, detrender_w, i1_w, q1_w,
                            ji_w, jq_w, i2_w, q2_w, period_w, output_2d):
    """Hilbert Transform Dominant Cycle Phase. Workspace arrays: (num_tickers, num_bars)."""
    tid = cuda.grid(1)
    if tid >= close_2d.shape[0]:
        return
    n = close_2d.shape[1]
    lookback = 32

    for i in range(min(lookback, n)):
        output_2d[tid, i] = math.nan

    for i in range(lookback, n):
        if i >= 3:
            smooth_w[tid, i] = (4.0 * close_2d[tid, i] + 3.0 * close_2d[tid, i-1] +
                                 2.0 * close_2d[tid, i-2] + close_2d[tid, i-3]) / 10.0
        else:
            smooth_w[tid, i] = close_2d[tid, i]

        if i >= 6:
            detrender_w[tid, i] = (0.0962 * smooth_w[tid, i] + 0.5769 * smooth_w[tid, i-2] -
                                    0.5769 * smooth_w[tid, i-4] - 0.0962 * smooth_w[tid, i-6]) * \
                                   (0.075 * period_w[tid, i-1] + 0.54)

        if i >= 7:
            q1_w[tid, i] = (0.0962 * detrender_w[tid, i] + 0.5769 * detrender_w[tid, i-2] -
                             0.5769 * detrender_w[tid, i-4] - 0.0962 * detrender_w[tid, i-6]) * \
                            (0.075 * period_w[tid, i-1] + 0.54)
            i1_w[tid, i] = detrender_w[tid, i-3]

        if i >= 9:
            ji_w[tid, i] = (0.0962 * i1_w[tid, i] + 0.5769 * i1_w[tid, i-2] -
                             0.5769 * i1_w[tid, i-4] - 0.0962 * i1_w[tid, i-6]) * \
                            (0.075 * period_w[tid, i-1] + 0.54)
            jq_w[tid, i] = (0.0962 * q1_w[tid, i] + 0.5769 * q1_w[tid, i-2] -
                             0.5769 * q1_w[tid, i-4] - 0.0962 * q1_w[tid, i-6]) * \
                            (0.075 * period_w[tid, i-1] + 0.54)

        if i >= 10:
            i2_w[tid, i] = i1_w[tid, i] - jq_w[tid, i]
            q2_w[tid, i] = q1_w[tid, i] + ji_w[tid, i]
            i2_w[tid, i] = 0.2 * i2_w[tid, i] + 0.8 * i2_w[tid, i-1]
            q2_w[tid, i] = 0.2 * q2_w[tid, i] + 0.8 * q2_w[tid, i-1]

        if i >= 11:
            dc_phase = 0.0
            if i2_w[tid, i] != 0.0:
                dc_phase = math.atan(q2_w[tid, i] / i2_w[tid, i]) * 180.0 / math.pi
            if i2_w[tid, i] < 0.0:
                dc_phase += 180.0
            if dc_phase < 0.0:
                dc_phase += 360.0
            output_2d[tid, i] = dc_phase
        else:
            output_2d[tid, i] = 0.0


@cuda.jit
def _ht_phasor_batch_cuda(close_2d, smooth_w, detrender_w, i1_w, q1_w,
                           ji_w, jq_w, i2_w, q2_w, period_w,
                           inphase_2d, quadrature_2d):
    """Hilbert Transform Phasor. Workspace arrays: (num_tickers, num_bars)."""
    tid = cuda.grid(1)
    if tid >= close_2d.shape[0]:
        return
    n = close_2d.shape[1]
    lookback = 32

    for i in range(min(lookback, n)):
        inphase_2d[tid, i] = math.nan
        quadrature_2d[tid, i] = math.nan

    for i in range(lookback, n):
        if i >= 3:
            smooth_w[tid, i] = (4.0 * close_2d[tid, i] + 3.0 * close_2d[tid, i-1] +
                                 2.0 * close_2d[tid, i-2] + close_2d[tid, i-3]) / 10.0
        else:
            smooth_w[tid, i] = close_2d[tid, i]

        if i >= 6:
            detrender_w[tid, i] = (0.0962 * smooth_w[tid, i] + 0.5769 * smooth_w[tid, i-2] -
                                    0.5769 * smooth_w[tid, i-4] - 0.0962 * smooth_w[tid, i-6]) * \
                                   (0.075 * period_w[tid, i-1] + 0.54)

        if i >= 7:
            q1_w[tid, i] = (0.0962 * detrender_w[tid, i] + 0.5769 * detrender_w[tid, i-2] -
                             0.5769 * detrender_w[tid, i-4] - 0.0962 * detrender_w[tid, i-6]) * \
                            (0.075 * period_w[tid, i-1] + 0.54)
            i1_w[tid, i] = detrender_w[tid, i-3]

        if i >= 9:
            ji_w[tid, i] = (0.0962 * i1_w[tid, i] + 0.5769 * i1_w[tid, i-2] -
                             0.5769 * i1_w[tid, i-4] - 0.0962 * i1_w[tid, i-6]) * \
                            (0.075 * period_w[tid, i-1] + 0.54)
            jq_w[tid, i] = (0.0962 * q1_w[tid, i] + 0.5769 * q1_w[tid, i-2] -
                             0.5769 * q1_w[tid, i-4] - 0.0962 * q1_w[tid, i-6]) * \
                            (0.075 * period_w[tid, i-1] + 0.54)

        if i >= 10:
            i2_w[tid, i] = i1_w[tid, i] - jq_w[tid, i]
            q2_w[tid, i] = q1_w[tid, i] + ji_w[tid, i]
            i2_w[tid, i] = 0.2 * i2_w[tid, i] + 0.8 * i2_w[tid, i-1]
            q2_w[tid, i] = 0.2 * q2_w[tid, i] + 0.8 * q2_w[tid, i-1]
            inphase_2d[tid, i] = i2_w[tid, i]
            quadrature_2d[tid, i] = q2_w[tid, i]
        else:
            inphase_2d[tid, i] = 0.0
            quadrature_2d[tid, i] = 0.0


@cuda.jit
def _ht_sine_batch_cuda(close_2d, smooth_w, detrender_w, i1_w, q1_w,
                         ji_w, jq_w, i2_w, q2_w, re_w, im_w,
                         period_w, smooth_period_w,
                         sine_2d, leadsine_2d):
    """Hilbert Transform Sine. Workspace arrays: (num_tickers, num_bars)."""
    tid = cuda.grid(1)
    if tid >= close_2d.shape[0]:
        return
    n = close_2d.shape[1]
    lookback = 32

    for i in range(min(lookback, n)):
        sine_2d[tid, i] = math.nan
        leadsine_2d[tid, i] = math.nan

    for i in range(lookback, n):
        if i >= 3:
            smooth_w[tid, i] = (4.0 * close_2d[tid, i] + 3.0 * close_2d[tid, i-1] +
                                 2.0 * close_2d[tid, i-2] + close_2d[tid, i-3]) / 10.0
        else:
            smooth_w[tid, i] = close_2d[tid, i]

        if i >= 6:
            detrender_w[tid, i] = (0.0962 * smooth_w[tid, i] + 0.5769 * smooth_w[tid, i-2] -
                                    0.5769 * smooth_w[tid, i-4] - 0.0962 * smooth_w[tid, i-6]) * \
                                   (0.075 * period_w[tid, i-1] + 0.54)

        if i >= 7:
            q1_w[tid, i] = (0.0962 * detrender_w[tid, i] + 0.5769 * detrender_w[tid, i-2] -
                             0.5769 * detrender_w[tid, i-4] - 0.0962 * detrender_w[tid, i-6]) * \
                            (0.075 * period_w[tid, i-1] + 0.54)
            i1_w[tid, i] = detrender_w[tid, i-3]

        if i >= 9:
            ji_w[tid, i] = (0.0962 * i1_w[tid, i] + 0.5769 * i1_w[tid, i-2] -
                             0.5769 * i1_w[tid, i-4] - 0.0962 * i1_w[tid, i-6]) * \
                            (0.075 * period_w[tid, i-1] + 0.54)
            jq_w[tid, i] = (0.0962 * q1_w[tid, i] + 0.5769 * q1_w[tid, i-2] -
                             0.5769 * q1_w[tid, i-4] - 0.0962 * q1_w[tid, i-6]) * \
                            (0.075 * period_w[tid, i-1] + 0.54)

        if i >= 10:
            i2_w[tid, i] = i1_w[tid, i] - jq_w[tid, i]
            q2_w[tid, i] = q1_w[tid, i] + ji_w[tid, i]
            i2_w[tid, i] = 0.2 * i2_w[tid, i] + 0.8 * i2_w[tid, i-1]
            q2_w[tid, i] = 0.2 * q2_w[tid, i] + 0.8 * q2_w[tid, i-1]

        if i >= 11:
            re_w[tid, i] = i2_w[tid, i] * i2_w[tid, i-1] + q2_w[tid, i] * q2_w[tid, i-1]
            im_w[tid, i] = i2_w[tid, i] * q2_w[tid, i-1] - q2_w[tid, i] * i2_w[tid, i-1]
            re_w[tid, i] = 0.2 * re_w[tid, i] + 0.8 * re_w[tid, i-1]
            im_w[tid, i] = 0.2 * im_w[tid, i] + 0.8 * im_w[tid, i-1]

            if im_w[tid, i] != 0.0 and re_w[tid, i] != 0.0:
                period_w[tid, i] = 360.0 / (math.atan(im_w[tid, i] / re_w[tid, i]) * 180.0 / math.pi)

            if period_w[tid, i] > 1.5 * period_w[tid, i-1]:
                period_w[tid, i] = 1.5 * period_w[tid, i-1]
            if period_w[tid, i] < 0.67 * period_w[tid, i-1]:
                period_w[tid, i] = 0.67 * period_w[tid, i-1]
            if period_w[tid, i] < 6:
                period_w[tid, i] = 6.0
            if period_w[tid, i] > 50:
                period_w[tid, i] = 50.0

            period_w[tid, i] = 0.2 * period_w[tid, i] + 0.8 * period_w[tid, i-1]
            smooth_period_w[tid, i] = 0.33 * period_w[tid, i] + 0.67 * smooth_period_w[tid, i-1]

            dc_phase = 0.0
            if i2_w[tid, i] != 0.0:
                dc_phase = math.atan(q2_w[tid, i] / i2_w[tid, i]) * 180.0 / math.pi
            if i2_w[tid, i] < 0.0:
                dc_phase += 180.0
            if dc_phase < 0.0:
                dc_phase += 360.0

            dc_phase_rad = dc_phase * math.pi / 180.0
            sine_2d[tid, i] = math.sin(dc_phase_rad)
            leadsine_2d[tid, i] = math.sin(dc_phase_rad + 45.0 * math.pi / 180.0)
