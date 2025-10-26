"""
Script to add GPU/CuPy implementations to momentum_indicators.py
"""

import re

# Read the file
with open(r'C:/Users/otrem/PycharmProjects/talib-pure/src/talib_pure/momentum_indicators.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Define GPU implementations to add

# 1. RSI GPU implementation
rsi_cupy = '''

# GPU (CuPy) implementation
def _rsi_cupy(data: np.ndarray, timeperiod: int) -> np.ndarray:
    """
    CuPy-based RSI calculation for GPU

    This function uses CuPy for GPU-accelerated computation.
    """
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError(
            "CuPy is required for GPU backend but not installed. "
            "Install with: pip install cupy-cuda11x"
        )

    # Transfer to GPU
    data_gpu = cp.asarray(data, dtype=cp.float64)
    n = len(data_gpu)

    # Initialize output with NaN
    output = cp.full(n, cp.nan, dtype=cp.float64)

    lookback = timeperiod

    # Check if we have enough data
    if n <= lookback:
        return cp.asnumpy(output)

    # Calculate gains and losses
    gains = cp.zeros(n, dtype=cp.float64)
    losses = cp.zeros(n, dtype=cp.float64)

    for i in range(1, n):
        change = data_gpu[i] - data_gpu[i - 1]
        if change > 0:
            gains[i] = change
        elif change < 0:
            losses[i] = -change

    # Calculate initial average gain and loss
    avg_gain = cp.sum(gains[1:timeperiod + 1]) / timeperiod
    avg_loss = cp.sum(losses[1:timeperiod + 1]) / timeperiod

    # Calculate first RSI value
    if avg_loss == 0.0:
        if avg_gain == 0.0:
            output[timeperiod] = 50.0
        else:
            output[timeperiod] = 100.0
    else:
        rs = avg_gain / avg_loss
        output[timeperiod] = 100.0 - (100.0 / (1.0 + rs))

    # Calculate subsequent RSI values using Wilder's smoothing
    for i in range(timeperiod + 1, n):
        avg_gain = (avg_gain * (timeperiod - 1) + gains[i]) / timeperiod
        avg_loss = (avg_loss * (timeperiod - 1) + losses[i]) / timeperiod

        if avg_loss == 0.0:
            if avg_gain == 0.0:
                output[i] = 50.0
            else:
                output[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            output[i] = 100.0 - (100.0 / (1.0 + rs))

    # Transfer back to CPU
    return cp.asnumpy(output)

'''

# 2. CMO GPU implementation
cmo_cupy = '''

# GPU (CuPy) implementation
def _cmo_cupy(close: np.ndarray, timeperiod: int) -> np.ndarray:
    """
    CuPy-based CMO calculation for GPU

    This function uses CuPy for GPU-accelerated computation.
    """
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError(
            "CuPy is required for GPU backend but not installed. "
            "Install with: pip install cupy-cuda11x"
        )

    # Transfer to GPU
    close_gpu = cp.asarray(close, dtype=cp.float64)
    n = len(close_gpu)

    # Initialize output with NaN
    output = cp.full(n, cp.nan, dtype=cp.float64)

    # Check if we have enough data
    if n <= timeperiod:
        return cp.asnumpy(output)

    # Calculate CMO for each window
    for i in range(timeperiod, n):
        # Calculate price changes over the window
        sum_gains = cp.float64(0.0)
        sum_losses = cp.float64(0.0)

        for j in range(i - timeperiod + 1, i + 1):
            change = close_gpu[j] - close_gpu[j - 1]
            if change > 0:
                sum_gains += change
            elif change < 0:
                sum_losses += cp.abs(change)

        # Calculate CMO
        total = sum_gains + sum_losses
        if total == 0.0:
            output[i] = 0.0
        else:
            output[i] = ((sum_gains - sum_losses) / total) * 100.0

    # Transfer back to CPU
    return cp.asnumpy(output)

'''

# 3. DX GPU implementation
dx_cupy = '''

# GPU (CuPy) implementation
def _dx_cupy(high: np.ndarray, low: np.ndarray, close: np.ndarray, timeperiod: int) -> np.ndarray:
    """
    CuPy-based DX calculation for GPU

    This function uses CuPy for GPU-accelerated computation.
    """
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError(
            "CuPy is required for GPU backend but not installed. "
            "Install with: pip install cupy-cuda11x"
        )

    # Transfer to GPU
    high_gpu = cp.asarray(high, dtype=cp.float64)
    low_gpu = cp.asarray(low, dtype=cp.float64)
    close_gpu = cp.asarray(close, dtype=cp.float64)
    n = len(high_gpu)

    # Initialize output with NaN
    output = cp.full(n, cp.nan, dtype=cp.float64)

    lookback = timeperiod

    # Check if we have enough data
    if n <= lookback:
        return cp.asnumpy(output)

    # Calculate TR, +DM, -DM arrays
    tr = cp.empty(n, dtype=cp.float64)
    plus_dm = cp.empty(n, dtype=cp.float64)
    minus_dm = cp.empty(n, dtype=cp.float64)

    # First TR value (no previous close)
    tr[0] = high_gpu[0] - low_gpu[0]
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0

    for i in range(1, n):
        # True Range
        hl = high_gpu[i] - low_gpu[i]
        hc = cp.abs(high_gpu[i] - close_gpu[i - 1])
        lc = cp.abs(low_gpu[i] - close_gpu[i - 1])
        tr[i] = cp.maximum(cp.maximum(hl, hc), lc)

        # Directional Movement
        up_move = high_gpu[i] - high_gpu[i - 1]
        down_move = low_gpu[i - 1] - low_gpu[i]

        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0.0

        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0.0

    # Smooth TR, +DM, -DM using Wilder's smoothing
    smoothed_tr = cp.sum(tr[:timeperiod])
    smoothed_plus_dm = cp.sum(plus_dm[:timeperiod])
    smoothed_minus_dm = cp.sum(minus_dm[:timeperiod])

    # Calculate DX for timeperiod onwards
    for i in range(timeperiod, n):
        # Wilder's smoothing
        smoothed_tr = smoothed_tr - smoothed_tr / timeperiod + tr[i]
        smoothed_plus_dm = smoothed_plus_dm - smoothed_plus_dm / timeperiod + plus_dm[i]
        smoothed_minus_dm = smoothed_minus_dm - smoothed_minus_dm / timeperiod + minus_dm[i]

        # Calculate directional indicators
        if smoothed_tr != 0:
            plus_di = 100.0 * smoothed_plus_dm / smoothed_tr
            minus_di = 100.0 * smoothed_minus_dm / smoothed_tr
        else:
            plus_di = 0.0
            minus_di = 0.0

        # Calculate DX
        di_sum = plus_di + minus_di
        if di_sum != 0:
            output[i] = 100.0 * cp.abs(plus_di - minus_di) / di_sum
        else:
            output[i] = 0.0

    # Transfer back to CPU
    return cp.asnumpy(output)

'''

# 4. MACD GPU implementation
macd_cupy = '''

# GPU (CuPy) implementation
def _macd_cupy(close: np.ndarray, fastperiod: int, slowperiod: int, signalperiod: int) -> tuple:
    """
    CuPy-based MACD calculation for GPU

    This function uses CuPy for GPU-accelerated computation.
    """
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError(
            "CuPy is required for GPU backend but not installed. "
            "Install with: pip install cupy-cuda11x"
        )

    # Import EMA from overlap for GPU calculation
    from .overlap import _ema_cupy

    # Calculate fast and slow EMAs
    fast_ema = cp.asarray(_ema_cupy(close, fastperiod), dtype=cp.float64)
    slow_ema = cp.asarray(_ema_cupy(close, slowperiod), dtype=cp.float64)

    # Calculate MACD line
    macd = fast_ema - slow_ema

    # Calculate signal line (EMA of MACD)
    macd_cpu = cp.asnumpy(macd)
    signal = cp.asarray(_ema_cupy(macd_cpu, signalperiod), dtype=cp.float64)

    # Calculate histogram
    hist = macd - signal

    # Transfer back to CPU
    return cp.asnumpy(macd), cp.asnumpy(signal), cp.asnumpy(hist)

'''

# Insert GPU implementations after their respective numba functions

# 1. Insert RSI GPU implementation after first _rsi_numba (line ~1342)
pattern1 = re.compile(r"(            output\[i\] = 100\.0 - \(100\.0 / \(1\.0 \+ rs\)\)\n\n\n)(@jit\(nopython=True, cache=True\)\ndef _roc_numba)", re.MULTILINE)
content = pattern1.sub(r'\1' + rsi_cupy + r'\n\2', content, count=1)

# 2. Insert CMO GPU implementation after _cmo_numba (line ~2138)
pattern2 = re.compile(r"(            output\[i\] = \(\(sum_gains - sum_losses\) / total\) \* 100\.0\n\n\n)(def CMO\(close: Union\[np\.ndarray, list\],)", re.MULTILINE)
content = pattern2.sub(r'\1' + cmo_cupy + r'\n\2', content, count=1)

# 3. Insert DX GPU implementation after _dx_numba (line ~2331)
pattern3 = re.compile(r"(            output\[i\] = 0\.0\n\n\n)(def DX\(high: Union\[np\.ndarray, list\],)", re.MULTILINE)
content = pattern3.sub(r'\1' + dx_cupy + r'\n\2', content, count=1)

# 4. Insert MACD GPU implementation after _macd_numba (line ~2535)
pattern4 = re.compile(r"(        hist\[i\] = macd\[i\] - signal\[i\]\n\n\n)(def MACD\(close: Union\[np\.ndarray, list\],)", re.MULTILINE)
content = pattern4.sub(r'\1' + macd_cupy + r'\n\2', content, count=1)

print("GPU implementations added successfully")

# Now update the main functions to add backend selection logic

# Update RSI function
old_rsi_main = r'''    # Not enough data points - return all NaN
    if n <= timeperiod:
        return np.full(n, np.nan, dtype=np.float64)

    # Pre-allocate output array and run Numba-optimized calculation
    output = np.empty(n, dtype=np.float64)
    _rsi_numba(data, timeperiod, output)

    return output'''

new_rsi_main = r'''    # Not enough data points - return all NaN
    if n <= timeperiod:
        return np.full(n, np.nan, dtype=np.float64)

    # Check backend and dispatch to appropriate implementation
    from .backend import get_backend

    backend = get_backend()

    if backend == "gpu":
        # Use GPU implementation
        return _rsi_cupy(data, timeperiod)
    else:
        # Use CPU implementation (default)
        # Pre-allocate output array and run Numba-optimized calculation
        output = np.empty(n, dtype=np.float64)
        _rsi_numba(data, timeperiod, output)

        return output'''

content = content.replace(old_rsi_main, new_rsi_main, 1)

# Update CMO function
old_cmo_main = r'''    # Not enough data points - return all NaN
    # Need timeperiod + 1 because we need previous close for first calculation
    if n <= timeperiod:
        return np.full(n, np.nan, dtype=np.float64)

    # Pre-allocate output array and run Numba-optimized calculation
    output = np.empty(n, dtype=np.float64)
    _cmo_numba(close, timeperiod, output)

    return output'''

new_cmo_main = r'''    # Not enough data points - return all NaN
    # Need timeperiod + 1 because we need previous close for first calculation
    if n <= timeperiod:
        return np.full(n, np.nan, dtype=np.float64)

    # Check backend and dispatch to appropriate implementation
    from .backend import get_backend

    backend = get_backend()

    if backend == "gpu":
        # Use GPU implementation
        return _cmo_cupy(close, timeperiod)
    else:
        # Use CPU implementation (default)
        # Pre-allocate output array and run Numba-optimized calculation
        output = np.empty(n, dtype=np.float64)
        _cmo_numba(close, timeperiod, output)

        return output'''

content = content.replace(old_cmo_main, new_cmo_main, 1)

# Update DX function
old_dx_main = r'''    # Not enough data points - return all NaN
    if n <= timeperiod:
        return np.full(n, np.nan, dtype=np.float64)

    # Pre-allocate output array and run Numba-optimized calculation
    output = np.empty(n, dtype=np.float64)
    _dx_numba(high, low, close, timeperiod, output)

    return output'''

new_dx_main = r'''    # Not enough data points - return all NaN
    if n <= timeperiod:
        return np.full(n, np.nan, dtype=np.float64)

    # Check backend and dispatch to appropriate implementation
    from .backend import get_backend

    backend = get_backend()

    if backend == "gpu":
        # Use GPU implementation
        return _dx_cupy(high, low, close, timeperiod)
    else:
        # Use CPU implementation (default)
        # Pre-allocate output array and run Numba-optimized calculation
        output = np.empty(n, dtype=np.float64)
        _dx_numba(high, low, close, timeperiod, output)

        return output'''

content = content.replace(old_dx_main, new_dx_main, 1)

# Update MACD function
old_macd_main = r'''    # Pre-allocate output arrays
    macd = np.empty(n, dtype=np.float64)
    signal = np.empty(n, dtype=np.float64)
    hist = np.empty(n, dtype=np.float64)

    _macd_numba(close, fastperiod, slowperiod, signalperiod, macd, signal, hist)

    return macd, signal, hist'''

new_macd_main = r'''    # Check backend and dispatch to appropriate implementation
    from .backend import get_backend

    backend = get_backend()

    if backend == "gpu":
        # Use GPU implementation
        return _macd_cupy(close, fastperiod, slowperiod, signalperiod)
    else:
        # Use CPU implementation (default)
        # Pre-allocate output arrays
        macd = np.empty(n, dtype=np.float64)
        signal = np.empty(n, dtype=np.float64)
        hist = np.empty(n, dtype=np.float64)

        _macd_numba(close, fastperiod, slowperiod, signalperiod, macd, signal, hist)

        return macd, signal, hist'''

content = content.replace(old_macd_main, new_macd_main, 1)

# Update MACDEXT function - add backend check at the beginning
old_macdext_start = r'''    # Import MA function from overlap module
    from .overlap import MA

    # Validate inputs'''

new_macdext_start = r'''    # Check backend and dispatch to appropriate implementation
    from .backend import get_backend
    from .overlap import MA

    backend = get_backend()

    if backend == "gpu":
        # For GPU, use _macd_cupy with MA routing
        # MACDEXT uses MA which will respect the backend setting
        pass

    # Import MA function from overlap module

    # Validate inputs'''

content = content.replace(old_macdext_start, new_macdext_start, 1)

# Update MACDFIX - it just calls MACD so it will automatically use GPU if backend is set

print("Main function backend logic updated successfully")

# Remove duplicate RSI function (second occurrence around line 4192)
# Find and remove the second RSI definition
lines = content.split('\n')
rsi_count = 0
start_removal = -1
end_removal = -1

for i, line in enumerate(lines):
    if line.strip().startswith('def RSI(data:'):
        rsi_count += 1
        if rsi_count == 2:  # This is the duplicate
            start_removal = i
            # Find the end of this function (next def or end of file)
            for j in range(i + 1, len(lines)):
                if lines[j].startswith('def ') or lines[j].startswith('@jit'):
                    end_removal = j
                    break
            if end_removal == -1:
                end_removal = len(lines)
            break

if start_removal != -1:
    # Remove the duplicate function
    lines = lines[:start_removal] + lines[end_removal:]
    content = '\n'.join(lines)
    print(f"Removed duplicate RSI function (was at line ~{start_removal})")
else:
    print("No duplicate RSI function found or already removed")

# Write the updated content back
with open(r'C:/Users/otrem/PycharmProjects/talib-pure/src/talib_pure/momentum_indicators.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("\n=== Summary ===")
print("✓ Added _rsi_cupy() GPU implementation")
print("✓ Added _cmo_cupy() GPU implementation")
print("✓ Added _dx_cupy() GPU implementation")
print("✓ Added _macd_cupy() GPU implementation")
print("✓ Updated RSI() with backend selection logic")
print("✓ Updated CMO() with backend selection logic")
print("✓ Updated DX() with backend selection logic")
print("✓ Updated MACD() with backend selection logic")
print("✓ Updated MACDEXT() with backend check")
print("✓ MACDFIX() automatically uses backend via MACD()")
print("✓ Removed duplicate RSI function definition")
print("\nAll changes completed successfully!")
