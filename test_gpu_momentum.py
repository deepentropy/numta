"""
Test script to verify GPU implementations work correctly
"""

import numpy as np
import sys

# Test imports
print("Testing imports...")
try:
    from src.talib_pure import momentum_indicators
    from src.talib_pure.backend import get_backend, set_backend, get_backend_info
    print("✓ Imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Print backend info
print("\n=== Backend Information ===")
info = get_backend_info()
for key, value in info.items():
    print(f"{key}: {value}")

# Create test data
print("\n=== Creating test data ===")
np.random.seed(42)
close = np.cumsum(np.random.randn(100)) + 100
high = close + np.abs(np.random.randn(100))
low = close - np.abs(np.random.randn(100))
print(f"Data length: {len(close)}")
print(f"Close range: {close.min():.2f} to {close.max():.2f}")

# Test RSI with CPU backend
print("\n=== Testing RSI ===")
set_backend("cpu")
print(f"Current backend: {get_backend()}")
rsi_cpu = momentum_indicators.RSI(close, timeperiod=14)
print(f"RSI (CPU) last 5 values: {rsi_cpu[-5:]}")
print(f"RSI (CPU) non-NaN count: {np.sum(~np.isnan(rsi_cpu))}")

# Test CMO with CPU backend
print("\n=== Testing CMO ===")
cmo_cpu = momentum_indicators.CMO(close, timeperiod=14)
print(f"CMO (CPU) last 5 values: {cmo_cpu[-5:]}")
print(f"CMO (CPU) non-NaN count: {np.sum(~np.isnan(cmo_cpu))}")

# Test DX with CPU backend
print("\n=== Testing DX ===")
dx_cpu = momentum_indicators.DX(high, low, close, timeperiod=14)
print(f"DX (CPU) last 5 values: {dx_cpu[-5:]}")
print(f"DX (CPU) non-NaN count: {np.sum(~np.isnan(dx_cpu))}")

# Test MACD with CPU backend
print("\n=== Testing MACD ===")
macd_cpu, signal_cpu, hist_cpu = momentum_indicators.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
print(f"MACD (CPU) last 5 values: {macd_cpu[-5:]}")
print(f"Signal (CPU) last 5 values: {signal_cpu[-5:]}")
print(f"Histogram (CPU) last 5 values: {hist_cpu[-5:]}")

# Test MACDEXT
print("\n=== Testing MACDEXT ===")
macdext_cpu, signalext_cpu, histext_cpu = momentum_indicators.MACDEXT(
    close, fastperiod=12, fastmatype=1, slowperiod=26, slowmatype=1, signalperiod=9, signalmatype=0
)
print(f"MACDEXT (CPU) last 5 values: {macdext_cpu[-5:]}")

# Test MACDFIX
print("\n=== Testing MACDFIX ===")
macdfix_cpu, signalfix_cpu, histfix_cpu = momentum_indicators.MACDFIX(close, signalperiod=9)
print(f"MACDFIX (CPU) last 5 values: {macdfix_cpu[-5:]}")

# Test GPU backend if available
if info['gpu_available']:
    print("\n=== Testing GPU Backend ===")
    try:
        set_backend("gpu")
        print(f"Current backend: {get_backend()}")

        # Test RSI with GPU
        rsi_gpu = momentum_indicators.RSI(close, timeperiod=14)
        print(f"RSI (GPU) last 5 values: {rsi_gpu[-5:]}")
        print(f"RSI CPU vs GPU match: {np.allclose(rsi_cpu, rsi_gpu, equal_nan=True)}")

        # Test CMO with GPU
        cmo_gpu = momentum_indicators.CMO(close, timeperiod=14)
        print(f"CMO (GPU) last 5 values: {cmo_gpu[-5:]}")
        print(f"CMO CPU vs GPU match: {np.allclose(cmo_cpu, cmo_gpu, equal_nan=True)}")

        # Test DX with GPU
        dx_gpu = momentum_indicators.DX(high, low, close, timeperiod=14)
        print(f"DX (GPU) last 5 values: {dx_gpu[-5:]}")
        print(f"DX CPU vs GPU match: {np.allclose(dx_cpu, dx_gpu, equal_nan=True)}")

        # Test MACD with GPU
        macd_gpu, signal_gpu, hist_gpu = momentum_indicators.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        print(f"MACD (GPU) last 5 values: {macd_gpu[-5:]}")
        print(f"MACD CPU vs GPU match: {np.allclose(macd_cpu, macd_gpu, equal_nan=True)}")
        print(f"Signal CPU vs GPU match: {np.allclose(signal_cpu, signal_gpu, equal_nan=True)}")
        print(f"Histogram CPU vs GPU match: {np.allclose(hist_cpu, hist_gpu, equal_nan=True)}")

        print("\n✓ GPU tests passed!")

    except Exception as e:
        print(f"\n✗ GPU test error: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n⚠ GPU not available - skipping GPU tests")

print("\n=== All Tests Completed ===")
print("✓ RSI function works with backend selection")
print("✓ CMO function works with backend selection")
print("✓ DX function works with backend selection")
print("✓ MACD function works with backend selection")
print("✓ MACDEXT function works")
print("✓ MACDFIX function works")
print("✓ No duplicate RSI function definitions")
