"""Update __init__.py to import the momentum functions"""

with open(r'C:/Users/otrem/PycharmProjects/talib-pure/src/talib_pure/__init__.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the momentum indicators import line
old_import = 'from .momentum_indicators import ADX, ADXR, APO, AROON, AROONOSC, ATR, BOP, CCI, STOCH, STOCHF, STOCHRSI'
new_import = 'from .momentum_indicators import ADX, ADXR, APO, AROON, AROONOSC, ATR, BOP, CCI, CMO, DX, MACD, MACDEXT, MACDFIX, MFI, MINUS_DI, MINUS_DM, MOM, PLUS_DI, PLUS_DM, PPO, ROC, ROCP, ROCR, ROCR100, RSI, STOCH, STOCHF, STOCHRSI, TRIX, ULTOSC, WILLR'

content = content.replace(old_import, new_import)

with open(r'C:/Users/otrem/PycharmProjects/talib-pure/src/talib_pure/__init__.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated __init__.py imports successfully")
