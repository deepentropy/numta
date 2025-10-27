# Development Tools

This folder contains development and testing utilities for the numta project.

## Contents

### Accuracy Testing
These scripts compare talib-pure outputs with the original TA-Lib library to ensure correctness:

- `accuracy_comparison.py` - General accuracy comparison framework
- `accuracy_math_operators.py` - Math operators accuracy tests
- `accuracy_overlap.py` - Overlap studies accuracy tests
- `accuracy_price_transform.py` - Price transform accuracy tests
- `accuracy_statistic_functions.py` - Statistical functions accuracy tests
- `ACCURACY.md` - Detailed accuracy test results and documentation

### Benchmarking
These scripts measure performance of various functions:

- `benchmark_comparison.py` - General benchmark comparison framework
- `benchmark_math_operators.py` - Math operators benchmarks
- `benchmark_overlap.py` - Overlap studies benchmarks
- `benchmark_price_transform.py` - Price transform benchmarks
- `benchmark_statistic_functions.py` - Statistical functions benchmarks

### Documentation
- `FUNCTION_IMPLEMENTATIONS.md` - Details about implemented functions
- `PERFORMANCE.md` - Performance analysis and optimization guides

## Usage

### Running Accuracy Tests
```bash
python development/accuracy_comparison.py
```

### Running Benchmarks
```bash
python development/benchmark_comparison.py
```

## Requirements

These tools require the original TA-Lib library installed for comparison:

```bash
# Install from project root with comparison dependencies
pip install -e ".[comparison]"
```

**Note**: Installing TA-Lib requires the TA-Lib C library to be installed on your system first. See the [TA-Lib installation guide](https://github.com/TA-Lib/ta-lib-python) for platform-specific instructions:

- **macOS**: `brew install ta-lib`
- **Linux**: Download and compile from source (see TA-Lib website)
- **Windows**: Use pre-built binaries or conda

These comparison tools are **optional** - the main library works without TA-Lib.
