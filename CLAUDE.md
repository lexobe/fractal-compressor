# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python text compression library called **Fractal Compress** that uses golden ratio splitting and LLM compression. The library provides both simple compression functions and advanced fractal compression capabilities.

### Core Architecture

The library is structured around three main components:

1. **Core API** (`src/fractal_compressor/core.py`): Simple compression functions (`compress`, `split_compress`, `llm_compress`, `split`)
2. **Text Processing** (`src/fractal_compressor/text_processing.py`): Smart text splitting with language-aware boundary detection
3. **LLM Integration** (`src/fractal_compressor/llm_compressor.py`): LLM-based compression using litellm with multiple strategies
4. **Fractal Encoder** (`src/fractal_compressor/fractal_compressor_class.py`): Advanced multi-level fractal compression

### Key Concepts

- **Golden Ratio Splitting**: Uses 0.382/0.618 ratio for optimal text segmentation
- **Language-Aware Processing**: Supports Chinese, English, and mixed content with optimized splitting
- **Multiple Compression Strategies**: "precise", "creative", and "fast" modes
- **Target Length Control**: Precise character count targeting (±5% accuracy)

## Development Commands

### Installation
```bash
pip install -e .
```

### Testing
```bash
# Run all tests with coverage
pytest

# Run specific test files
python tests/test_api.py
python tests/test_fractal_compressor.py

# Run examples
python examples/quickstart.py
python examples/advanced_usage.py
```

### Code Quality
```bash
# Format code
black src/ tests/ examples/

# Sort imports  
isort src/ tests/ examples/

# Type checking
mypy src/

# Run all quality checks
black src/ tests/ examples/ && isort src/ tests/ examples/ && mypy src/
```

### Package Building
```bash
# Build package
python -m build

# Install in editable mode for development
pip install -e .
```

## Environment Requirements

- Python 3.8+
- `OPENAI_API_KEY` environment variable must be set for LLM functionality
- Dependencies: `litellm>=1.0.0`

## Testing Strategy

Tests are organized by component:
- `test_api.py`: Core API functions
- `test_fractal_compressor.py`: Fractal compression algorithms
- `test_text_processing.py`: Text splitting and processing
- `test_llm_compressor.py`: LLM integration

All tests use the src path configuration in pyproject.toml and include coverage reporting.

## Code Patterns

### Import Structure
```python
# Core functions
from fractal_compress import compress, split_compress, llm_compress, split

# Advanced features  
from fractal_compress import FractalCompressor, smart_split
```

### Error Handling
- Empty text inputs return empty strings
- Invalid parameters raise `ValueError` with descriptive messages
- LLM failures are handled gracefully with fallback responses

### Language Detection
The library auto-detects content language but accepts explicit hints:
- `"auto"`: Automatic detection (default)
- `"chinese"`: Chinese-optimized processing
- `"english"`: English-optimized processing
- `"mixed"`: Mixed language content

## Configuration Files

- `pyproject.toml`: Complete project configuration including dependencies, testing, formatting, and build settings
- `README.md`: Comprehensive user documentation with examples
- Tests configured with pytest, coverage, and type checking via mypy