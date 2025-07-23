"""
Fractal Text Compression

A modern text compression library using golden ratio splitting and LLM compression.
Supports fractal encoder for multi-level text compression.

Example:
    >>> from fractal_compress import FractalCompressor, smart_split, compress
    >>>
    >>> # Smart text splitting
    >>> part1, part2 = smart_split("Long text here...", ratio=0.5)
    >>>
    >>> # Fractal encoding
    >>> encoder = FractalCompressor(ratio=0.618, base_threshold=100)
    >>> fractal = encoder.compress([""], "Very long text content...")
    >>>
    >>> # Simple compression
    >>> result = compress("Text to compress...")
"""

# Simple compression functions
from .core import compress, llm_compress, split, split_compress

# Fractal encoder class (main implementation)
from .fractal_compressor_class import (
    FractalCompressor,
    create_fractal_compressor,
    encode_text_fractal,
)

# Text processing functions
from .text_processing import smart_split

__version__ = "1.0.0"
__author__ = "CogletNet Team"

__all__ = [
    # Simple compression
    "compress",
    "split_compress",
    "llm_compress",
    "split",
    # Text processing
    "smart_split",
    # Fractal encoder class
    "FractalCompressor",
    "create_fractal_compressor",
    "encode_text_fractal",
]
