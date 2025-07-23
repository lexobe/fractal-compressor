"""
Fractal Text Compression Library

A modern text compression library using golden ratio splitting and LLM compression.
"""

import time
from typing import Optional, Tuple

from .llm_compressor import compress_with_llm
from .text_processing import smart_split as split_text


def compress(
    text: str,
    split_ratio: float = 0.382,
    compression_ratio: float = 0.618,
    model: str = "gpt-4o-mini",
    language: str = "auto",
) -> str:
    """
    Compress text using fractal compression algorithm.

    Args:
        text: Input text to compress
        split_ratio: Ratio for text splitting (default: golden ratio)
        compression_ratio: Target compression ratio for LLM
        model: LLM model to use
        language: Language hint ("chinese", "english", "auto")

    Returns:
        Compressed text string

    Example:
        >>> compress("Long text here...", split_ratio=0.4, compression_ratio=0.5)
        "Compressed text"
    """
    if not text or not text.strip():
        return ""

    # Step 1: Split text using golden ratio
    first_part, _ = split_text(text, ratio=split_ratio, language=language)

    # Step 2: Calculate target length for compression
    target_length = max(1, int(len(first_part) * compression_ratio))

    # Step 3: Compress using LLM
    compressed = compress_with_llm(
        text=first_part, target_length=target_length, model=model, language=language
    )

    return compressed


def split_compress(
    text: str,
    split_ratio: float = 0.382,
    compression_ratio: float = 0.618,
    model: str = "gpt-4o-mini",
    language: str = "auto",
) -> Tuple[str, str]:
    """
    Split text and compress the first part, return both parts.

    Args:
        text: Input text to process
        split_ratio: Ratio for text splitting
        compression_ratio: Target compression ratio for first part
        model: LLM model to use
        language: Language hint

    Returns:
        Tuple of (compressed_first_part, second_part)

    Example:
        >>> compressed, remaining = split_compress("Long text...")
        >>> print(f"Compressed: {compressed}")
        >>> print(f"Remaining: {remaining}")
    """
    if not text or not text.strip():
        return "", ""

    # Split text
    first_part, second_part = split_text(text, ratio=split_ratio, language=language)

    # Compress first part
    target_length = max(1, int(len(first_part) * compression_ratio))
    compressed_first = compress_with_llm(
        text=first_part, target_length=target_length, model=model, language=language
    )

    return compressed_first, second_part


def llm_compress(
    text: str,
    target_length: int,
    model: str = "gpt-4o-mini",
    language: str = "auto",
    strategy: str = "precise",
) -> str:
    """
    Compress text directly using LLM to target length.

    Args:
        text: Input text to compress
        target_length: Exact target length in characters
        model: LLM model to use
        language: Language hint
        strategy: Compression strategy ("precise", "creative", "fast")

    Returns:
        Compressed text string

    Example:
        >>> result = llm_compress("Long text here...", target_length=20)
        >>> len(result) <= 20
        True
    """
    if not text or not text.strip():
        return ""

    if target_length <= 0:
        return ""

    return compress_with_llm(
        text=text,
        target_length=target_length,
        model=model,
        language=language,
        strategy=strategy,
    )


def split(text: str, ratio: float = 0.382, language: str = "auto") -> Tuple[str, str]:
    """
    Split text at optimal position using intelligent boundary detection.

    Args:
        text: Input text to split
        ratio: Split ratio (0.0 to 1.0)
        language: Language hint for better splitting

    Returns:
        Tuple of (first_part, second_part)

    Example:
        >>> part1, part2 = split("Hello world! How are you?", ratio=0.5)
        >>> print(f"Part 1: {part1}")
        >>> print(f"Part 2: {part2}")
    """
    if not text or not text.strip():
        return "", ""

    return split_text(text, ratio=ratio, language=language)
