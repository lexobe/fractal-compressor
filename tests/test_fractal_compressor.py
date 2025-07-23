"""
简化的FractalCompressor测试，移除所有复杂的mock
"""

import pytest

from fractal_compressor.fractal_compressor_class import (
    FractalCompressor,
    create_fractal_compressor,
    encode_text_fractal,
)


class TestFractalCompressorBasic:
    """FractalCompressor基本功能测试"""

    def setup_method(self):
        """Setup方法"""
        llm_config = {
            "model": "gpt-4o-mini",
            "language": "mixed",
        }
        self.encoder = FractalCompressor(
            ratio=0.618,
            base_threshold=100,
            max_levels=5,
            llm_config=llm_config,
        )

    def test_compress_short_text(self):
        """测试压缩短文本"""
        fractal_text = [""]
        result = self.encoder.compress(fractal_text, "短文本")

        assert len(result) == 5  # max_levels
        assert result[0] == "短文本"
        assert all(result[i] == "" for i in range(1, 5))

    def test_compress_empty_text(self):
        """测试压缩空文本"""
        fractal_text = [""]
        result = self.encoder.compress(fractal_text, "")
        assert result[0] == ""

    def test_compress_to_existing_fractal(self):
        """测试压缩到已有分形结构"""
        fractal_text = ["现有内容"]
        result = self.encoder.compress(fractal_text, "新内容")

        # Level 0 应该直接连接
        assert "现有内容" in result[0]
        assert "新内容" in result[0]

    def test_basic_functionality(self):
        """基本功能测试"""
        text = "这是一个测试文本。"
        result = self.encoder.compress_single_text(text)
        
        # 基本验证
        assert isinstance(result, list)
        assert len(result) == 5
        assert all(isinstance(level, str) for level in result)


class TestConvenienceFunctions:
    """便捷函数测试"""

    def test_create_fractal_compressor(self):
        """测试创建分形编码器"""
        encoder = create_fractal_compressor(ratio=0.5, base_threshold=200)
        assert isinstance(encoder, FractalCompressor)
        assert encoder.ratio == 0.5
        assert encoder.base_threshold == 200

    def test_encode_text_fractal(self):
        """测试一站式编码"""
        text = "测试文本"
        result = encode_text_fractal(text)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert any(level for level in result if level.strip())