"""
测试 FractalCompressor 类的功能
使用真实 LLM API 调用，不使用 mock
"""

import os
import sys
from typing import Any, Dict

import pytest

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fractal_compressor.fractal_compressor_class import (
    FractalCompressor,
    create_fractal_compressor,
    encode_text_fractal,
)


class TestFractalCompressorBasic:
    """测试 FractalCompressor 基本功能"""

    @pytest.fixture
    def sample_llm_config(self) -> Dict[str, Any]:
        """提供测试用的 LLM 配置"""
        return {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "url": "https://api.openai.com/v1",
            "temperature": 0.1,
            "max_tokens": 2000,
            "language": "mixed",
        }

    @pytest.fixture
    def encoder(self, sample_llm_config) -> FractalCompressor:
        """创建测试用的 FractalCompressor 实例"""
        return FractalCompressor(
            ratio=0.618, base_threshold=100, max_levels=5, llm_config=sample_llm_config
        )

    def test_init_default_parameters(self):
        """测试默认参数初始化"""
        encoder = FractalCompressor()
        assert encoder.ratio == 0.618
        assert encoder.base_threshold == 1000
        assert encoder.max_levels == 10
        assert encoder.llm_config == {}

    def test_init_custom_parameters(self, sample_llm_config):
        """测试自定义参数初始化"""
        encoder = FractalCompressor(
            ratio=0.5, base_threshold=500, max_levels=8, llm_config=sample_llm_config
        )
        assert encoder.ratio == 0.5
        assert encoder.base_threshold == 500
        assert encoder.max_levels == 8
        assert encoder.llm_config == sample_llm_config

    def test_get_threshold(self, encoder):
        """测试层级门限计算"""
        assert encoder.get_threshold(0) == 100
        assert encoder.get_threshold(1) == int(100 * 0.618)
        assert encoder.get_threshold(2) == int(100 * 0.618 * 0.618)

    def test_create_empty_fractal(self, encoder):
        """测试创建空分形结构"""
        fractal = encoder.create_empty_fractal()
        assert len(fractal) == 5
        assert all(level == "" for level in fractal)


class TestFractalCompressorAddToLevel:
    """测试 _add_to_level 方法"""

    @pytest.fixture
    def encoder(self) -> FractalCompressor:
        """创建测试用的编码器"""
        return FractalCompressor(max_levels=3)

    def test_add_to_empty_level(self, encoder):
        """测试添加到空层级"""
        fractal_text = [""]
        encoder._add_to_level(fractal_text, "新文本", 0)
        assert fractal_text[0] == "新文本"

    def test_add_to_existing_level_0(self, encoder):
        """测试添加到已有内容的Level 0（应该直接连接）"""
        fractal_text = ["现有文本"]
        encoder._add_to_level(fractal_text, "新文本", 0)
        assert fractal_text[0] == "现有文本新文本"

    def test_add_to_existing_level_1_plus(self, encoder):
        """测试添加到已有内容的Level 1+（应该用换行分隔）"""
        fractal_text = ["", "现有文本"]
        encoder._add_to_level(fractal_text, "新文本", 1)
        assert fractal_text[1] == "现有文本\n新文本"

    def test_add_to_new_level(self, encoder):
        """测试添加到新层级（自动扩展）"""
        fractal_text = ["level0"]
        encoder._add_to_level(fractal_text, "level2文本", 2)
        assert len(fractal_text) == 3
        assert fractal_text[0] == "level0"
        assert fractal_text[1] == ""
        assert fractal_text[2] == "level2文本"


class TestFractalCompressorEncoding:
    """测试编码功能"""

    @pytest.fixture
    def encoder_with_api(self) -> FractalCompressor:
        """创建带有真实API配置的编码器"""
        llm_config = {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "temperature": 0.1,
        }

        # 如果没有API key，跳过需要LLM的测试
        if not llm_config["api_key"]:
            pytest.skip("需要 OPENAI_API_KEY 环境变量")

        return FractalCompressor(
            ratio=0.618,
            base_threshold=50,  # 小门限便于测试
            max_levels=3,
            llm_config=llm_config,
        )

    def test_encode_short_text(self, encoder_with_api):
        """测试编码短文本（不触发压缩）"""
        fractal_text = [""]
        short_text = "短文本测试"
        result = encoder_with_api.compress(fractal_text, short_text)

        assert len(result) == 3  # max_levels
        assert result[0] == "短文本测试"
        assert all(result[i] == "" for i in range(1, 3))

    def test_encode_empty_text(self, encoder_with_api):
        """测试编码空文本"""
        fractal_text = [""]
        result = encoder_with_api.compress(fractal_text, "")
        assert result[0] == ""

    def test_encode_to_existing_fractal(self, encoder_with_api):
        """测试编码到已有分形结构"""
        fractal_text = ["现有内容"]
        result = encoder_with_api.compress(fractal_text, "新内容")

        # Level 0 应该直接连接
        assert "现有内容" in result[0]
        assert "新内容" in result[0]
        assert result[0] == "现有内容新内容"

    @pytest.mark.integration
    def test_encode_long_text_with_compression(self, encoder_with_api):
        """测试长文本编码（触发LLM压缩）"""
        fractal_text = [""]
        # 创建超过门限的长文本
        long_text = """机器人技术正在从工业制造向服务领域扩展。现代机器人配备了先进的传感器和人工智能算法，
能够在复杂环境中自主导航和执行任务。协作机器人(cobot)可以与人类安全地共同工作，在医疗、教育和
家庭服务中发挥重要作用。自主移动机器人正在仓储物流、清洁维护和安全巡逻等领域展现价值。
随着技术的不断进步，机器人将在更多场景中协助人类完成复杂的工作。"""

        result = encoder_with_api.compress(fractal_text, long_text)

        # 验证结果结构
        assert len(result) >= 1
        assert len(result[0]) > 0  # Level 0 应该有内容

        # 如果文本足够长，应该触发多层级
        total_length = sum(len(level) for level in result if level)
        original_length = len(long_text)

        # 压缩后的总长度应该小于或接近原始长度
        assert total_length <= original_length + 100  # 允许一些误差

    def test_compress_single_text(self, encoder_with_api):
        """测试单文本压缩便捷方法"""
        text = "这是一段测试文本，用来验证分形编码的基本功能。"
        result = encoder_with_api.compress_single_text(text)

        assert len(result) == 3  # max_levels
        assert isinstance(result, list)
        assert all(isinstance(level, str) for level in result)
        assert result[0]  # Level 0 应该有内容


class TestFractalCompressorEdgeCases:
    """测试边界情况"""

    def test_very_small_threshold(self):
        """测试极小门限"""
        llm_config = {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "temperature": 0.1,
        }

        if not llm_config["api_key"]:
            pytest.skip("需要 OPENAI_API_KEY 环境变量")

        encoder = FractalCompressor(
            ratio=0.5, base_threshold=5, max_levels=2, llm_config=llm_config  # 极小门限
        )

        fractal_text = [""]
        text = "长文本测试"
        result = encoder.compress(fractal_text, text)

        # 应该能正常处理
        assert isinstance(result, list)
        assert len(result) == 2

    def test_exact_threshold_length(self):
        """测试刚好达到门限的文本"""
        encoder = FractalCompressor(base_threshold=5)
        fractal_text = [""]
        text = "12345"  # 门限为5

        result = encoder.compress(fractal_text, text)
        assert result[0] == "12345"

    def test_minimum_over_boundary(self):
        """测试边界情况处理"""
        encoder = FractalCompressor(base_threshold=5)
        fractal_text = ["1234"]  # 4个字符，门限5
        text = "56"  # 2个字符

        result = encoder.compress(fractal_text, text)
        # 验证结果基本正确 - 如果无法容纳，可能会压缩或分层
        assert isinstance(result, list)
        assert len(result) > 0
        # 至少应该包含原有内容的长度
        assert len(result[0]) >= 2  # 新文本至少被保留


class TestConvenienceFunctions:
    """测试便捷函数"""

    def test_create_fractal_compressor(self):
        """测试创建编码器便捷函数"""
        llm_config = {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "test-key",
        }

        encoder = create_fractal_compressor(
            ratio=0.7, base_threshold=500, llm_config=llm_config
        )

        assert isinstance(encoder, FractalCompressor)
        assert encoder.ratio == 0.7
        assert encoder.base_threshold == 500
        assert encoder.llm_config == llm_config

    @pytest.mark.integration
    def test_encode_text_fractal(self):
        """测试一站式编码便捷函数"""
        llm_config = {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "temperature": 0.1,
        }

        if not llm_config["api_key"]:
            pytest.skip("需要 OPENAI_API_KEY 环境变量")

        text = "这是测试文本内容"
        result = encode_text_fractal(text, llm_config=llm_config)

        assert isinstance(result, list)
        assert len(result) >= 1
        assert result[0]  # Level 0 应该有内容


@pytest.mark.integration
class TestRealLLMIntegration:
    """测试真实LLM集成"""

    @pytest.fixture
    def real_encoder(self):
        """创建使用真实LLM的编码器"""
        llm_config = {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "url": "https://api.openai.com/v1",
            "temperature": 0.1,
            "max_tokens": 2000,
        }

        if not llm_config["api_key"]:
            pytest.skip("需要 OPENAI_API_KEY 环境变量进行集成测试")

        return FractalCompressor(
            ratio=0.618, base_threshold=80, max_levels=4, llm_config=llm_config
        )

    def test_real_llm_compression(self, real_encoder):
        """测试真实LLM压缩功能"""
        # 创建足够长的文本触发压缩
        long_text = """
        人工智能技术的发展正在深刻改变我们的生活方式和工作模式。
        从自然语言处理到计算机视觉，从机器学习到深度学习，
        AI技术在各个领域都展现出了巨大的潜力和价值。
        特别是在医疗诊断、自动驾驶、金融风控等关键应用场景中，
        AI技术的应用不仅提高了效率，也为解决复杂问题提供了新的思路和方法。
        """

        result = real_encoder.compress_single_text(long_text.strip())

        # 验证压缩效果
        assert len(result) >= 2  # 应该有多个层级
        assert len(result[0]) > 0  # Level 0 有内容

        # 验证总长度
        total_compressed_length = sum(len(level) for level in result if level)
        original_length = len(long_text.strip())

        # 压缩应该是有效的
        assert total_compressed_length < original_length

        print(f"原始长度: {original_length}")
        print(f"压缩后长度: {total_compressed_length}")
        print(f"压缩比: {total_compressed_length/original_length:.2%}")
        print(f"Level 0: {result[0]}")
        if result[1]:
            print(f"Level 1: {result[1]}")
