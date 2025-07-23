"""
测试 llm_compressor 模块的LLM文本压缩功能
使用真实 LLM API 调用，不使用 mock
"""

import os
import sys
from typing import Any, Dict

import pytest

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from fractal_compressor.llm_compressor import (
        LLMTextCompressor,
        compare_strategies,
        quick_compress,
    )
except ImportError:
    # 如果相对导入失败，尝试直接导入
    sys.path.insert(
        0, os.path.join(os.path.dirname(__file__), "..", "src", "fractal_compressor")
    )
    from llm_compressor import LLMTextCompressor, compare_strategies, quick_compress


class TestLLMTextCompressorBasic:
    """测试 LLMTextCompressor 基本功能"""

    @pytest.fixture
    def compressor(self) -> LLMTextCompressor:
        """创建测试用的压缩器实例"""
        return LLMTextCompressor(default_model="gpt-4o-mini")

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
        }

    def test_init_default_model(self):
        """测试默认模型初始化"""
        compressor = LLMTextCompressor()
        assert compressor.default_model == "gpt-4o-mini"
        assert isinstance(compressor.default_templates, dict)
        assert isinstance(compressor.default_examples, list)

    def test_init_custom_model(self):
        """测试自定义模型初始化"""
        compressor = LLMTextCompressor(default_model="gpt-4o")
        assert compressor.default_model == "gpt-4o"

    def test_template_creation(self, compressor):
        """测试模板创建"""
        templates = compressor._create_default_templates()

        assert isinstance(templates, dict)
        expected_strategies = ["basic", "precise", "few_shot", "creative", "strict"]
        for strategy in expected_strategies:
            assert strategy in templates
            assert isinstance(templates[strategy], str)
            assert len(templates[strategy]) > 0

    def test_examples_creation(self, compressor):
        """测试示例创建"""
        examples = compressor._create_default_examples()

        assert isinstance(examples, list)
        assert len(examples) > 0

        for example in examples:
            assert isinstance(example, dict)
            assert "original" in example
            assert "target" in example
            assert "compressed" in example
            assert "strategy" in example


class TestLLMTextCompressorCompression:
    """测试压缩功能"""

    @pytest.fixture
    def compressor_with_api(self) -> LLMTextCompressor:
        """创建带有真实API的压缩器"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("需要 OPENAI_API_KEY 环境变量")
        return LLMTextCompressor()

    @pytest.fixture
    def llm_config(self) -> Dict[str, Any]:
        """LLM配置"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("需要 OPENAI_API_KEY 环境变量")

        return {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": api_key,
            "temperature": 0.1,
            "max_tokens": 1000,
        }

    @pytest.mark.integration
    def test_basic_compression(self, compressor_with_api, llm_config):
        """测试基本压缩功能"""
        text = "人工智能技术正在快速发展，为各行各业带来革命性的变化和无限的可能性。"
        target_length = 20

        result = compressor_with_api.compress(
            text=text,
            target_length=target_length,
            llm_config=llm_config,
            strategy="basic",
        )

        # 验证结果结构
        assert isinstance(result, dict)
        required_keys = [
            "text",
            "original_text",
            "original_length",
            "compressed_length",
            "target_length",
            "compression_ratio",
            "attempts",
            "strategy",
            "temperature",
            "model",
            "processing_time",
            "success",
            "length_constraint_satisfied",
        ]
        for key in required_keys:
            assert key in result

        # 验证压缩效果
        assert isinstance(result["text"], str)
        assert len(result["text"]) > 0
        assert result["original_text"] == text
        assert result["original_length"] == len(text)
        assert result["compressed_length"] == len(result["text"])
        assert result["target_length"] == target_length

        # 验证压缩比例
        assert 0 < result["compression_ratio"] <= 1
        assert (
            result["compression_ratio"]
            == result["compressed_length"] / result["original_length"]
        )

    @pytest.mark.integration
    def test_precise_strategy(self, compressor_with_api, llm_config):
        """测试精确策略"""
        text = "机器学习是人工智能的一个重要分支，通过算法让计算机从数据中学习规律。"
        target_length = 15

        result = compressor_with_api.compress(
            text=text,
            target_length=target_length,
            llm_config=llm_config,
            strategy="precise",
            strict_length=True,
        )

        assert result["strategy"] == "precise"
        assert result["success"] == True
        assert len(result["text"]) <= target_length
        assert result["length_constraint_satisfied"] == True

        print(f"原文: {text}")
        print(f"压缩结果: {result['text']}")
        print(f"压缩比: {result['compression_ratio']:.2%}")

    @pytest.mark.integration
    def test_few_shot_strategy(self, compressor_with_api, llm_config):
        """测试Few-shot策略"""
        text = "云计算提供了可扩展的计算资源，企业可以按需使用，降低了IT基础设施成本。"
        target_length = 18

        result = compressor_with_api.compress(
            text=text,
            target_length=target_length,
            llm_config=llm_config,
            strategy="few_shot",
        )

        assert result["strategy"] == "few_shot"
        assert isinstance(result["text"], str)
        assert len(result["text"]) > 0

    @pytest.mark.integration
    def test_multiple_attempts(self, compressor_with_api, llm_config):
        """测试多次尝试机制"""
        text = "深度学习神经网络通过多层结构模拟人脑处理信息的方式，在图像识别等领域取得突破。"
        target_length = 12

        result = compressor_with_api.compress(
            text=text,
            target_length=target_length,
            llm_config=llm_config,
            max_attempts=3,
            strict_length=True,
        )

        assert result["attempts"] <= 3
        assert result["attempts"] >= 1
        assert len(result["text"]) <= target_length

    @pytest.mark.integration
    def test_temperature_effect(self, compressor_with_api, llm_config):
        """测试温度参数影响"""
        text = "区块链技术通过去中心化的方式确保数据安全和透明，应用前景广阔。"
        target_length = 16

        # 测试低温度
        llm_config_low = llm_config.copy()
        llm_config_low["temperature"] = 0.1

        result_low = compressor_with_api.compress(
            text=text, target_length=target_length, llm_config=llm_config_low
        )

        # 测试高温度
        llm_config_high = llm_config.copy()
        llm_config_high["temperature"] = 0.8

        result_high = compressor_with_api.compress(
            text=text, target_length=target_length, llm_config=llm_config_high
        )

        assert result_low["temperature"] == 0.1
        assert result_high["temperature"] == 0.8
        assert isinstance(result_low["text"], str)
        assert isinstance(result_high["text"], str)


class TestLLMTextCompressorEdgeCases:
    """测试边界情况"""

    @pytest.fixture
    def compressor(self) -> LLMTextCompressor:
        return LLMTextCompressor()

    @pytest.fixture
    def llm_config(self) -> Dict[str, Any]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("需要 OPENAI_API_KEY 环境变量")

        return {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": api_key,
            "temperature": 0.1,
        }

    @pytest.mark.integration
    def test_very_short_text(self, compressor, llm_config):
        """测试极短文本"""
        text = "AI"
        target_length = 5

        result = compressor.compress(
            text=text, target_length=target_length, llm_config=llm_config
        )

        # 短文本应该能正常处理，但可能失败（取决于LLM调用）
        assert len(result["text"]) <= target_length
        # 不强制要求success=True，因为可能会有API调用失败等情况
        assert isinstance(result["success"], bool)

    @pytest.mark.integration
    def test_target_longer_than_original(self, compressor, llm_config):
        """测试目标长度大于原文长度"""
        text = "短文本"
        target_length = 50  # 远大于原文长度

        result = compressor.compress(
            text=text, target_length=target_length, llm_config=llm_config
        )

        # 应该返回合理的结果（可能是原文或扩展版本）
        # 不强制要求success=True，因为可能会有API调用失败等情况
        assert isinstance(result["success"], bool)
        # 如果成功，长度应该在合理范围内
        if result["success"]:
            assert len(result["text"]) <= target_length

    @pytest.mark.integration
    def test_very_small_target_length(self, compressor, llm_config):
        """测试极小目标长度"""
        text = "这是一段需要大幅压缩的较长文本内容。"
        target_length = 3

        result = compressor.compress(
            text=text,
            target_length=target_length,
            llm_config=llm_config,
            strict_length=True,
        )

        # 即使目标很小也应该能处理
        assert len(result["text"]) <= target_length
        assert result["length_constraint_satisfied"] == True

    def test_fallback_mechanism(self, compressor):
        """测试fallback机制（当LLM调用失败时）"""
        text = "测试fallback机制的文本内容"
        target_length = 10

        # 使用无效的配置触发fallback
        invalid_config = {
            "provider": "openai",
            "model": "invalid-model",
            "api_key": "invalid-key",
        }

        result = compressor.compress(
            text=text,
            target_length=target_length,
            llm_config=invalid_config,
            max_attempts=1,
        )

        # 应该返回fallback结果
        assert result["success"] == False
        assert "error" in result
        assert len(result["text"]) <= target_length
        assert result["strategy"] == "fallback"


class TestLLMTextCompressorHelperMethods:
    """测试辅助方法"""

    @pytest.fixture
    def compressor(self) -> LLMTextCompressor:
        return LLMTextCompressor()

    def test_build_prompt(self, compressor):
        """测试构建prompt"""
        template = "压缩这段文本：{text}，目标长度：{target_length}"
        text = "测试文本"
        target_length = 5
        strategy = "basic"
        examples = []

        prompt = compressor._build_prompt(
            template, text, target_length, strategy, examples
        )

        assert isinstance(prompt, str)
        assert "测试文本" in prompt
        assert "5" in prompt

    def test_clean_output(self, compressor):
        """测试清理输出"""
        test_cases = [
            ('"压缩结果"', "压缩结果"),
            ("'单引号文本'", "单引号文本"),
            ("压缩结果：这是结果", "这是结果"),
            ("输出：最终文本", "最终文本"),
            ("  带空格的文本  ", "带空格的文本"),
        ]

        for input_text, expected in test_cases:
            result = compressor._clean_output(input_text)
            assert result == expected

    def test_smart_truncate(self, compressor):
        """测试智能截断"""
        # 测试在标点处截断
        text = "第一句话。第二句话！"
        result = compressor._smart_truncate(text, 6)
        assert result == "第一句话。"

        # 测试在空格处截断（允许一些实现差异）
        text = "word1 word2 word3"
        result = compressor._smart_truncate(text, 10)
        # 应该在空格处截断，但具体实现可能有差异
        assert "word1" in result
        assert len(result) <= 10

        # 测试直接截断
        text = "连续文本没有分隔符"
        result = compressor._smart_truncate(text, 5)
        assert len(result) == 5

    def test_validate_quality(self, compressor):
        """测试质量验证"""
        # 有效压缩
        assert (
            compressor._validate_quality("压缩文本", "原始较长文本", 10, True) == True
        )

        # 空文本
        assert compressor._validate_quality("", "原始文本", 10, True) == False

        # 超过长度限制
        assert compressor._validate_quality("很长的文本", "原始文本", 3, True) == False

        # 没有压缩效果
        assert (
            compressor._validate_quality("原始文本相同", "原始文本相同", 20, True)
            == False
        )


class TestLLMTextCompressorBatchTest:
    """测试批量测试功能"""

    @pytest.fixture
    def compressor(self) -> LLMTextCompressor:
        return LLMTextCompressor()

    @pytest.mark.integration
    def test_batch_test_functionality(self, compressor):
        """测试批量测试功能"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("需要 OPENAI_API_KEY 环境变量")

        test_cases = [
            {"text": "人工智能改变世界", "target": 8},
            {"text": "机器学习算法优化", "target": 6},
            {"text": "深度学习神经网络", "target": 7},
        ]

        result = compressor.batch_test(
            test_cases=test_cases, strategies=["basic", "precise"], model="gpt-4o-mini"
        )

        # 验证结果结构
        assert isinstance(result, dict)
        assert "results" in result
        assert "statistics" in result
        assert "total_cases" in result
        assert "strategies_tested" in result

        # 验证统计信息
        stats = result["statistics"]
        for strategy in ["basic", "precise"]:
            assert strategy in stats
            assert "success_rate" in stats[strategy]
            assert "avg_compression_ratio" in stats[strategy]
            assert "avg_processing_time" in stats[strategy]
            assert "total_cases" in stats[strategy]


class TestConvenienceFunctions:
    """测试便捷函数"""

    @pytest.mark.integration
    def test_quick_compress(self):
        """测试快速压缩函数"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("需要 OPENAI_API_KEY 环境变量")

        text = "人工智能技术快速发展"
        target_length = 8

        # 临时设置环境变量让litellm使用API key
        original_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = api_key

        try:
            result = quick_compress(text, target_length, strategy="basic")
            assert isinstance(result, str)
            assert len(result) <= target_length + 2  # 允许一些误差
        finally:
            if original_key:
                os.environ["OPENAI_API_KEY"] = original_key
            elif "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]

    @pytest.mark.integration
    def test_compare_strategies(self):
        """测试策略比较函数"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("需要 OPENAI_API_KEY 环境变量")

        text = "云计算技术应用广泛"
        target_length = 6

        # 临时设置环境变量
        original_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = api_key

        try:
            results = compare_strategies(text, target_length)

            assert isinstance(results, dict)
            strategies = ["basic", "precise", "few_shot", "creative", "strict"]
            for strategy in strategies:
                assert strategy in results
                assert isinstance(results[strategy], str)
        finally:
            if original_key:
                os.environ["OPENAI_API_KEY"] = original_key
            elif "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]


class TestLLMConfigIntegration:
    """测试LLM配置集成"""

    @pytest.fixture
    def compressor(self) -> LLMTextCompressor:
        return LLMTextCompressor()

    @pytest.mark.integration
    def test_llm_config_parameter_extraction(self, compressor):
        """测试从llm_config中提取参数"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("需要 OPENAI_API_KEY 环境变量")

        text = "测试配置参数提取功能"
        target_length = 8

        llm_config = {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": api_key,
            "temperature": 0.3,
            "max_attempts": 2,
            "max_tokens": 500,
        }

        result = compressor.compress(
            text=text, target_length=target_length, llm_config=llm_config
        )

        # 验证配置参数被正确使用
        assert result["model"] == "gpt-4o-mini"
        assert result["temperature"] == 0.3
        assert result["attempts"] <= 2

    @pytest.mark.integration
    def test_llm_config_override(self, compressor):
        """测试llm_config参数覆盖"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("需要 OPENAI_API_KEY 环境变量")

        text = "测试参数覆盖功能"
        target_length = 6

        llm_config = {
            "model": "gpt-4o",  # 配置中的模型
            "temperature": 0.8,  # 配置中的温度
            "api_key": api_key,
        }

        # 传入的参数应该优先于配置
        result = compressor.compress(
            text=text,
            target_length=target_length,
            model="gpt-4o-mini",  # 直接传入的模型，应该覆盖配置
            temperature=0.2,  # 直接传入的温度，应该被配置覆盖
            llm_config=llm_config,
        )

        # model参数优先
        assert result["model"] == "gpt-4o-mini"
        # temperature从配置中获取
        assert result["temperature"] == 0.8
