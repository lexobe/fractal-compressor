"""
Test Suite for Fractal Compress API

Clean, comprehensive tests for the new API.
"""

import os
import sys

import pytest

from fractal_compressor import compress, llm_compress, split, split_compress

# 使用标准的src layout导入
# Add src to path


class TestBasicAPI:
    """Test basic API functionality."""

    @pytest.fixture
    def sample_texts(self):
        """Sample texts for testing."""
        return {
            "chinese": "人工智能技术正在快速发展，机器学习和深度学习领域取得了重大突破。",
            "english": "Artificial intelligence technology is rapidly advancing with breakthrough achievements.",
            "mixed": "AI人工智能technology正在rapidly发展，bringing革命性changes。",
            "short": "短文本",
            "empty": "",
            "long": """人工智能正在改变我们的世界。从自动驾驶汽车到智能医疗诊断，AI技术的应用越来越广泛。
机器学习算法能够从大量数据中学习模式，并做出准确的预测。深度学习网络模仿人脑神经元的工作方式，
在图像识别、语音处理和自然语言理解方面取得了突破性进展。随着计算能力的提升和数据的积累，
我们正站在一个技术革命的关键时刻，AI将继续塑造人类社会的未来。""",
        }

    def test_compress_basic(self, sample_texts):
        """Test basic compression functionality."""
        # Test with different texts
        for key, text in sample_texts.items():
            if key == "empty":
                assert compress(text) == ""
            else:
                result = compress(text)
                assert isinstance(result, str)
                if text:
                    assert len(result) > 0
                    assert len(result) <= len(text)

    def test_compress_custom_ratios(self, sample_texts):
        """Test compression with custom ratios."""
        text = sample_texts["chinese"]

        # Test different split ratios
        result1 = compress(text, split_ratio=0.3)
        result2 = compress(text, split_ratio=0.7)

        assert isinstance(result1, str)
        assert isinstance(result2, str)
        assert len(result1) > 0
        assert len(result2) > 0

        # Test different compression ratios
        result3 = compress(text, compression_ratio=0.3)
        result4 = compress(text, compression_ratio=0.8)

        # Lower compression ratio should produce shorter result
        assert len(result3) <= len(result4)

    def test_split_compress(self, sample_texts):
        """Test split_compress functionality."""
        text = sample_texts["chinese"]

        compressed, remaining = split_compress(text)

        assert isinstance(compressed, str)
        assert isinstance(remaining, str)
        assert len(compressed) > 0
        assert len(remaining) > 0
        assert len(compressed) + len(remaining) < len(text)  # Some compression occurred

    def test_llm_compress(self, sample_texts):
        """Test direct LLM compression."""
        text = sample_texts["english"]

        # Test with specific target length
        result = llm_compress(text, target_length=20)
        assert isinstance(result, str)
        assert len(result) <= 25  # Allow small deviation
        assert len(result) > 0

        # Test with zero target
        result_zero = llm_compress(text, target_length=0)
        assert result_zero == ""

    def test_split(self, sample_texts):
        """Test text splitting functionality."""
        text = sample_texts["chinese"]

        part1, part2 = split(text)

        assert isinstance(part1, str)
        assert isinstance(part2, str)
        assert len(part1) > 0
        assert len(part2) > 0
        assert len(part1) + len(part2) == len(text)  # Lossless splitting

    def test_language_parameters(self, sample_texts):
        """Test language parameter handling."""
        text = sample_texts["mixed"]

        # Test different language settings
        result_auto = compress(text, language="auto")
        result_chinese = compress(text, language="chinese")
        result_english = compress(text, language="english")

        assert isinstance(result_auto, str)
        assert isinstance(result_chinese, str)
        assert isinstance(result_english, str)

        # All should produce valid results
        assert len(result_auto) > 0
        assert len(result_chinese) > 0
        assert len(result_english) > 0

    def test_edge_cases(self, sample_texts):
        """Test edge cases and error conditions."""
        # Empty text
        assert compress("") == ""
        assert llm_compress("", 10) == ""
        assert split("") == ("", "")
        assert split_compress("") == ("", "")

        # Very short text
        short_result = compress(sample_texts["short"])
        assert isinstance(short_result, str)

        # Invalid split ratio
        with pytest.raises(ValueError):
            split("test", ratio=-0.1)

        with pytest.raises(ValueError):
            split("test", ratio=1.1)


class TestAdvancedFeatures:
    """Test advanced features and parameters."""

    def test_compression_strategies(self):
        """Test different LLM compression strategies."""
        text = "Machine learning algorithms are transforming data analysis capabilities worldwide."

        strategies = ["precise", "creative", "fast"]

        for strategy in strategies:
            result = llm_compress(text, target_length=25, strategy=strategy)
            assert isinstance(result, str)
            assert len(result) > 0
            assert len(result) <= 30  # Allow deviation

    def test_model_parameters(self):
        """Test different model parameters."""
        text = "人工智能技术正在改变世界"

        # Test with different models (will use default if not available)
        result = compress(text, model="gpt-4o-mini")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_ratio_combinations(self):
        """Test various ratio combinations."""
        text = """区块链技术作为一种分布式账本技术，正在革命性地改变金融、供应链管理和数字身份验证等领域。
它通过密码学哈希和共识机制确保数据的不可篡改性和透明度。每个区块包含前一个区块的哈希值，
形成一个时间序列的链式结构，这使得历史记录几乎不可能被恶意修改。"""

        combinations = [(0.2, 0.3), (0.4, 0.5), (0.6, 0.7), (0.8, 0.9)]

        for split_r, comp_r in combinations:
            result = compress(text, split_ratio=split_r, compression_ratio=comp_r)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_multilingual_texts(self):
        """Test processing of different languages."""
        texts = {
            "pure_chinese": "中文文本处理测试内容",
            "pure_english": "English text processing test content",
            "mixed_content": "Mixed中文and英文content测试",
            "with_numbers": "包含123数字和symbols!@#的文本",
            "with_punctuation": "标点符号？！，。；：的处理测试",
        }

        for lang_type, text in texts.items():
            result = compress(text)
            assert isinstance(result, str)
            assert len(result) > 0


class TestPerformance:
    """Test performance characteristics."""

    def test_length_consistency(self):
        """Test that results are consistent with length constraints."""
        text = """云计算已经成为现代数字基础设施的重要支柱。通过虚拟化技术，云服务提供商能够高效地分配计算资源，
让企业和个人用户按需使用服务器、存储和网络资源。云计算的三种主要模式包括基础设施即服务(IaaS)、
平台即服务(PaaS)和软件即服务(SaaS)。这种模式不仅降低了IT成本，还提高了系统的可扩展性和可靠性。
许多企业正在将传统的本地部署系统迁移到云端，以获得更好的灵活性和性能。边缘计算作为云计算的重要补充，
正在解决延迟敏感应用的需求。随着5G网络的普及，边缘节点能够将计算和存储资源更靠近用户，
显著降低响应时间并改善用户体验。"""

        # Multiple runs should produce consistent lengths
        results = []
        for _ in range(3):
            result = llm_compress(text, target_length=30)
            results.append(len(result))

        # All results should respect length constraint
        for length in results:
            assert length <= 35  # Allow small deviation

    def test_compression_efficiency(self):
        """Test compression efficiency."""
        texts = [
            "量子计算技术发展的简介",
            "物联网技术正在连接世界各地的设备，形成智能生态系统",
            """生物信息学是计算机科学与生物学的交叉领域，它利用算法和数据分析方法来理解复杂的生物数据。
基因组测序技术的快速发展产生了海量的DNA序列数据，需要强大的计算工具来进行比对、注释和功能预测。
蛋白质结构预测是生物信息学的一个重要分支，通过计算模型预测蛋白质的三维结构对于药物设计具有重要意义。"""
        ]

        for text in texts:
            result = compress(text)
            compression_ratio = len(result) / len(text)

            # Should achieve some compression
            assert compression_ratio < 0.2
            # But not over-compress
            assert compression_ratio > 0.05


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_pipeline_workflow(self):
        """Test complete processing pipeline."""
        # Simulate document processing pipeline
        document = """
        数据科学是一个跨学科领域，结合了统计学、计算机科学和领域专业知识。
        数据科学家使用各种工具和技术来从数据中提取见解。这个过程包括
        数据收集、清理、分析和结果解释等步骤。
        """

        # Step 1: Split document
        part1, part2 = split(document, ratio=0.6)
        assert len(part1) > 0
        assert len(part2) > 0

        # Step 2: Process parts differently
        summary = llm_compress(part1, target_length=30)
        compressed = compress(part2, compression_ratio=0.5)

        # Step 3: Verify results
        assert isinstance(summary, str)
        assert isinstance(compressed, str)
        assert len(summary) <= 35
        assert len(compressed) < len(part2)

    def test_batch_processing(self):
        """Test batch processing of multiple texts."""
        texts = [
            "文本1：人工智能发展迅速。",
            "Text 2: Machine learning advances.",
            "文本3：Mixed中英文content处理。",
        ]

        results = []
        for text in texts:
            result = compress(text, split_ratio=0.4, compression_ratio=0.6)
            results.append(result)

        # All results should be valid
        assert len(results) == len(texts)
        for result in results:
            assert isinstance(result, str)
            assert len(result) > 0


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
