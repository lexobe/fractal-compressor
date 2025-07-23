#!/usr/bin/env python3
"""
测试LLM智能整合功能的单元测试

测试新重构的核心功能：
1. _llm_integrate_and_add方法的基本功能
2. 长度控制的准确性
3. fallback机制的有效性
4. 与现有分形压缩流程的兼容性
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fractal_compressor import FractalCompressor


class TestLLMIntegration:
    """测试LLM智能整合功能"""
    
    def setup_method(self):
        """测试前的设置"""
        self.llm_config = {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "test-key",
            "temperature": 0.1
        }
        self.compressor = FractalCompressor(
            ratio=0.618,
            base_threshold=100,
            max_levels=5,
            llm_config=self.llm_config
        )

    def test_llm_integrate_and_add_empty_existing(self):
        """测试空现有内容的情况"""
        existing_content = ""
        new_text = "这是新添加的文本内容"
        
        result = self.compressor._llm_integrate_and_add(existing_content, new_text)
        
        # 空内容时应该直接返回新文本
        assert result == new_text
        assert len(result) == len(new_text)

    def test_llm_integrate_and_add_whitespace_existing(self):
        """测试只有空白字符的现有内容"""
        existing_content = "   \n  \t  "
        new_text = "新文本内容"
        
        result = self.compressor._llm_integrate_and_add(existing_content, new_text)
        
        # 只有空白字符时也应该直接返回新文本
        assert result == new_text

    @patch('fractal_compressor.fractal_compressor_class.LLMTextCompressor')
    def test_llm_integrate_and_add_with_existing_content(self, mock_compressor_class):
        """测试有现有内容时的整合功能"""
        # 设置mock
        mock_compressor = Mock()
        mock_compressor_class.return_value = mock_compressor
        mock_compressor.compress.return_value = {
            "text": "整合后的内容：原有内容与新内容的智能组合",
            "success": True
        }
        
        existing_content = "这是现有的内容，包含重要信息"
        new_text = "这是新添加的内容，也很重要"
        expected_length = len(existing_content) + int(len(new_text) * 0.618)
        
        result = self.compressor._llm_integrate_and_add(existing_content, new_text)
        
        # 验证LLM被正确调用
        mock_compressor.compress.assert_called_once()
        call_args = mock_compressor.compress.call_args
        
        # 验证参数
        assert call_args[1]['target_length'] == expected_length
        assert call_args[1]['llm_config'] == self.llm_config
        assert call_args[1]['strategy'] == "precise"
        assert call_args[1]['strict_length'] is True
        
        # 验证返回结果
        assert result == "整合后的内容：原有内容与新内容的智能组合"

    def test_length_calculation_accuracy(self):
        """测试长度计算的准确性"""
        existing_content = "A" * 100  # 100字符
        new_text = "B" * 50           # 50字符
        
        # 预期长度 = 100 + int(50 * 0.618) = 100 + 30 = 130
        expected_length = 100 + int(50 * 0.618)
        
        # 通过检查内部调用来验证长度计算
        with patch.object(self.compressor, '_call_llm_integration') as mock_call:
            mock_call.return_value = "A" * expected_length
            
            result = self.compressor._llm_integrate_and_add(existing_content, new_text)
            
            # 验证传递给LLM的目标长度
            mock_call.assert_called_once()
            call_args = mock_call.call_args[0]  # 位置参数
            assert call_args[1] == expected_length  # target_length参数

    @patch('fractal_compressor.fractal_compressor_class.LLMTextCompressor')
    def test_llm_integration_failure_fallback(self, mock_compressor_class):
        """测试LLM整合失败时的fallback机制"""
        # 设置LLM调用失败
        mock_compressor = Mock()
        mock_compressor_class.return_value = mock_compressor
        mock_compressor.compress.side_effect = Exception("API调用失败")
        
        existing_content = "现有内容"
        new_text = "新内容"
        
        with patch.object(self.compressor, '_fallback_integration') as mock_fallback:
            mock_fallback.return_value = "fallback结果"
            
            result = self.compressor._llm_integrate_and_add(existing_content, new_text)
            
            # 验证fallback被调用
            mock_fallback.assert_called_once()
            assert result == "fallback结果"

    def test_fallback_integration_within_limit(self):
        """测试fallback整合：组合文本在限制内"""
        combined_text = "组合文本内容"
        target_length = 20
        
        result = self.compressor._fallback_integration(combined_text, target_length)
        
        # 长度在限制内时直接返回
        assert result == combined_text
        assert len(result) <= target_length

    def test_fallback_integration_exceeds_limit(self):
        """测试fallback整合：组合文本超出限制"""
        combined_text = "这是一个很长的组合文本内容，超出了目标长度限制。"
        target_length = 10
        
        with patch.object(self.compressor, '_smart_truncate_for_integration') as mock_truncate:
            mock_truncate.return_value = "截断结果"
            
            result = self.compressor._fallback_integration(combined_text, target_length)
            
            # 验证智能截断被调用
            mock_truncate.assert_called_once_with(combined_text, target_length)
            assert result == "截断结果"

    def test_smart_truncate_for_integration(self):
        """测试智能截断功能"""
        # 测试句子边界截断
        text = "第一句话。第二句话！第三句话？还有更多内容。"
        max_length = 15
        
        result = self.compressor._smart_truncate_for_integration(text, max_length)
        
        # 应该在句号处截断
        assert result.endswith(('。', '！', '？'))
        assert len(result) <= max_length

    def test_smart_truncate_no_sentence_boundary(self):
        """测试没有句子边界时的截断"""
        text = "这是一段没有句号的长文本，只有逗号，分号；但是没有句子结束符号"
        max_length = 20
        
        result = self.compressor._smart_truncate_for_integration(text, max_length)
        
        # 应该在逗号或分号处截断，或直接截断
        assert len(result) <= max_length

    def test_smart_truncate_within_limit(self):
        """测试文本在限制内时不截断"""
        text = "短文本"
        max_length = 20
        
        result = self.compressor._smart_truncate_for_integration(text, max_length)
        
        assert result == text

    @patch('fractal_compressor.fractal_compressor_class.LLMTextCompressor')
    def test_integration_with_recursive_compress(self, mock_compressor_class):
        """测试整合功能与递归压缩的集成"""
        # 设置mock LLM返回适当长度的内容
        mock_compressor = Mock()
        mock_compressor_class.return_value = mock_compressor
        
        # 第一次整合调用 - 不超限
        mock_compressor.compress.return_value = {
            "text": "A" * 50,  # 50字符，不超限
            "success": True
        }
        
        fractal_text = [""] * 5
        new_text = "B" * 30
        
        result = self.compressor._recursive_compress(fractal_text, new_text, 0)
        
        # 验证整合被调用且结果正确设置
        assert len(result[0]) == 50
        mock_compressor.compress.assert_called()

    def test_integration_parameters_consistency(self):
        """测试整合参数的一致性"""
        existing_content = "现有内容" * 10  # 40字符
        new_text = "新内容" * 5           # 15字符
        
        with patch.object(self.compressor, '_call_llm_integration') as mock_call:
            mock_call.return_value = "整合结果"
            
            self.compressor._llm_integrate_and_add(existing_content, new_text)
            
            # 验证参数传递的一致性
            args = mock_call.call_args[0]
            combined_text, target_length, existing_length, new_length = args
            
            assert existing_length == len(existing_content)
            assert new_length == len(new_text)
            assert target_length == len(existing_content) + int(len(new_text) * 0.618)
            assert combined_text == existing_content + "\n" + new_text

    def test_multiple_integrations_length_control(self):
        """测试多次整合的长度控制"""
        with patch.object(self.compressor, '_call_llm_integration') as mock_call:
            # 设置返回值模拟实际整合效果
            def side_effect(combined_text, target_length, *args):
                return "整合" * (target_length // 2)  # 返回目标长度的文本
            
            mock_call.side_effect = side_effect
            
            # 第一次整合
            content1 = self.compressor._llm_integrate_and_add("", "第一段文本" * 5)
            
            # 第二次整合
            content2 = self.compressor._llm_integrate_and_add(content1, "第二段文本" * 3)
            
            # 第三次整合
            content3 = self.compressor._llm_integrate_and_add(content2, "第三段文本" * 2)
            
            # 验证每次整合都控制了长度增长
            assert len(content2) >= len(content1)  # 有增长
            assert len(content3) >= len(content2)  # 继续增长
            
            # 验证增长是受控的（不是简单拼接）
            simple_concat_length = len("第一段文本" * 5) + len("第二段文本" * 3) + len("第三段文本" * 2)
            assert len(content3) < simple_concat_length  # 应该比简单拼接短


if __name__ == "__main__":
    pytest.main([__file__, "-v"])