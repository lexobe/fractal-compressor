#!/usr/bin/env python3
"""
独立的LLM文本压缩工具
专门用于prompt优化实验和研究
"""

import logging
import time
from typing import Any, Dict, List, Optional

import litellm


class LLMTextCompressor:
    """
    独立的LLM文本压缩器

    专门设计用于prompt优化实验，提供：
    - 灵活的prompt模板系统
    - 可自定义的Few-shot示例
    - 多策略压缩尝试
    - 详细的结果分析
    """

    def __init__(self, default_model: str = "gpt-4o-mini"):
        self.default_model = default_model
        self.default_templates = self._create_default_templates()
        self.default_examples = self._create_default_examples()
        self.logger = logging.getLogger(f"{__name__}.LLMTextCompressor")

    def _create_default_templates(self) -> Dict[str, str]:
        """创建默认prompt模板"""
        return {
            "basic": """请将以下文本压缩到{target_length}字符以内：

原文：{text}
目标长度：≤{target_length}字符

压缩结果：""",
            "precise": """你是一个专业的文本压缩专家。请严格按照要求压缩文本。

📝 原文："{text}"
📏 原文长度：{original_length}字符
🎯 目标长度：≤{target_length}字符
📊 压缩比例：{compression_ratio:.1%}

要求：
1. 保持核心意思
2. 长度必须≤{target_length}字符
3. 表达简洁准确

直接输出压缩结果：""",
            "few_shot": """以下是文本压缩示例，请学习压缩技巧：

{examples}

现在请压缩以下文本：
原文："{text}" (长度: {original_length})
目标长度：≤{target_length}字符

压缩结果：""",
            "creative": """创造性文本压缩任务：

原文：{text}
挑战：用≤{target_length}字符表达相同意思
技巧：可以使用缩写、同义词、重新表述

输出：""",
            "strict": """严格长度控制压缩：

输入：{text}
输出要求：≤{target_length}字符
规则：超出长度视为失败

输出：""",
        }

    def _create_default_examples(self) -> List[Dict[str, Any]]:
        """创建默认Few-shot示例"""
        return [
            {
                "original": "人工智能技术在医疗诊断领域的应用越来越广泛",
                "target": 12,
                "compressed": "AI在医疗诊断应用广",
                "strategy": "缩写+关键词",
            },
            {
                "original": "Machine learning algorithms improve data analysis",
                "target": 20,
                "compressed": "ML algorithms enhance data analysis",
                "strategy": "缩写+同义词",
            },
            {
                "original": "Cloud computing提供scalable infrastructure solutions",
                "target": 18,
                "compressed": "云计算提供可扩展基础设施",
                "strategy": "翻译+核心概念",
            },
        ]

    def compress(
        self,
        text: str,
        target_length: int,
        model: Optional[str] = None,
        strategy: str = "precise",
        max_attempts: int = 3,
        temperature: float = 0.1,
        custom_template: Optional[str] = None,
        custom_examples: Optional[List[Dict]] = None,
        strict_length: bool = True,
        llm_config: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        压缩文本

        Args:
            text: 待压缩文本
            target_length: 目标长度
            model: LLM模型
            strategy: 压缩策略 ("basic", "precise", "few_shot", "creative", "strict")
            max_attempts: 最大尝试次数
            temperature: 温度参数
            custom_template: 自定义模板
            custom_examples: 自定义示例
            strict_length: 是否严格控制长度
            llm_config: LLM配置字典，包含provider、model、api_key等

        Returns:
            包含压缩结果和详细信息的字典
        """
        # 从llm_config中提取配置，优先使用传入的参数
        if llm_config:
            model = model or llm_config.get("model", self.default_model)
            temperature = llm_config.get("temperature", temperature)
            max_attempts = llm_config.get("max_attempts", max_attempts)
        else:
            model = model or self.default_model

        self.logger.debug(
            "开始LLM压缩: 原文长度=%d, 目标长度=%d, 策略=%s", 
            len(text), target_length, strategy
        )
        self.logger.debug(
            "压缩前文本: %s",
            text[:100] + "..." if len(text) > 100 else text
        )

        start_time = time.time()

        # 选择模板
        if custom_template:
            template = custom_template
        else:
            template = self.default_templates.get(
                strategy, self.default_templates["precise"]
            )

        # 选择示例
        examples = custom_examples or self.default_examples

        for attempt in range(max_attempts):
            self.logger.debug("尝试第%d次压缩", attempt + 1)
            try:
                # 构造prompt
                prompt = self._build_prompt(
                    template, text, target_length, strategy, examples
                )

                # 调用LLM
                response = litellm.completion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=min(800, target_length * 4),
                    timeout=30,
                )

                compressed = response.choices[0].message.content.strip()
                compressed = self._clean_output(compressed)
                
                self.logger.debug(
                    "LLM原始输出: %s",
                    compressed[:100] + "..." if len(compressed) > 100 else compressed
                )

                # 长度检查
                if strict_length and len(compressed) > target_length:
                    self.logger.debug("长度超限，进行智能分割: %d -> %d", len(compressed), target_length)
                    compressed, overflow = self._smart_split(compressed, target_length)
                    if overflow:
                        self.logger.debug("分割产生溢出内容: %s", overflow[:50] + "..." if len(overflow) > 50 else overflow)

                self.logger.debug(
                    "压缩后文本: %s",
                    compressed[:100] + "..." if len(compressed) > 100 else compressed
                )
                self.logger.debug("压缩效果: %d -> %d 字符 (%.1f%%)", 
                                len(text), len(compressed), len(compressed)/len(text)*100)

                # 质量验证
                if self._validate_quality(
                    compressed, text, target_length, strict_length
                ):
                    self.logger.debug("压缩成功，第%d次尝试", attempt + 1)
                    return {
                        "text": compressed,
                        "original_text": text,
                        "original_length": len(text),
                        "compressed_length": len(compressed),
                        "target_length": target_length,
                        "compression_ratio": (
                            len(compressed) / len(text) if len(text) > 0 else 0
                        ),
                        "attempts": attempt + 1,
                        "strategy": strategy,
                        "temperature": temperature,
                        "model": model,
                        "processing_time": time.time() - start_time,
                        "success": True,
                        "length_constraint_satisfied": len(compressed) <= target_length,
                        "prompt": prompt,  # 用于调试
                    }

            except Exception as e:
                self.logger.debug("第%d次尝试失败: %s", attempt + 1, str(e))
                if attempt == max_attempts - 1:
                    # 最后一次失败，直接抛出异常
                    self.logger.error("LLM压缩失败，已达最大尝试次数: %s", str(e))
                    raise Exception(f"LLM压缩失败: {str(e)}")
                continue

        # 所有尝试失败，抛出异常
        self.logger.error("达到最大尝试次数，LLM压缩失败")
        raise Exception("LLM压缩失败: 达到最大尝试次数")

    def _build_prompt(
        self,
        template: str,
        text: str,
        target_length: int,
        strategy: str,
        examples: List[Dict],
    ) -> str:
        """构造prompt"""
        original_length = len(text)
        compression_ratio = (
            target_length / original_length if original_length > 0 else 0
        )

        # 格式化示例（仅用于few_shot策略）
        examples_text = ""
        if strategy == "few_shot":
            examples_text = self._format_examples(examples)

        # 填充模板
        try:
            return template.format(
                text=text,
                target_length=target_length,
                original_length=original_length,
                compression_ratio=compression_ratio,
                examples=examples_text,
            )
        except KeyError:
            # 模板缺少某些变量，使用基础格式
            return (
                f"请将以下文本压缩到{target_length}字符以内：\n\n{text}\n\n压缩结果："
            )

    def _format_examples(self, examples: List[Dict]) -> str:
        """格式化示例文本"""
        formatted = ""
        for i, example in enumerate(examples[:2], 1):  # 最多2个示例
            formatted += f"""示例{i}：
原文："{example['original']}" (长度: {len(example['original'])})
目标：≤{example['target']}字符
压缩："{example['compressed']}" (长度: {len(example['compressed'])})
技巧：{example.get('strategy', '核心提取')}

"""
        return formatted

    def _clean_output(self, text: str) -> str:
        """清理LLM输出"""
        # 移除引号
        quotes = ['"', "'", '"', '"', """, """]
        for quote in quotes:
            if text.startswith(quote) and text.endswith(quote):
                text = text[1:-1]

        # 移除常见前缀
        prefixes = ["压缩结果：", "输出：", "结果：", "答案：", "压缩："]
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()

        return text.strip()

    def _smart_split(self, text: str, max_length: int) -> tuple[str, str]:
        """智能分割文本到指定长度，返回(保留部分, 溢出部分)"""
        if len(text) <= max_length:
            return text, ""

        # 尝试在句子边界分割（句号、感叹号、问号）
        sentence_punctuation = ["。", "！", "？", ".", "!", "?"]
        for i in range(max_length - 1, max(0, max_length - 20), -1):
            if i < len(text) and text[i] in sentence_punctuation:
                return text[:i + 1], text[i + 1:].strip()

        # 尝试在短语边界分割（逗号、分号）
        phrase_punctuation = ["，", ",", "；", ";", "、"]
        for i in range(max_length - 1, max(0, max_length - 15), -1):
            if i < len(text) and text[i] in phrase_punctuation:
                return text[:i + 1], text[i + 1:].strip()

        # 尝试在词汇边界分割（空格）
        for i in range(max_length - 1, max(0, max_length - 10), -1):
            if i < len(text) and text[i] == " ":
                return text[:i], text[i + 1:].strip()
        
        # 尝试在中文词汇边界分割（避免拆分常见词汇）
        # 检查是否在常见双字词中间
        common_words = ["量子", "科学", "技术", "系统", "计算", "研究", "实验", "突破", "应用", "发展"]
        for word in common_words:
            if len(word) == 2 and max_length - 2 >= 0:
                word_start = text.find(word, max(0, max_length - 5))
                if word_start != -1 and word_start < max_length < word_start + len(word):
                    # 在双字词中间，调整分割点到词的开始或结束
                    if max_length - word_start <= len(word) // 2:
                        # 更接近词的开始，分割到词前
                        return text[:word_start], text[word_start:]
                    else:
                        # 更接近词的结束，分割到词后
                        return text[:word_start + len(word)], text[word_start + len(word):]

        # 最后的安全分割：确保不在中文字符中间分割
        split_pos = max_length
        while split_pos > max_length - 5 and split_pos > 0:
            char = text[split_pos - 1] if split_pos > 0 else ""
            # 如果是中文字符，往前找合适的分割点
            if '\u4e00' <= char <= '\u9fff':
                split_pos -= 1
            else:
                break
        
        if split_pos <= 0:
            split_pos = max_length
            
        return text[:split_pos], text[split_pos:]

    def _validate_quality(
        self, compressed: str, original: str, target_length: int, strict_length: bool
    ) -> bool:
        """验证压缩质量"""
        if not compressed or len(compressed) == 0:
            return False

        if strict_length and len(compressed) > target_length:
            return False

        # 基本质量检查
        if len(compressed) >= len(original):  # 没有压缩效果
            return False

        return True

    def batch_test(
        self,
        test_cases: List[Dict[str, Any]],
        strategies: Optional[List[str]] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        批量测试不同策略和参数

        Args:
            test_cases: 测试案例列表 [{"text": "...", "target": 10}, ...]
            strategies: 要测试的策略列表
            model: LLM模型

        Returns:
            包含所有测试结果的字典
        """
        strategies = strategies or ["basic", "precise", "few_shot"]
        results: Dict[str, List[Dict[str, Any]]] = {}

        for strategy in strategies:
            results[strategy] = []
            for case in test_cases:
                result = self.compress(
                    text=case["text"],
                    target_length=case["target"],
                    model=model,
                    strategy=strategy,
                    max_attempts=2,  # 批量测试时减少尝试次数
                )
                results[strategy].append(result)

        # 计算统计信息
        stats = {}
        for strategy, strategy_results in results.items():
            success_count = sum(1 for r in strategy_results if r["success"])
            avg_compression = sum(
                r["compression_ratio"] for r in strategy_results
            ) / len(strategy_results)
            avg_time = sum(r["processing_time"] for r in strategy_results) / len(
                strategy_results
            )

            stats[strategy] = {
                "success_rate": success_count / len(strategy_results),
                "avg_compression_ratio": avg_compression,
                "avg_processing_time": avg_time,
                "total_cases": len(strategy_results),
            }

        return {
            "results": results,
            "statistics": stats,
            "total_cases": len(test_cases),
            "strategies_tested": strategies,
        }


# 便捷函数
def quick_compress(text: str, target_length: int, strategy: str = "precise") -> str:
    """快速压缩文本"""
    compressor = LLMTextCompressor()
    result = compressor.compress(text, target_length, strategy=strategy)
    return result["text"]


def compare_strategies(text: str, target_length: int) -> Dict[str, str]:
    """比较不同策略的压缩效果"""
    compressor = LLMTextCompressor()
    strategies = ["basic", "precise", "few_shot", "creative", "strict"]
    results = {}

    for strategy in strategies:
        result = compressor.compress(
            text, target_length, strategy=strategy, max_attempts=1
        )
        results[strategy] = result["text"]

    return results


def compress_with_llm(text: str, target_length: int, **kwargs) -> str:
    """
    兼容性函数：使用LLM压缩文本

    这是对quick_compress的包装，保持与旧版本的兼容性

    Args:
        text: 要压缩的文本
        target_length: 目标长度
        **kwargs: 其他参数传递给quick_compress

    Returns:
        压缩后的文本
    """
    strategy = kwargs.get("strategy", "precise")
    return quick_compress(text, target_length, strategy=strategy)
