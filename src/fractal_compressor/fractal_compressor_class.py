"""
分形文本压缩器 - Fractal Text Compressor

基于分形理论的多层级文本压缩系统。通过智能分块、层级放置和递归压缩，
实现文本的分形结构化存储，在保持信息完整性的同时实现高效压缩。

核心特性：
- 分块分发：将长文本智能切分并分发到分形结构
- 层级压缩：每层容量超限时自动触发LLM压缩
- 递归分形：压缩内容递归存储到上层，形成分形结构
- 容量控制：每层都有基于黄金比例的容量限制
- 详细日志：完整记录压缩过程，便于调试和分析

工作原理：
Level 0 (主内容层) → Level 1 (一级压缩) → Level 2 (二级压缩) → ...

LLM配置示例：
# OpenAI配置
llm_config = {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "api_key": "sk-xxx...",
    "url": "https://api.openai.com/v1",
    "temperature": 0.1
}

# Anthropic配置
llm_config = {
    "provider": "anthropic",
    "model": "claude-3-haiku-20240307",
    "api_key": "sk-ant-xxx...",
    "url": "https://api.anthropic.com",
    "temperature": 0.2
}

# Azure OpenAI配置
llm_config = {
    "provider": "azure_openai",
    "model": "gpt-4o",
    "api_key": "xxx...",
    "url": "https://xxx.openai.azure.com/",
    "api_version": "2024-02-15-preview"
}

# 本地模型配置
llm_config = {
    "provider": "local",
    "model": "llama2",
    "url": "http://localhost:11434",
    "temperature": 0.1
}
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

# 导入必要的模块
from .llm_compressor import LLMTextCompressor
from .text_processing import smart_split

# 获取模块日志记录器
logger = logging.getLogger(__name__)


class FractalCompressor:
    """
    分形文本压缩器

    实现基于分形理论的多层级文本压缩系统。系统将文本分解成多个层级，
    每层都有容量限制，超出时通过LLM压缩并递归到上层，形成分形结构。

    结构设计：
    - Level 0: 主内容层，存储原始文本片段，直接连接无分隔符
    - Level 1+: 压缩层，存储LLM压缩的摘要内容，用换行符分隔

    容量算法：
    每层容量 = base_threshold × ratio^level (基于黄金比例递减)

    参数：
        ratio: 压缩比例，默认0.618（黄金比例）
        base_threshold: 第0层的基础容量限制，默认1000字符
        max_levels: 最大层级数，默认10层
        llm_config: LLM配置字典，包含提供商、模型、API密钥、URL等完整配置
    """

    def __init__(
        self,
        ratio: float = 0.618,
        base_threshold: int = 1000,
        max_levels: int = 10,
        llm_config: Optional[Dict[str, Any]] = None,
    ):
        self.ratio = ratio
        self.base_threshold = base_threshold
        self.max_levels = max_levels
        self.llm_config = llm_config or {}
        
        # 使用标准的logging
        self.logger = logging.getLogger(f"{__name__}.FractalCompressor")
        
        # 记录初始化信息
        self.logger.info(
            "初始化分形压缩器: ratio=%.3f, base_threshold=%d, max_levels=%d",
            ratio, base_threshold, max_levels
        )
        self.logger.debug(
            "LLM配置: provider=%s, model=%s",
            self.llm_config.get('provider', 'None'),
            self.llm_config.get('model', 'None')
        )

    def compress(self, fractal_text: List[str], new_text: str) -> List[str]:
        """
        分块分发新文本到分形结构中

        这是系统的入口方法，负责将新输入的文本智能分割并分发到分形结构。
        采用"容量感知分块"策略，考虑Level 0的剩余容量，实现最优分块。

        分块策略：
        1. 计算Level 0的剩余容量
        2. 如果有剩余容量，优先填满（允许+1字符的最小超界）
        3. 如果Level 0已满，按标准容量+1进行分块
        4. 每个分块递归处理，可能触发多层级压缩

        Args:
            fractal_text: 现有的分形结构（多层级文本列表）
            new_text: 需要编码的新文本

        Returns:
            更新后的分形结构

        注意：
            +1超界机制避免产生过小的文本块，提高编码效率
        """
        # 记录开始压缩
        self.logger.info("开始压缩文本: 长度=%d, 目标层级=0", len(new_text))
        
        # 初始化：确保分形结构有足够的层级
        while len(fractal_text) < self.max_levels:
            fractal_text.append("")

        # 主循环：容量感知的智能分块处理
        remaining_text = new_text
        level0_capacity = self.get_threshold(0)
        

        chunk_count = 0
        while remaining_text:
            chunk_count += 1
            
            # 步骤1：分析Level 0的容量状态
            current_length = len(fractal_text[0])
            available_space = level0_capacity - current_length

            # 步骤2：智能分块策略（+1超界机制）
            if available_space > 0:
                # 情况A：Level 0还有空间，优先填满（允许+1超界防止碎片）
                chunk_size = min(available_space + 1, len(remaining_text))
                strategy = "填充剩余空间"
            else:
                # 情况B：Level 0已满，使用标准容量+1进行分块
                chunk_size = min(level0_capacity + 1, len(remaining_text))
                strategy = "标准分块"

            # 步骤3：执行分块
            chunk = remaining_text[:chunk_size]
            remaining_text = remaining_text[chunk_size:]
            

            # 步骤4：递归处理分块（可能触发多层级压缩）
            fractal_text = self._recursive_compress(fractal_text, chunk, 0)

        total_levels = sum(1 for level in fractal_text if level.strip())
        total_length = sum(len(level) for level in fractal_text)
        self.logger.info(
            "压缩完成: 共%d个分块, %d个活跃层级, 总长度=%d",
            chunk_count, total_levels, total_length
        )

        return fractal_text

    def _add_to_level(self, fractal_text: List[str], text: str, level: int):
        """
        层级文本追加器

        负责将文本追加到指定层级，根据层级特性采用不同的连接策略：
        - Level 0: 直接连接，保持文本流畅性
        - Level 1+: 换行分隔，便于区分不同的压缩块

        自动扩展机制：如果目标层级不存在，自动创建到该层级的所有空层。

        Args:
            fractal_text: 分形结构
            text: 要追加的文本
            level: 目标层级（0为主内容层）

        连接策略：
            Level 0: "现有内容" + "新文本"
            Level 1+: "现有内容" + "\n" + "新文本"
        """
        # 自动扩展：确保目标层级及之前的所有层级都存在
        while len(fractal_text) <= level:
            fractal_text.append("")

        old_length = len(fractal_text[level])
        
        # 智能追加：根据层级特性和现有内容状态选择连接策略
        if fractal_text[level].strip():
            if level == 0:
                # Level 0策略：直接连接，保持文本连续性
                fractal_text[level] += text
                strategy = "直接连接"
            else:
                # Level 1+策略：换行分隔，便于阅读不同的压缩块
                fractal_text[level] += "\n" + text
                strategy = "换行分隔"
        else:
            # 空层级：直接设置为新文本
            fractal_text[level] = text
            strategy = "直接设置"
        
        new_length = len(fractal_text[level])

    def _recursive_compress(self, fractal_text: List[str], text: str, level: int):
        """
        递归层级编码器

        系统的核心递归引擎，负责将文本块放置到指定层级，并在容量超限时
        触发分割压缩，形成向上递归的分形结构。

        工作流程：
        1. 边界检查：如果达到最大层级，直接返回
        2. 文本放置：将文本追加到当前层级
        3. 容量检查：如果当前层级超出容量限制
        4. 分割压缩：使用LLM将超出部分压缩
        5. 递归上升：将压缩结果递归到上一层级

        Args:
            fractal_text: 分形结构
            text: 要编码的文本块
            level: 当前处理的层级

        Returns:
            更新后的分形结构

        注意：
            这里不涉及+1超界控制，该机制仅在encode方法的分块阶段使用
        """
        # 步骤1：边界保护 - 防止无限递归
        if level >= self.max_levels:
                return fractal_text

        # 步骤2：文本放置 - 将文本块追加到当前层级
        self._add_to_level(fractal_text, text, level)

        # 步骤3：容量检查与分形递归
        current_capacity = self.get_threshold(level)
        if len(fractal_text[level]) > current_capacity:
            
            # 步骤4A：容量超限，触发分割压缩
            preserved_part, compressed_part = self._llm_compress(
                fractal_text[level], level
            )
            fractal_text[level] = preserved_part  # 保留部分留在当前层
            

            # 步骤4B：递归上升 - 压缩部分递归到上层
            if compressed_part:
                fractal_text = self._recursive_compress(
                    fractal_text, compressed_part, level + 1
                )

        return fractal_text

    def get_threshold(self, level: int) -> int:
        """
        分形容量计算器

        基于黄金比例理论计算各层级的容量限制。容量随层级递减，
        形成分形结构的数学基础。

        计算公式：
            容量 = base_threshold × ratio^level

        示例（ratio=0.618, base_threshold=1000）：
            Level 0: 1000 字符
            Level 1: 618 字符
            Level 2: 382 字符
            Level 3: 236 字符

        Args:
            level: 层级编号（0为最底层）

        Returns:
            该层级的容量限制（字符数）
        """
        return int(self.base_threshold * (self.ratio**level))

    def _llm_compress(self, text: str, level: int) -> Tuple[str, str]:
        """
        智能分割压缩器

        当层级容量超限时，使用"分割+压缩"策略处理文本：
        1. 智能分割：按黄金比例分割文本，前0.382包含核心信息
        2. LLM压缩：对前部分进行智能压缩，保持关键信息上升到高层级

        分割策略（修正后）：
            前部分比例 = 1 - ratio = 1 - 0.618 = 0.382 (压缩后上升)
            后部分比例 = ratio = 0.618 (保留在当前层)

        这确保了核心信息(前0.382)在上层形成长程关联，
        具体细节(后0.618)在当前层保留。

        Args:
            text: 需要处理的超限文本
            level: 当前层级（影响压缩策略）

        Returns:
            (保留部分, 压缩部分) - 保留部分留在当前层，压缩部分递归到上层

        注意：
            如果分割后的前部分为空，返回空字符串而不调用LLM
        """
        
        # 步骤1：智能分割 - 按黄金比例分为前后两部分
        front_ratio = 1 - self.ratio  # 0.382 前部分比例
        front_part, back_part = smart_split(
            text,
            ratio=front_ratio,
            language=self.llm_config.get("language", "mixed"),
        )
        
        self.logger.debug(
            "拆分前文本: %s",
            text[:200] + "..." if len(text) > 200 else text
        )
        self.logger.debug(
            "拆分后-前部分(%.1f%%): %s",
            len(front_part)/len(text)*100,
            front_part[:100] + "..." if len(front_part) > 100 else front_part
        )
        self.logger.debug(
            "拆分后-后部分(%.1f%%): %s",
            len(back_part)/len(text)*100,
            back_part[:100] + "..." if len(back_part) > 100 else back_part
        )

        # 步骤2：LLM压缩前部分 - 核心信息上升到高层级
        if front_part:
            # 计算目标压缩长度
            target_length = int(len(front_part) * self.ratio)
            self.logger.debug(
                "压缩前文本: %s",
                front_part[:200] + "..." if len(front_part) > 200 else front_part
            )
            compressed_part = self._call_llm_compression(
                text=front_part, target_length=target_length
            )
        else:
            # 无前部分，无需压缩
            compressed_part = ""

        # 后部分保留在当前层级
        preserved_part = back_part
        
        if compressed_part:
            self.logger.debug(
                "压缩后文本: %s",
                compressed_part[:200] + "..." if len(compressed_part) > 200 else compressed_part
            )

        return preserved_part, compressed_part

    def _call_llm_compression(self, text: str, target_length: int) -> str:
        """
        调用LLM进行文本压缩

        根据配置的LLM提供商和参数，调用相应的API进行文本压缩。

        Args:
            text: 需要压缩的文本
            target_length: 目标压缩长度

        Returns:
            压缩后的文本
        """
        
        try:
            compressor = LLMTextCompressor()
            result = compressor.compress(
                text=text,
                target_length=target_length,
                llm_config=self.llm_config,
                strategy="precise",
                max_attempts=2,
                strict_length=True,
            )
            compressed_text = result["text"]
            return compressed_text
        except Exception as e:
            # 如果LLM调用失败，返回简单截断作为fallback
            self.logger.warning("LLM压缩失败，使用fallback截断: %s", str(e))
            if len(text) <= target_length:
                fallback_text = text
            else:
                fallback_text = text[:target_length] + "..."
            
            return fallback_text

    def create_empty_fractal(self) -> List[str]:
        """
        创建空分形结构

        初始化一个空的多层级分形结构，为文本编码做准备。
        每个层级都初始化为空字符串。

        Returns:
            空的分形结构列表，长度为max_levels
        """
        return [""] * self.max_levels

    def compress_single_text(self, text: str) -> List[str]:
        """
        单文本分形编码

        便捷方法：对单个文本进行完整的分形编码。
        内部创建空分形结构，然后调用encode方法处理。

        Args:
            text: 要编码的完整文本

        Returns:
            编码后的分形结构

        使用场景：
            独立文本的一次性编码，不需要保持分形状态
        """
        self.logger.info("开始单文本分形编码: 长度=%d", len(text))
        fractal_text = self.create_empty_fractal()
        result = self.compress(fractal_text, text)
        self.logger.info("单文本分形编码完成")
        return result


# ==================== 便捷函数 ====================


def create_fractal_compressor(llm_config: Optional[Dict[str, Any]] = None, **kwargs):
    """
    创建分形编码器实例

    便捷函数，用于快速创建配置好的FractalCompressor实例。

    Args:
        llm_config: LLM配置字典
        **kwargs: FractalCompressor的其他初始化参数

    Returns:
        配置好的FractalCompressor实例

    Examples:
        # 简单配置
        encoder = create_fractal_compressor(
            llm_config={"provider": "openai", "api_key": "sk-xxx"}
        )

        # 完整配置
        encoder = create_fractal_compressor(
            ratio=0.7,
            base_threshold=2000,
            llm_config={
                "provider": "anthropic",
                "model": "claude-3-sonnet-20240229",
                "api_key": "sk-ant-xxx",
                "url": "https://api.anthropic.com",
                "temperature": 0.2
            }
        )
    """
    return FractalCompressor(llm_config=llm_config, **kwargs)


def encode_text_fractal(
    text: str, llm_config: Optional[Dict[str, Any]] = None, **kwargs
) -> List[str]:
    """
    一站式文本分形编码

    最简便的使用方式：直接对文本进行分形编码，返回完整的分形结构。
    内部会创建编码器实例并执行编码。

    Args:
        text: 要编码的文本
        llm_config: LLM配置字典
        **kwargs: FractalCompressor的其他初始化参数

    Returns:
        编码后的分形结构

    Examples:
        # 基本使用
        result = encode_text_fractal(
            "长文本内容...",
            llm_config={"provider": "openai", "api_key": "sk-xxx"}
        )
        print("Level 0:", result[0])  # 主内容层
        print("Level 1:", result[1])  # 一级压缩层

        # 自定义配置
        result = encode_text_fractal(
            "长文本内容...",
            ratio=0.7,
            llm_config={
                "provider": "anthropic",
                "model": "claude-3-sonnet-20240229",
                "api_key": "sk-ant-xxx",
                "temperature": 0.3
            }
        )
    """
    encoder = FractalCompressor(llm_config=llm_config, **kwargs)
    return encoder.compress_single_text(text)
