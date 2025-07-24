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

        # 智能整合模式：直接处理整个文本，由LLM控制长度增长
        self.logger.debug("智能整合模式：直接处理完整文本，长度=%d", len(new_text))
        
        # 直接将整个新文本递归处理到Level 0
        # 长度控制由_recursive_compress负责
        # 采取最小超界的方式循环填充
        leave_text = new_text
        count = 0
        while len(leave_text) + len(fractal_text[0]) > self.get_threshold(0):
            add_text = leave_text[:self.get_threshold(0) - len(fractal_text[0]) + 1]
            fractal_text[0] += add_text
            leave_text = leave_text[len(add_text):]
            self.logger.debug("Loop %d: Level 0 容量超限: %d > %d，触发分割压缩", count, len(fractal_text[0]), 
                          self.get_threshold(0))
            fractal_text = self._recursive_compress(fractal_text, None, 0)
            count += 1
            
        fractal_text[0] += leave_text

        return fractal_text

    def _llm_integrate_and_add(self, existing_content: str, new_text: str) -> str:
        """
        使用LLM智能整合现有内容和新文本
        
        这是对原_add_to_level方法的革命性重构，将机械的文本拼接
        升级为智能的语义整合，同时实现长度的精确控制。
        
        Args:
            existing_content: 当前层级的现有内容, 可能为空
            new_text: 需要添加的新文本
            
        Returns:
            整合后的内容，长度 = len(existing_content) + int(len(new_text) * 0.618)
            
        核心优势：
            1. 语义整合：消除简单拼接的痕迹，提升内容连贯性
            2. 长度控制：每次增长固定比例，避免长度爆炸
            3. 信息去重：自动识别和合并重复内容
            4. 质量提升：在添加过程中就开始信息密度优化
        """
            
        # 计算目标长度：现有长度 + 新文本长度的61.8%
        target_length = len(existing_content) + int(len(new_text) * self.ratio)
        
        self.logger.debug(
            "LLM智能整合: 现有=%d字符, 新增=%d字符, 目标=%d字符",
            len(existing_content), len(new_text), target_length
        )
        
        # 调用LLM进行智能整合，明确分离现有内容和新文本
        try:
            integrated_result = self._call_llm_integration(
                existing_content, new_text, target_length
            )
            
            self.logger.debug(
                "整合完成: %d字符 -> %d字符 (目标%d)",
                len(existing_content) + len(new_text), len(integrated_result), target_length
            )
            
            return integrated_result
            
        except Exception as e:
            # 如果LLM整合失败，直接抛出异常
            self.logger.error("LLM整合失败: %s", str(e))
            raise Exception(f"LLM智能整合失败: {str(e)}")
    
    def _call_llm_integration(self, existing_content: str, new_text: str, target_length: int) -> str:
        """
        调用LLM进行内容整合的具体实现
        
        Args:
            existing_content: 现有内容
            new_text: 新增文本
            target_length: 目标长度
            
        Returns:
            整合后的文本
        """
        compressor = LLMTextCompressor()
        
        # 构建分离式整合提示词
        integration_prompt = f"""你是一个专业的文本整合专家。请智能整合以下两部分内容：

🔵 现有内容（{len(existing_content)}字符）：
{existing_content}

🟢 新增内容（{len(new_text)}字符）：
{new_text}

整合目标：
1. 保持现有内容的核心信息和逻辑结构
2. 将新增内容的关键信息自然融入
3. 去除重复和冗余表述
4. 提升整体语义连贯性
5. 严格控制在{target_length}字符以内

请直接输出整合结果："""
        
        # 使用自定义的整合策略
        result = compressor.compress(
            text=existing_content + "\n" + new_text,
            target_length=target_length,
            llm_config=self.llm_config,
            strategy="precise",
            max_attempts=2,
            strict_length=True,
            custom_template=integration_prompt,
        )
        return result["text"]

    def _recursive_compress(self, fractal_text: List[str], text: str, level: int):
        """
        递归层级编码器（重构版）

        核心重构：将原来的"机械添加+超限压缩"模式升级为"智能整合+容量管理"模式。
        每次文本添加都通过LLM进行智能整合，实现长度的精确控制和语义的优化。

        新工作流程：
        1. 边界检查：防止无限递归
        2. 智能整合：使用LLM将新文本智能整合到现有内容中
        3. 容量检查：检查整合后是否超出容量限制
        4. 分割压缩：超限时按0.382/0.618比例分割并压缩
        5. 递归上升：压缩部分递归到上一层级

        革命性改进：
        - 取消了机械的_add_to_level操作
        - 每次添加都进行语义优化整合
        - 长度增长可控：现有长度 + 新文本长度*0.618
        - 提升了信息密度和语义连贯性

        Args:
            fractal_text: 分形结构
            text: 要编码的文本块
            level: 当前处理的层级

        Returns:
            更新后的分形结构
        """
        # 步骤1：边界保护 - 防止无限递归
        if level >= self.max_levels:
            self.logger.debug("达到最大层级 %d，停止递归", level)
            return fractal_text

        # 自动扩展：确保目标层级存在
        while len(fractal_text) <= level:
            fractal_text.append("")

        # 步骤2：智能整合 - 使用LLM整合现有内容和新文本
        existing_content = fractal_text[level]
       
        if text:
            if existing_content.strip():
                # 有现有内容，进行智能整合
                integrated_content = self._llm_integrate_and_add(existing_content, text)
                self.logger.debug(
                    "Level %d 智能整合完成: %d + %d -> %d 字符",
                    level, len(existing_content), len(text), len(integrated_content)
                )
            else:
                # 空层级，直接使用新文本
                integrated_content = text
                self.logger.debug("Level %d 首次添加: %d 字符", level, len(text))
        else:
            integrated_content = existing_content

        fractal_text[level] = integrated_content

        # 步骤3：容量检查与分形递归
        current_capacity = self.get_threshold(level)
        if len(integrated_content) > current_capacity:
            self.logger.debug(
                "Level %d 容量超限: %d > %d，触发分割压缩",
                level, len(integrated_content), current_capacity
            )
            
            # 步骤4A：容量超限，触发分割压缩
            up_part, preserved_part = smart_split(integrated_content, 1 - self.ratio)
            fractal_text[level] = preserved_part  # 保留部分留在当前层
            
            self.logger.debug(
                "Level %d 分割完成: 保留%d字符，压缩%d字符到上层",
                level, len(preserved_part), len(up_part) if up_part else 0
            )

            # 步骤4B：递归上升 - 压缩部分递归到上层
            if up_part:
                fractal_text = self._recursive_compress(
                    fractal_text, up_part, level + 1
                )
        else:
            self.logger.debug(
                "Level %d 容量充足: %d <= %d",
                level, len(integrated_content), current_capacity
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

        当层级容量超限时，使用"容量控制分割+压缩"策略处理文本：
        1. 容量控制分割：保留部分不超过当前层容量，超出部分全部压缩上升
        2. LLM压缩：对超出部分进行智能压缩，保持关键信息上升到高层级

        修正后的分割策略：
            保留部分长度 = min(当前层容量, 文本长度 * 0.618)
            压缩部分长度 = 文本长度 - 保留部分长度

        这确保了：
        - 保留部分绝对不会超过当前层容量限制
        - 超出的所有内容都被压缩并递归到上层
        - 维持分形的容量控制特性

        Args:
            text: 需要处理的超限文本
            level: 当前层级（影响压缩策略）

        Returns:
            (保留部分, 压缩部分) - 保留部分留在当前层，压缩部分递归到上层
        """
        
        # 步骤1：容量控制分割 - 确保保留部分不超过门限
        current_threshold = self.get_threshold(level)
        
        # 直接按门限截断，确保保留部分不超限
        if len(text) <= current_threshold:
            # 文本本身就不超限，无需分割
            preserved_part = text
            compressed_part = ""
        else:
            # 超限时，严格按门限分割
            preserved_part = text[:current_threshold]
            compressed_part = text[current_threshold:]
        
        self.logger.debug(
            "拆分前文本: %s",
            text[:200] + "..." if len(text) > 200 else text
        )
        self.logger.debug(
            "容量控制分割: 门限=%d, 保留长度=%d, 压缩长度=%d",
            current_threshold, len(preserved_part), len(compressed_part)
        )
        self.logger.debug(
            "拆分后-保留部分(%d字符): %s",
            len(preserved_part),
            preserved_part[:100] + "..." if len(preserved_part) > 100 else preserved_part
        )
        self.logger.debug(
            "拆分后-压缩部分(%d字符): %s",
            len(compressed_part),
            compressed_part[:100] + "..." if len(compressed_part) > 100 else compressed_part
        )

        # 步骤2：LLM压缩超出部分 - 压缩后上升到高层级
        if compressed_part:
            # 计算目标压缩长度：压缩部分按0.618比例压缩
            target_length = int(len(compressed_part) * self.ratio)
            self.logger.debug(
                "压缩前文本: %s",
                compressed_part[:200] + "..." if len(compressed_part) > 200 else compressed_part
            )
            final_compressed_part = self._call_llm_compression(
                text=compressed_part, target_length=target_length
            )
        else:
            # 无需压缩的部分
            final_compressed_part = ""
        
        if final_compressed_part:
            self.logger.debug(
                "压缩后文本: %s",
                final_compressed_part[:200] + "..." if len(final_compressed_part) > 200 else final_compressed_part
            )

        return preserved_part, final_compressed_part

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
