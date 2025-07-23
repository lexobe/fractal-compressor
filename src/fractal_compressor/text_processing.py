"""
智能文本分割工具

提供高效的两段文本分割功能，支持中英文混合文本，
使用偏差阈值机制优化递归策略。
"""

import re
from typing import List, Tuple


def smart_split(
    text: str,
    ratio: float = 0.382,  # 1 - 0.618，黄金分割的互补比例
    deviation_threshold: float = 0.1,
    language: str = "mixed",
) -> Tuple[str, str]:
    """
    智能文本分割：在目标比例附近寻找最佳分割点

    使用递归策略和偏差阈值机制，在目标分割比例附近寻找最佳的语义分割点。
    优先选择句子边界，保持文本的语义完整性。

    Args:
        text: 输入文本
        ratio: 第一段的目标分割比例 (0.0-1.0)，第二段自动为 1-ratio
        deviation_threshold: 偏差阈值，控制搜索范围和递归精度 (0.01-0.5)
        language: 语言类型，支持 "chinese", "english", "mixed"

    Returns:
        Tuple[str, str]: (第一部分, 第二部分)

    Raises:
        ValueError: 当参数不合法时抛出

    Examples:
        >>> part1, part2 = smart_split("Hello world! How are you?", ratio=0.5)
        >>> # 在50%附近寻找最佳分割点

        >>> part1, part2 = smart_split("文本内容。", ratio=0.382, deviation_threshold=0.05)
        >>> # 黄金分割比例，5%偏差阈值
    """
    # 参数验证
    if not text or not text.strip():
        return "", ""

    if not (0.0 <= ratio <= 1.0):
        raise ValueError("分割比例必须在 0.0 到 1.0 之间")

    if not (0.001 <= deviation_threshold <= 0.5):
        raise ValueError("偏差阈值必须在 0.001 到 0.5 之间")

    # 处理自动语言检测
    if language == "auto":
        # 简单的语言检测逻辑
        chinese_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
        english_chars = sum(1 for char in text if char.isalpha() and ord(char) < 128)

        if chinese_chars > 0 and english_chars > 0:
            language = "mixed"
        elif chinese_chars > english_chars:
            language = "chinese"
        else:
            language = "english"

    if language not in ["chinese", "english", "mixed"]:
        raise ValueError("语言类型必须是: 'chinese', 'english', 'mixed'")

    text_length = len(text)
    target_pos = int(text_length * ratio)
    threshold_range = int(text_length * deviation_threshold)

    # 根据偏差阈值计算搜索范围
    min_pos = max(0, target_pos - threshold_range)
    max_pos = min(text_length, target_pos + threshold_range)

    # 确保范围有效
    if min_pos >= max_pos:
        min_pos = max(0, target_pos - 1)
        max_pos = min(text_length, target_pos + 1)

    # 获取分隔符优先级列表
    if language == "chinese":
        separators = ["。", "！", "？", "；", "：", "，", "、", "\n", " "]
    elif language == "english":
        separators = [".", "!", "?", ";", ":", ",", "\n", " "]
    else:  # mixed
        separators = [
            "。",
            "！",
            "？",
            ".",
            "!",
            "?",
            "；",
            ";",
            "：",
            ":",
            "，",
            ",",
            "、",
            "\n",
            " ",
        ]

    def find_best_split_in_range(
        text: str, min_pos: int, max_pos: int, target: int, separators: List[str]
    ) -> int:
        """
        在指定范围内递归查找最佳分割位置

        Args:
            text: 文本内容
            min_pos: 最小位置
            max_pos: 最大位置
            target: 目标位置
            separators: 分隔符优先级列表

        Returns:
            最佳分割位置
        """
        if not separators:
            return target

        current_separator = separators[0]
        remaining_separators = separators[1:]

        # 查找范围内的分隔符位置
        positions = []
        start = min_pos
        while start <= max_pos:
            pos = text.find(current_separator, start)
            if pos == -1 or pos > max_pos:
                break
            split_pos = pos + len(current_separator)
            if min_pos <= split_pos <= max_pos:
                positions.append(split_pos)
            start = pos + 1

        if positions:
            # 找到最接近目标的位置
            best_pos = min(positions, key=lambda x: abs(x - target))
            deviation = abs(best_pos - target)

            # 检查偏差阈值 - 关键优化点
            if deviation <= threshold_range:
                return best_pos

            # 尝试下一个分隔符
            if remaining_separators:
                recursive_pos = find_best_split_in_range(
                    text, min_pos, max_pos, target, remaining_separators
                )
                if abs(recursive_pos - target) < deviation:
                    return recursive_pos

            return best_pos

        # 当前分隔符没有找到，尝试下一个
        if remaining_separators:
            return find_best_split_in_range(
                text, min_pos, max_pos, target, remaining_separators
            )

        return target

    # 查找最佳分割位置
    split_pos = find_best_split_in_range(text, min_pos, max_pos, target_pos, separators)

    # 确保分割位置在有效范围内
    split_pos = max(1, min(split_pos, text_length - 1))

    # 分割并清理首尾空白
    first_part = text[:split_pos].rstrip()
    second_part = text[split_pos:].strip()

    return first_part, second_part
