#!/usr/bin/env python3
"""
LLM Prompt进化算法核心数据结构
实现个体、种群和基础进化操作
"""

import json
import time
import random
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
import litellm


@dataclass
class PromptIndividual:
    """进化个体：代表一个prompt"""
    
    prompt: str                         # DNA（prompt文本）
    fitness: float = 0.0               # 适应度分数
    generation: int = 0                # 所属代数
    token_count: int = 0               # token数量
    parent_ids: List[str] = field(default_factory=list)  # 父代ID列表
    evolution_history: List[str] = field(default_factory=list)  # 进化历史
    birth_timestamp: float = field(default_factory=time.time)   # 出生时间戳
    
    @property
    def id(self) -> str:
        """生成唯一ID"""
        # 使用prompt的hash + 时间戳生成唯一ID
        prompt_hash = hashlib.md5(self.prompt.encode()).hexdigest()[:8]
        return f"G{self.generation:02d}_{prompt_hash}_{int(self.birth_timestamp)}"
    
    @property
    def is_valid(self) -> bool:
        """检查是否在token限制内"""
        return self.token_count <= 300
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptIndividual':
        """从字典创建个体"""
        return cls(**data)


class TokenCounter:
    """Token计数器"""
    
    @staticmethod
    def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
        """计算文本的token数量"""
        try:
            # 使用litellm的token计数功能
            import tiktoken
            
            # 根据模型选择编码器
            if "gpt-4" in model or "gpt-3.5" in model:
                encoding = tiktoken.encoding_for_model("gpt-4")
            elif "claude" in model:
                encoding = tiktoken.get_encoding("cl100k_base")
            else:
                encoding = tiktoken.get_encoding("cl100k_base")
            
            return len(encoding.encode(text))
        except Exception:
            # 简单估算：英文约4字符/token，中文约1.5字符/token
            chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
            other_chars = len(text) - chinese_chars
            return int(chinese_chars / 1.5 + other_chars / 4)


class PromptValidator:
    """Prompt验证器"""
    
    @staticmethod
    def validate_prompt(prompt: str) -> Tuple[bool, str]:
        """验证prompt的有效性"""
        if not prompt or not prompt.strip():
            return False, "Prompt不能为空"
        
        # 检查token数量
        token_count = TokenCounter.count_tokens(prompt)
        if token_count > 300:
            return False, f"Prompt超出300 token限制 (当前: {token_count})"
        
        # 检查是否包含基本的分割指令
        split_keywords = ["分割", "split", "SPLIT", "分块", "segment"]
        if not any(keyword in prompt for keyword in split_keywords):
            return False, "Prompt缺少文本分割相关指令"
        
        return True, "Prompt有效"
    
    @staticmethod
    def extract_task_intent(prompt: str) -> Dict[str, Any]:
        """提取prompt的任务意图"""
        intent = {
            "has_format_instruction": False,
            "has_quality_criteria": False,
            "has_example": False,
            "language": "unknown",
            "style": "unknown"
        }
        
        # 检查格式指令
        format_keywords = ["格式", "format", "输出", "output", "标记", "mark"]
        intent["has_format_instruction"] = any(kw in prompt for kw in format_keywords)
        
        # 检查质量标准
        quality_keywords = ["质量", "quality", "标准", "criteria", "要求", "requirement"]
        intent["has_quality_criteria"] = any(kw in prompt for kw in quality_keywords)
        
        # 检查示例
        example_keywords = ["例子", "example", "示例", "demo", "如下", "如："]
        intent["has_example"] = any(kw in prompt for kw in example_keywords)
        
        # 语言检测
        chinese_ratio = len([c for c in prompt if '\u4e00' <= c <= '\u9fff']) / len(prompt)
        if chinese_ratio > 0.3:
            intent["language"] = "chinese"
        elif chinese_ratio < 0.1:
            intent["language"] = "english"
        else:
            intent["language"] = "mixed"
        
        return intent


class Population:
    """种群管理类"""
    
    def __init__(self, max_size: int = 20):
        self.max_size = max_size
        self.individuals: List[PromptIndividual] = []
        self.generation = 0
        self.history: List[Dict[str, Any]] = []
    
    def add_individual(self, individual: PromptIndividual) -> bool:
        """添加个体到种群"""
        # 验证个体
        is_valid, reason = PromptValidator.validate_prompt(individual.prompt)
        if not is_valid:
            print(f"个体验证失败: {reason}")
            return False
        
        # 更新token计数
        individual.token_count = TokenCounter.count_tokens(individual.prompt)
        individual.generation = self.generation
        
        # 添加到种群
        self.individuals.append(individual)
        
        # 维护种群大小
        if len(self.individuals) > self.max_size:
            # 移除适应度最低的个体
            self.individuals.sort(key=lambda x: x.fitness, reverse=True)
            self.individuals = self.individuals[:self.max_size]
        
        return True
    
    def get_elite(self, count: int = None) -> List[PromptIndividual]:
        """获取精英个体"""
        if count is None:
            count = max(1, len(self.individuals) // 5)  # 默认20%
        
        sorted_individuals = sorted(self.individuals, key=lambda x: x.fitness, reverse=True)
        return sorted_individuals[:count]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取种群统计信息"""
        if not self.individuals:
            return {"size": 0}
        
        fitnesses = [ind.fitness for ind in self.individuals]
        token_counts = [ind.token_count for ind in self.individuals]
        
        return {
            "size": len(self.individuals),
            "generation": self.generation,
            "fitness": {
                "max": max(fitnesses),
                "min": min(fitnesses),
                "avg": sum(fitnesses) / len(fitnesses),
                "std": (sum((f - sum(fitnesses)/len(fitnesses))**2 for f in fitnesses) / len(fitnesses))**0.5
            },
            "token_usage": {
                "max": max(token_counts),
                "min": min(token_counts),
                "avg": sum(token_counts) / len(token_counts)
            }
        }
    
    def evolve_generation(self):
        """进化到下一代"""
        self.generation += 1
        stats = self.get_statistics()
        self.history.append(stats)
        
        # 更新所有个体的代数
        for individual in self.individuals:
            if individual.generation < self.generation:
                individual.generation = self.generation


class LLMInterface:
    """LLM接口封装"""
    
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.7):
        self.model = model
        self.temperature = temperature
    
    def generate(self, prompt: str, max_tokens: int = 500, **kwargs) -> str:
        """生成文本"""
        try:
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM生成失败: {e}")
            return ""
    
    def apply_prompt_to_text(self, prompt: str, text: str) -> List[str]:
        """应用prompt对文本进行分割"""
        try:
            # 构建完整的请求
            full_prompt = prompt.replace("{text}", text) if "{text}" in prompt else f"{prompt}\n\n文本: {text}"
            
            response = self.generate(full_prompt)
            
            # 解析分割结果
            if "[SPLIT]" in response:
                chunks = [chunk.strip() for chunk in response.split("[SPLIT]") if chunk.strip()]
            else:
                # 如果没有找到分割标记，尝试其他解析方法
                chunks = [response.strip()] if response.strip() else [text]
            
            # 验证分割结果
            joined_text = "".join(chunks)
            if len(joined_text) == 0:
                return [text]  # 分割失败，返回原文
            
            return chunks
            
        except Exception as e:
            print(f"应用prompt失败: {e}")
            return [text]  # 返回原文作为单个chunk


if __name__ == "__main__":
    # 测试代码
    print("🧬 测试LLM Prompt进化算法核心组件")
    
    # 测试个体创建
    prompt1 = """请将以下文本按照主题进行分割，在分割点插入[SPLIT]标记：

{text}

要求：
1. 保持主题内容完整性
2. 确保逻辑转换清晰
3. 避免过度分割"""
    
    individual = PromptIndividual(prompt=prompt1)
    print(f"个体ID: {individual.id}")
    print(f"Token数量: {TokenCounter.count_tokens(prompt1)}")
    print(f"有效性: {individual.is_valid}")
    
    # 测试种群
    population = Population(max_size=5)
    success = population.add_individual(individual)
    print(f"添加到种群: {success}")
    
    # 测试LLM接口
    llm = LLMInterface()
    test_text = "人工智能发展迅速。机器学习是核心技术。深度学习取得突破。"
    chunks = llm.apply_prompt_to_text(prompt1, test_text)
    print(f"分割结果: {chunks}")
    
    print("✅ 核心组件测试完成")