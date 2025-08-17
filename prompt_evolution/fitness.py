#!/usr/bin/env python3
"""
LLM Prompt进化算法适应度评估器
基于文本分割任务的多维度评估
"""

import json
import re
import time
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .core import PromptIndividual, LLMInterface, TokenCounter


@dataclass
class SegmentationSample:
    """文本分割样本"""
    text: str                                    # 原始文本
    ground_truth_boundaries: List[int]           # 标准分割边界位置
    topic_labels: Optional[List[int]] = None     # 主题标签
    metadata: Optional[Dict[str, Any]] = None    # 元数据
    
    @property
    def ground_truth_segments(self) -> List[str]:
        """根据边界获取标准分割段落"""
        if not self.ground_truth_boundaries:
            return [self.text]
        
        segments = []
        start = 0
        for boundary in self.ground_truth_boundaries:
            if boundary > start:
                segments.append(self.text[start:boundary])
                start = boundary
        
        # 添加最后一段
        if start < len(self.text):
            segments.append(self.text[start:])
        
        return [seg.strip() for seg in segments if seg.strip()]


class BoundaryMatcher:
    """边界匹配器"""
    
    @staticmethod
    def find_segment_boundaries(text: str, segments: List[str]) -> List[int]:
        """根据分割段落反推边界位置"""
        boundaries = []
        current_pos = 0
        
        for segment in segments[:-1]:  # 除了最后一个段落
            # 在剩余文本中查找这个段落
            remaining_text = text[current_pos:]
            segment_clean = segment.strip()
            
            if segment_clean in remaining_text:
                start_in_remaining = remaining_text.index(segment_clean)
                segment_end = current_pos + start_in_remaining + len(segment_clean)
                boundaries.append(segment_end)
                current_pos = segment_end
            else:
                # 如果找不到精确匹配，尝试模糊匹配
                print(f"⚠️ 无法找到段落: {segment_clean[:50]}...")
                current_pos += len(segment_clean)
        
        return boundaries
    
    @staticmethod
    def calculate_boundary_accuracy(predicted_boundaries: List[int], 
                                  true_boundaries: List[int], 
                                  tolerance: int = 5) -> Dict[str, float]:
        """计算边界准确率"""
        if not true_boundaries:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        
        if not predicted_boundaries:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        # 计算匹配的边界
        true_positives = 0
        for pred_boundary in predicted_boundaries:
            # 检查是否有真实边界在容忍范围内
            for true_boundary in true_boundaries:
                if abs(pred_boundary - true_boundary) <= tolerance:
                    true_positives += 1
                    break
        
        # 计算精确率和召回率
        precision = true_positives / len(predicted_boundaries) if predicted_boundaries else 0.0
        recall = true_positives / len(true_boundaries) if true_boundaries else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": true_positives,
            "predicted_count": len(predicted_boundaries),
            "true_count": len(true_boundaries)
        }


class SemanticAnalyzer:
    """语义分析器"""
    
    @staticmethod
    def calculate_topic_consistency(segments: List[str], 
                                  topic_labels: Optional[List[int]] = None) -> float:
        """计算主题一致性分数"""
        if not segments or not topic_labels:
            # 如果没有主题标签，使用启发式方法
            return SemanticAnalyzer._heuristic_consistency(segments)
        
        # 基于主题标签的一致性计算
        total_consistency = 0.0
        
        for i, segment in enumerate(segments):
            if i < len(topic_labels):
                # 检查该段落内的主题一致性
                # 这里简化为检查段落长度合理性
                segment_length = len(segment.strip())
                if 10 <= segment_length <= 500:  # 合理长度范围
                    total_consistency += 1.0
                else:
                    total_consistency += 0.5
        
        return total_consistency / len(segments) if segments else 0.0
    
    @staticmethod
    def _heuristic_consistency(segments: List[str]) -> float:
        """启发式一致性评估"""
        if not segments:
            return 0.0
        
        consistency_score = 0.0
        
        for segment in segments:
            segment_clean = segment.strip()
            
            # 长度合理性
            length_score = 1.0
            if len(segment_clean) < 5:
                length_score = 0.2  # 太短
            elif len(segment_clean) > 1000:
                length_score = 0.6  # 太长
            
            # 完整性检查（避免在句子中间分割）
            completeness_score = 1.0
            if segment_clean and not segment_clean[-1] in '.。!！?？':
                if len(segment_clean) > 20:  # 长段落没有结束标点扣分
                    completeness_score = 0.8
            
            # 避免碎片化
            fragment_score = 1.0
            if len(segment_clean.split()) < 3:  # 少于3个词的段落
                fragment_score = 0.5
            
            segment_score = (length_score + completeness_score + fragment_score) / 3
            consistency_score += segment_score
        
        return consistency_score / len(segments)


class GranularityAnalyzer:
    """颗粒度分析器"""
    
    @staticmethod
    def calculate_granularity_score(predicted_segments: List[str], 
                                  true_segments: List[str]) -> float:
        """计算颗粒度合理性分数"""
        if not predicted_segments or not true_segments:
            return 0.0
        
        # 1. 分割数量合理性
        count_ratio = len(predicted_segments) / len(true_segments)
        count_score = 1.0 - abs(1.0 - count_ratio)  # 越接近1.0越好
        count_score = max(0.0, count_score)
        
        # 2. 长度分布合理性
        pred_lengths = [len(seg) for seg in predicted_segments]
        true_lengths = [len(seg) for seg in true_segments]
        
        pred_avg = sum(pred_lengths) / len(pred_lengths)
        true_avg = sum(true_lengths) / len(true_lengths)
        
        length_ratio = min(pred_avg, true_avg) / max(pred_avg, true_avg)
        length_score = length_ratio
        
        # 3. 极端分割惩罚
        extreme_penalty = 0.0
        
        # 检查过度分割（太多小段落）
        small_segments = [seg for seg in predicted_segments if len(seg.strip()) < 10]
        if len(small_segments) > len(predicted_segments) * 0.3:
            extreme_penalty += 0.3
        
        # 检查分割不足（太少大段落）
        if len(predicted_segments) == 1 and len(true_segments) > 1:
            extreme_penalty += 0.5
        
        final_score = (count_score * 0.4 + length_score * 0.4 + (1.0 - extreme_penalty) * 0.2)
        return max(0.0, min(1.0, final_score))


class FitnessEvaluator(ABC):
    """适应度评估器抽象基类"""
    
    @abstractmethod
    def evaluate(self, individual: PromptIndividual, test_samples: List[SegmentationSample]) -> float:
        """评估个体适应度"""
        pass


class TextSegmentationFitnessEvaluator(FitnessEvaluator):
    """文本分割任务适应度评估器"""
    
    def __init__(self, 
                 boundary_weight: float = 0.45,
                 semantic_weight: float = 0.25, 
                 granularity_weight: float = 0.20,
                 efficiency_weight: float = 0.05,
                 penalty_weight: float = 0.05,
                 boundary_tolerance: int = 5):
        
        self.weights = {
            "boundary": boundary_weight,
            "semantic": semantic_weight,
            "granularity": granularity_weight,
            "efficiency": efficiency_weight,
            "penalty": penalty_weight
        }
        self.boundary_tolerance = boundary_tolerance
        self.llm = LLMInterface()
    
    def evaluate(self, individual: PromptIndividual, test_samples: List[SegmentationSample]) -> float:
        """评估prompt的综合适应度"""
        if not test_samples:
            return 0.0
        
        total_score = 0.0
        successful_evaluations = 0
        
        for sample in test_samples:
            try:
                # 应用prompt进行分割
                start_time = time.time()
                predicted_segments = self.llm.apply_prompt_to_text(individual.prompt, sample.text)
                processing_time = time.time() - start_time
                
                # 计算各项指标
                boundary_score = self._evaluate_boundary_accuracy(
                    predicted_segments, sample, individual.prompt
                )
                
                semantic_score = self._evaluate_semantic_consistency(
                    predicted_segments, sample
                )
                
                granularity_score = self._evaluate_granularity(
                    predicted_segments, sample
                )
                
                efficiency_score = self._evaluate_efficiency(
                    individual.token_count, processing_time, len(sample.text)
                )
                
                # 综合评分
                sample_score = (
                    boundary_score * self.weights["boundary"] +
                    semantic_score * self.weights["semantic"] +
                    granularity_score * self.weights["granularity"] +
                    efficiency_score * self.weights["efficiency"]
                )
                
                total_score += sample_score
                successful_evaluations += 1
                
            except Exception as e:
                print(f"⚠️ 样本评估失败: {e}")
                # 给失败的评估一个低分
                total_score += 0.1
                successful_evaluations += 1
        
        if successful_evaluations == 0:
            return 0.0
        
        # 计算平均分
        avg_score = total_score / successful_evaluations
        
        # 应用长度惩罚
        length_penalty = self._calculate_length_penalty(individual.token_count)
        
        final_score = avg_score - length_penalty * self.weights["penalty"]
        return max(0.0, min(1.0, final_score))
    
    def _evaluate_boundary_accuracy(self, predicted_segments: List[str], 
                                   sample: SegmentationSample, 
                                   prompt: str) -> float:
        """评估边界准确率"""
        try:
            # 根据预测的段落计算边界
            predicted_boundaries = BoundaryMatcher.find_segment_boundaries(
                sample.text, predicted_segments
            )
            
            # 计算准确率指标
            accuracy_metrics = BoundaryMatcher.calculate_boundary_accuracy(
                predicted_boundaries, 
                sample.ground_truth_boundaries,
                self.boundary_tolerance
            )
            
            # 使用F1分数作为边界准确率
            return accuracy_metrics["f1"]
            
        except Exception as e:
            print(f"边界准确率计算失败: {e}")
            return 0.1
    
    def _evaluate_semantic_consistency(self, predicted_segments: List[str], 
                                     sample: SegmentationSample) -> float:
        """评估语义一致性"""
        try:
            return SemanticAnalyzer.calculate_topic_consistency(
                predicted_segments, sample.topic_labels
            )
        except Exception as e:
            print(f"语义一致性计算失败: {e}")
            return 0.1
    
    def _evaluate_granularity(self, predicted_segments: List[str], 
                            sample: SegmentationSample) -> float:
        """评估颗粒度合理性"""
        try:
            true_segments = sample.ground_truth_segments
            return GranularityAnalyzer.calculate_granularity_score(
                predicted_segments, true_segments
            )
        except Exception as e:
            print(f"颗粒度评估失败: {e}")
            return 0.1
    
    def _evaluate_efficiency(self, token_count: int, 
                           processing_time: float, 
                           text_length: int) -> float:
        """评估效率"""
        try:
            # Token使用效率
            ideal_token_count = 200  # 理想的token数量
            token_efficiency = min(1.0, ideal_token_count / token_count) if token_count > 0 else 0.0
            
            # 处理速度（字符/秒）
            chars_per_second = text_length / processing_time if processing_time > 0 else 0.0
            speed_score = min(1.0, chars_per_second / 100.0)  # 100字符/秒为满分
            
            return (token_efficiency + speed_score) / 2
            
        except Exception:
            return 0.5
    
    def _calculate_length_penalty(self, token_count: int, max_tokens: int = 300) -> float:
        """计算长度惩罚"""
        if token_count <= max_tokens:
            return 0.0
        else:
            # 超出部分按比例惩罚
            excess_ratio = (token_count - max_tokens) / max_tokens
            return min(1.0, excess_ratio)  # 最大惩罚1.0


# 简化的适应度评估器（用于快速测试）
class SimpleFitnessEvaluator(FitnessEvaluator):
    """简化的适应度评估器"""
    
    def __init__(self):
        self.llm = LLMInterface()
    
    def evaluate(self, individual: PromptIndividual, test_samples: List[SegmentationSample]) -> float:
        """简化的适应度评估"""
        if not test_samples:
            return 0.0
        
        total_score = 0.0
        
        for sample in test_samples[:3]:  # 只使用前3个样本快速评估
            try:
                # 应用prompt
                segments = self.llm.apply_prompt_to_text(individual.prompt, sample.text)
                
                # 简单评分：基于分割数量合理性和内容完整性
                expected_segments = len(sample.ground_truth_segments)
                actual_segments = len(segments)
                
                # 数量合理性
                count_score = 1.0 - abs(expected_segments - actual_segments) / max(expected_segments, 1)
                count_score = max(0.0, count_score)
                
                # 内容完整性
                total_length = sum(len(seg) for seg in segments)
                completeness_score = min(1.0, total_length / len(sample.text))
                
                sample_score = (count_score + completeness_score) / 2
                total_score += sample_score
                
            except Exception:
                total_score += 0.1
        
        avg_score = total_score / min(3, len(test_samples))
        
        # Token长度惩罚
        if individual.token_count > 300:
            avg_score *= 0.5
        
        return max(0.0, min(1.0, avg_score))


if __name__ == "__main__":
    # 测试适应度评估器
    print("🧪 测试适应度评估器")
    
    # 创建测试样本
    test_sample = SegmentationSample(
        text="人工智能发展迅速。机器学习是核心技术。深度学习取得突破。",
        ground_truth_boundaries=[9, 20],  # "人工智能发展迅速。" "机器学习是核心技术。"
        topic_labels=[0, 1, 1]
    )
    
    # 创建测试个体
    test_prompt = """请将文本按主题分割，在分割点插入[SPLIT]：

{text}

要求：保持主题完整性"""
    
    individual = PromptIndividual(prompt=test_prompt)
    individual.token_count = TokenCounter.count_tokens(test_prompt)
    
    # 测试评估器
    evaluator = SimpleFitnessEvaluator()
    fitness = evaluator.evaluate(individual, [test_sample])
    
    print(f"适应度分数: {fitness:.3f}")
    print("✅ 适应度评估器测试完成")