#!/usr/bin/env python3
"""
FACE系统参数LLM进化优化演示
使用LLM进化算法优化fractal-context-encoding的参数配置
"""

import os
import sys
import json
from typing import List, Dict, Any
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_evolution_system import *
from src.fractal_context_encoding.processors.text_splitter import LLMTextSplitter
from src.fractal_context_encoding.core.encoder import FACEEncoder
from src.fractal_context_encoding.evaluation.evaluator import ComprehensiveEvaluator


class FACEParameterEvaluator(FitnessEvaluator):
    """FACE系统参数适应度评估器"""
    
    def __init__(self, test_texts: List[str], evaluation_weights: Dict[str, float] = None):
        self.test_texts = test_texts
        self.weights = evaluation_weights or {
            "compression_ratio": 0.25,
            "semantic_quality": 0.30,
            "processing_speed": 0.20,
            "content_integrity": 0.25
        }
        self.evaluator = ComprehensiveEvaluator()
    
    def evaluate(self, individual: Individual) -> float:
        """评估FACE参数配置的适应度"""
        try:
            params = DNA.decode_parameters(individual.dna)
            
            total_score = 0.0
            successful_evaluations = 0
            
            for text in self.test_texts:
                try:
                    # 使用参数创建编码器
                    encoder = FACEEncoder(
                        tau_buffer=params.get("tau_buffer", 100),
                        tau_slice=params.get("tau_slice", 80),
                        batch_size=params.get("batch_size", 4),
                        retain_count=params.get("retain_count", 2)
                    )
                    
                    # 执行编码
                    start_time = time.time()
                    result = encoder.encode_text(text)
                    processing_time = time.time() - start_time
                    
                    # 计算各项指标
                    compression_ratio = self._calculate_compression_ratio(text, result)
                    semantic_quality = self._evaluate_semantic_quality(text, result)
                    processing_speed = self._calculate_speed_score(len(text), processing_time)
                    content_integrity = self._check_content_integrity(text, result)
                    
                    # 综合评分
                    text_score = (
                        compression_ratio * self.weights["compression_ratio"] +
                        semantic_quality * self.weights["semantic_quality"] +
                        processing_speed * self.weights["processing_speed"] +
                        content_integrity * self.weights["content_integrity"]
                    )
                    
                    total_score += text_score
                    successful_evaluations += 1
                    
                except Exception as e:
                    # 参数配置导致错误，给予惩罚分数
                    total_score += 0.1
                    successful_evaluations += 1
            
            if successful_evaluations == 0:
                return 0.0
            
            # 添加参数合理性检查
            rationality_penalty = self._check_parameter_rationality(params)
            
            final_score = (total_score / successful_evaluations) * rationality_penalty
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            print(f"  ⚠️ 参数评估异常: {e}")
            return 0.0
    
    def _calculate_compression_ratio(self, original_text: str, result) -> float:
        """计算压缩比分数"""
        if not hasattr(result, 'nodes') or len(result.nodes) == 0:
            return 0.1
        
        original_length = len(original_text)
        compressed_length = sum(len(node.content) for node in result.nodes)
        
        if compressed_length == 0:
            return 0.1
        
        ratio = compressed_length / original_length
        # 压缩比在0.3-0.8之间最佳
        if 0.3 <= ratio <= 0.8:
            return 1.0
        elif ratio < 0.3:
            return 0.3 / ratio  # 过度压缩惩罚
        else:
            return 0.8 / ratio  # 压缩不足惩罚
    
    def _evaluate_semantic_quality(self, original_text: str, result) -> float:
        """评估语义质量"""
        if not hasattr(result, 'nodes') or len(result.nodes) == 0:
            return 0.1
        
        # 简化的语义质量评估
        # 检查节点数量是否合理
        node_count = len(result.nodes)
        text_length = len(original_text)
        
        # 理想的节点密度：每100-200字符一个节点
        ideal_density = text_length / 150
        density_score = min(1.0, ideal_density / node_count) if node_count > 0 else 0.1
        
        return density_score
    
    def _calculate_speed_score(self, text_length: int, processing_time: float) -> float:
        """计算处理速度分数"""
        if processing_time <= 0:
            return 1.0
        
        chars_per_second = text_length / processing_time
        
        # 理想速度：每秒处理100-500字符
        if 100 <= chars_per_second <= 500:
            return 1.0
        elif chars_per_second > 500:
            return 1.0  # 速度快是好事
        else:
            return chars_per_second / 100  # 速度慢的惩罚
    
    def _check_content_integrity(self, original_text: str, result) -> float:
        """检查内容完整性"""
        try:
            if not hasattr(result, 'nodes'):
                return 0.0
            
            # 重构文本
            reconstructed = ''.join(node.content for node in result.nodes)
            
            # 检查长度
            length_ratio = len(reconstructed) / len(original_text)
            if abs(length_ratio - 1.0) > 0.1:  # 长度差异超过10%
                return 0.5
            
            # 简化的内容一致性检查
            # 检查关键词保留率
            original_words = set(original_text.split())
            reconstructed_words = set(reconstructed.split())
            
            if len(original_words) == 0:
                return 1.0
            
            retention_rate = len(original_words.intersection(reconstructed_words)) / len(original_words)
            return retention_rate
            
        except Exception:
            return 0.0
    
    def _check_parameter_rationality(self, params: Dict[str, Any]) -> float:
        """检查参数合理性"""
        penalty = 1.0
        
        # tau_buffer合理范围：50-300
        tau_buffer = params.get("tau_buffer", 100)
        if not (50 <= tau_buffer <= 300):
            penalty *= 0.8
        
        # tau_slice合理范围：30-150
        tau_slice = params.get("tau_slice", 80)
        if not (30 <= tau_slice <= 150):
            penalty *= 0.8
        
        # batch_size合理范围：1-10
        batch_size = params.get("batch_size", 4)
        if not (1 <= batch_size <= 10):
            penalty *= 0.8
        
        # retain_count合理范围：1-5
        retain_count = params.get("retain_count", 2)
        if not (1 <= retain_count <= 5):
            penalty *= 0.8
        
        # tau_slice应该小于tau_buffer
        if tau_slice >= tau_buffer:
            penalty *= 0.7
        
        return penalty


def create_test_corpus() -> List[str]:
    """创建测试语料库"""
    return [
        # 中文技术文本
        """
        深度学习是机器学习的一个子领域，它基于人工神经网络的表示学习。学习可以是监督式的、半监督式的或无监督式的。
        深度学习架构，如深度神经网络、深度信念网络、深度强化学习、递归神经网络、卷积神经网络和Transformer，
        已被应用于计算机视觉、语音识别、自然语言处理、机器翻译、生物信息学和药物设计等领域。
        """,
        
        # 数学文本
        """
        在数学中，向量空间（也称为线性空间）是一个集合V，其元素称为向量，与两个运算一起定义：
        向量加法（V × V → V）和标量乘法（F × V → V），其中F是一个字段（通常是实数字段ℝ或复数字段ℂ）。
        这些运算必须满足八个公理，包括结合律、交换律、分配律等。
        """,
        
        # 文学文本
        """
        春天来了，万物复苏。柳树抽出了新芽，嫩绿的叶子在春风中轻舞。
        桃花盛开，粉红色的花瓣如云朵般美丽。小鸟在枝头欢快地歌唱，
        似乎在庆祝这个美好的季节。整个大地都焕发出勃勃生机。
        """,
        
        # 新闻文本
        """
        据最新报道，人工智能技术在医疗诊断领域取得了重大突破。
        研究团队开发的AI系统能够在几秒钟内准确识别多种疾病症状，
        诊断准确率达到95%以上。这一技术有望在未来几年内投入临床应用，
        为医疗行业带来革命性的改变。
        """
    ]


def run_face_evolution_experiment():
    """运行FACE参数进化实验"""
    print("🧬 FACE系统参数LLM进化优化实验")
    print("=" * 50)
    
    # 创建测试语料
    test_texts = create_test_corpus()
    print(f"📝 测试语料: {len(test_texts)} 个文本")
    
    # 创建适应度评估器
    fitness_evaluator = FACEParameterEvaluator(test_texts)
    
    # 创建进化引擎
    evolution_engine = LLMEvolutionEngine(
        fitness_evaluator=fitness_evaluator,
        population_size=12,
        elite_ratio=0.25,
        mutation_rate=0.7,
        crossover_rate=0.5,
        max_generations=15,
        model_name="gpt-4o-mini"
    )
    
    # 初始种子参数（基于FACE系统的默认和经验值）
    seed_params = [
        # 默认配置
        {"tau_buffer": 100, "tau_slice": 80, "batch_size": 4, "retain_count": 2},
        
        # 高压缩配置
        {"tau_buffer": 150, "tau_slice": 60, "batch_size": 6, "retain_count": 3},
        
        # 高质量配置
        {"tau_buffer": 80, "tau_slice": 70, "batch_size": 3, "retain_count": 2},
        
        # 快速处理配置
        {"tau_buffer": 120, "tau_slice": 90, "batch_size": 8, "retain_count": 1},
        
        # 平衡配置
        {"tau_buffer": 110, "tau_slice": 85, "batch_size": 5, "retain_count": 2},
        
        # 精细配置
        {"tau_buffer": 90, "tau_slice": 75, "batch_size": 2, "retain_count": 3}
    ]
    
    print(f"🌱 初始种子: {len(seed_params)} 个配置")
    for i, params in enumerate(seed_params):
        print(f"  种子 {i+1}: {params}")
    
    # 初始化种群
    print(f"\n🧬 初始化种群...")
    evolution_engine.initialize_population(seed_params)
    
    # 执行进化
    print(f"\n🚀 开始进化过程...")
    best_solution = evolution_engine.evolve()
    
    # 保存结果
    report_path = "face_evolution_report.json"
    evolution_engine.save_evolution_report(report_path)
    
    # 输出最终结果
    print(f"\n" + "=" * 50)
    print(f"🏆 进化完成！最优FACE参数配置:")
    print(f"=" * 50)
    
    best_params = DNA.decode_parameters(best_solution.dna)
    print(f"📊 适应度分数: {best_solution.fitness:.4f}")
    print(f"🧬 进化代数: {best_solution.generation}")
    print(f"📋 参数配置:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    print(f"\n📈 进化历史:")
    for i, stats in enumerate(evolution_engine.evolution_history[-5:], 1):
        print(f"  最近第{i}代: 最佳={stats['best_fitness']:.4f}, 平均={stats['avg_fitness']:.4f}")
    
    print(f"\n📄 详细报告已保存至: {report_path}")
    
    return best_solution, best_params


if __name__ == "__main__":
    # 设置环境变量（如果需要）
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ 请设置 OPENAI_API_KEY 环境变量")
        print("可以在 .env 文件中设置，或通过命令行导出")
        sys.exit(1)
    
    try:
        # 运行进化实验
        best_solution, best_params = run_face_evolution_experiment()
        
        print(f"\n✨ 实验完成！可以使用以下配置创建优化的FACE编码器:")
        print(f"```python")
        print(f"from fractal_context_encoding import FACEEncoder")
        print(f"")
        print(f"# 使用进化优化的参数")
        print(f"encoder = FACEEncoder(")
        for key, value in best_params.items():
            print(f"    {key}={value},")
        print(f")")
        print(f"```")
        
    except KeyboardInterrupt:
        print(f"\n🛑 用户中断进化过程")
    except Exception as e:
        print(f"\n❌ 进化过程中发生错误: {e}")
        import traceback
        traceback.print_exc()