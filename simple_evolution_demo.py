#!/usr/bin/env python3
"""
简化版LLM进化演示
演示核心概念和工作流程
"""

import json
import time
import random
from llm_evolution_system import *

# 简单的数学函数优化问题
class MathFunctionEvaluator(FitnessEvaluator):
    """数学函数适应度评估器 - 优化多元函数"""
    
    def evaluate(self, individual: Individual) -> float:
        """评估数学函数 f(x,y) = -(x-3)²-(y-2)² + 10 的最大值"""
        try:
            params = DNA.decode_parameters(individual.dna)
            x = params.get("x", 0)
            y = params.get("y", 0)
            
            # 目标函数：在(3,2)处取得最大值10
            fitness = -(x - 3)**2 - (y - 2)**2 + 10
            
            # 归一化到[0,1]区间
            normalized_fitness = max(0, fitness / 10)
            
            return normalized_fitness
            
        except Exception as e:
            return 0.0


def run_simple_demo():
    """运行简单演示"""
    print("🧬 简化版LLM进化算法演示")
    print("目标：优化函数 f(x,y) = -(x-3)²-(y-2)² + 10")
    print("最优解应该在 x=3, y=2 处")
    print("=" * 50)
    
    # 创建适应度评估器
    evaluator = MathFunctionEvaluator()
    
    # 创建进化引擎
    engine = LLMEvolutionEngine(
        fitness_evaluator=evaluator,
        population_size=8,
        max_generations=5,  # 演示用小数值
        model_name="gpt-4o-mini"
    )
    
    # 初始种子
    seeds = [
        {"x": 0, "y": 0},
        {"x": 1, "y": 1},
        {"x": 5, "y": 4},
        {"x": 2, "y": 3}
    ]
    
    print(f"🌱 初始种子: {seeds}")
    
    # 初始化并进化
    engine.initialize_population(seeds)
    best = engine.evolve()
    
    # 显示结果
    best_params = DNA.decode_parameters(best.dna)
    print(f"\n🏆 最优解:")
    print(f"  参数: x={best_params.get('x', 'N/A'):.3f}, y={best_params.get('y', 'N/A'):.3f}")
    print(f"  适应度: {best.fitness:.4f}")
    print(f"  理论最优: x=3, y=2, fitness=1.0")
    
    # 计算误差
    x_error = abs(best_params.get('x', 0) - 3)
    y_error = abs(best_params.get('y', 0) - 2)
    print(f"  误差: Δx={x_error:.3f}, Δy={y_error:.3f}")


if __name__ == "__main__":
    try:
        run_simple_demo()
    except Exception as e:
        print(f"演示失败: {e}")
        import traceback
        traceback.print_exc()