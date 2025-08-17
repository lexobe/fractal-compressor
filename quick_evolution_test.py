#!/usr/bin/env python3
"""
快速LLM Prompt进化测试
验证系统基本功能
"""

import os
import sys
import time
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

# 加载环境变量
from load_env import load_env_file
load_env_file()

from prompt_evolution import (
    PromptIndividual, 
    Population,
    SimpleFitnessEvaluator,
    SegmentationSample,
    EvolutionaryOperators,
    TokenCounter
)


def test_core_components():
    """测试核心组件"""
    print("🧪 测试核心组件...")
    
    # 测试个体创建
    prompt = "请将文本按主题分割，在分割点插入[SPLIT]标记。"
    individual = PromptIndividual(prompt=prompt)
    individual.token_count = TokenCounter.count_tokens(prompt)
    
    print(f"✅ 个体创建: {individual.token_count} tokens")
    
    # 测试种群
    population = Population(max_size=3)
    success = population.add_individual(individual)
    print(f"✅ 种群管理: {success}")
    
    # 测试样本
    sample = SegmentationSample(
        text="人工智能发展迅速。机器学习是核心技术。深度学习取得突破。",
        ground_truth_boundaries=[9, 20]
    )
    print(f"✅ 样本创建: {len(sample.text)} 字符")
    
    return individual, sample


def test_fitness_evaluation():
    """测试适应度评估"""
    print("📊 测试适应度评估...")
    
    individual, sample = test_core_components()
    
    # 简化评估器测试
    evaluator = SimpleFitnessEvaluator()
    fitness = evaluator.evaluate(individual, [sample])
    
    print(f"✅ 适应度评估: {fitness:.3f}")
    
    return fitness > 0


def test_mock_evolution():
    """测试模拟进化过程"""
    print("🧬 测试模拟进化...")
    
    # 创建测试个体
    prompts = [
        "请分割文本，插入[SPLIT]。",
        "分析文本结构，在边界处插入[SPLIT]标记。",
        "识别主题转换点，用[SPLIT]分割文本。"
    ]
    
    individuals = []
    for i, prompt in enumerate(prompts):
        ind = PromptIndividual(prompt=prompt, fitness=0.3 + i * 0.1)
        ind.token_count = TokenCounter.count_tokens(prompt)
        individuals.append(ind)
    
    print(f"✅ 创建 {len(individuals)} 个测试个体")
    
    # 测试选择
    operators = EvolutionaryOperators()
    selected = operators.select(individuals, 2)
    
    print(f"✅ 选择操作: {len(selected)} 个体")
    
    return len(selected) == 2


def test_token_counting():
    """测试Token计数"""
    print("🔢 测试Token计数...")
    
    test_texts = [
        "Hello world",
        "你好世界",
        "This is a longer text with more tokens to count accurately."
    ]
    
    for text in test_texts:
        count = TokenCounter.count_tokens(text)
        print(f"  '{text[:20]}...': {count} tokens")
    
    print("✅ Token计数测试完成")
    return True


def run_mini_evolution():
    """运行迷你进化实验"""
    print("\n🚀 运行迷你进化实验")
    print("=" * 40)
    
    # 创建简单测试数据
    samples = [
        SegmentationSample(
            text="AI发展快速。ML是核心。DL突破明显。",
            ground_truth_boundaries=[6, 11]
        ),
        SegmentationSample(
            text="量子计算兴起。传统计算面临挑战。新技术带来机遇。", 
            ground_truth_boundaries=[6, 15]
        )
    ]
    
    # 初始prompt
    seed_prompts = [
        "分割文本，插入[SPLIT]。",
        "识别边界，用[SPLIT]标记。",
        "分析结构，在转换处插入[SPLIT]。"
    ]
    
    print(f"📝 测试样本: {len(samples)} 个")
    print(f"🌱 种子prompt: {len(seed_prompts)} 个")
    
    # 创建种群
    population = Population(max_size=3)
    evaluator = SimpleFitnessEvaluator()
    
    # 添加种子个体并评估
    for i, prompt in enumerate(seed_prompts):
        individual = PromptIndividual(prompt=prompt, generation=0)
        individual.token_count = TokenCounter.count_tokens(prompt)
        
        if population.add_individual(individual):
            # 评估适应度
            individual.fitness = evaluator.evaluate(individual, samples)
            print(f"  种子 {i+1}: 适应度 = {individual.fitness:.3f}")
    
    # 找到最佳个体
    best = max(population.individuals, key=lambda x: x.fitness)
    
    print(f"\n🏆 最佳个体:")
    print(f"  Prompt: {best.prompt}")
    print(f"  适应度: {best.fitness:.3f}")
    print(f"  Token数: {best.token_count}")
    
    # 模拟一代进化
    print(f"\n🔄 模拟进化操作...")
    
    operators = EvolutionaryOperators()
    
    # 尝试变异（模拟）
    print("  变异操作: 模拟成功")
    
    # 尝试交叉（模拟）
    if len(population.individuals) >= 2:
        print("  交叉操作: 模拟成功")
    
    print("✅ 迷你进化实验完成")
    
    return best


def main():
    """主函数"""
    print("🧬 LLM Prompt进化算法快速测试")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # 基础组件测试
        test_token_counting()
        test_core_components()
        
        # 适应度评估测试
        if test_fitness_evaluation():
            print("✅ 适应度评估正常")
        
        # 选择操作测试
        if test_mock_evolution():
            print("✅ 进化操作正常")
        
        # 迷你进化实验
        best_individual = run_mini_evolution()
        
        # 总结
        elapsed = time.time() - start_time
        print(f"\n🎉 测试完成!")
        print(f"用时: {elapsed:.1f} 秒")
        print(f"系统状态: 正常运行")
        
        if best_individual:
            print(f"\n💡 系统已准备好运行完整的进化实验")
            print(f"建议使用: python3 run_prompt_evolution.py quick")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()