#!/usr/bin/env python3
"""
LLM Prompt进化算法引擎
实现完整的进化过程管理和控制
"""

import json
import time
import random
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

from .core import PromptIndividual, Population, LLMInterface, TokenCounter
from .fitness import FitnessEvaluator, TextSegmentationFitnessEvaluator, SegmentationSample
from .operators import EvolutionaryOperators, LLMMutationOperator, LLMCrossoverOperator, TournamentSelection


@dataclass
class EvolutionConfig:
    """进化算法配置"""
    population_size: int = 20
    max_generations: int = 30
    elite_ratio: float = 0.2
    mutation_rate: float = 0.7
    crossover_rate: float = 0.5
    tournament_size: int = 3
    convergence_threshold: float = 0.001
    max_prompt_tokens: int = 300
    stagnation_limit: int = 5  # 连续多少代无改进则停止
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class GenerationStats:
    """每代统计信息"""
    generation: int
    best_fitness: float
    avg_fitness: float
    min_fitness: float
    fitness_std: float
    population_size: int
    diversity_score: float
    elite_count: int
    new_individuals: int
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


class PromptEvolutionEngine:
    """LLM Prompt进化引擎"""
    
    def __init__(self,
                 fitness_evaluator: FitnessEvaluator,
                 config: EvolutionConfig = None,
                 model_name: str = "gpt-4o-mini"):
        
        self.config = config or EvolutionConfig()
        self.fitness_evaluator = fitness_evaluator
        self.model_name = model_name
        
        # 初始化组件
        self.population = Population(max_size=self.config.population_size)
        self.operators = EvolutionaryOperators(
            mutation_operator=LLMMutationOperator(model_name),
            crossover_operator=LLMCrossoverOperator(model_name),
            selection_operator=TournamentSelection(self.config.tournament_size)
        )
        
        # 进化状态
        self.current_generation = 0
        self.best_individual: Optional[PromptIndividual] = None
        self.evolution_history: List[GenerationStats] = []
        self.stagnation_count = 0
        self.start_time = time.time()
        
        # 实验记录
        self.experiment_log = {
            "config": self.config.to_dict(),
            "model_name": model_name,
            "start_time": self.start_time,
            "generations": [],
            "best_individual": None,
            "convergence_info": {}
        }
    
    def initialize_population(self, seed_prompts: List[str]) -> None:
        """初始化种群"""
        print(f"🧬 初始化种群 (目标大小: {self.config.population_size})")
        
        # 添加种子prompt
        for i, prompt in enumerate(seed_prompts):
            individual = PromptIndividual(
                prompt=prompt,
                generation=0,
                evolution_history=[f"种子prompt {i+1}"]
            )
            
            if self.population.add_individual(individual):
                print(f"  ✅ 种子 {i+1}: {len(prompt)} 字符")
            else:
                print(f"  ❌ 种子 {i+1}: 验证失败")
        
        # 如果种子不足，使用LLM生成更多个体
        while len(self.population.individuals) < self.config.population_size:
            new_individual = self._generate_random_individual()
            if new_individual and self.population.add_individual(new_individual):
                print(f"  🤖 LLM生成个体: {new_individual.token_count} tokens")
        
        print(f"✅ 种群初始化完成，实际大小: {len(self.population.individuals)}")
    
    def evaluate_population(self, test_samples: List[SegmentationSample]) -> None:
        """评估整个种群的适应度"""
        print(f"📊 评估第 {self.current_generation} 代种群适应度...")
        
        for i, individual in enumerate(self.population.individuals):
            if individual.fitness == 0.0:  # 只评估未评估的个体
                try:
                    individual.fitness = self.fitness_evaluator.evaluate(individual, test_samples)
                    print(f"  个体 {i+1}: {individual.fitness:.3f}")
                except Exception as e:
                    print(f"  个体 {i+1}: 评估失败 - {e}")
                    individual.fitness = 0.1
        
        # 更新最佳个体
        self._update_best_individual()
    
    def evolve_generation(self) -> GenerationStats:
        """执行一代进化"""
        print(f"\n🚀 开始第 {self.current_generation + 1} 代进化")
        
        # 选择精英个体
        elite_count = max(1, int(len(self.population.individuals) * self.config.elite_ratio))
        elite_individuals = self.population.get_elite(elite_count)
        print(f"🏆 保留精英: {len(elite_individuals)} 个")
        
        # 生成新一代
        new_population = elite_individuals.copy()
        new_individuals_count = 0
        
        while len(new_population) < self.config.population_size:
            # 决定使用交叉还是变异
            if random.random() < self.config.crossover_rate and len(self.population.individuals) >= 2:
                # 交叉操作
                parent1, parent2 = self.operators.select_parents(self.population.individuals)
                child = self.operators.crossover(parent1, parent2)
                operation = "交叉"
            else:
                # 变异操作
                parent = self.operators.select(self.population.individuals, 1)[0]
                mutation_strength = random.uniform(0.2, 0.8)
                child = self.operators.mutate(parent, mutation_strength)
                operation = "变异"
            
            if child:
                child.generation = self.current_generation + 1
                new_population.append(child)
                new_individuals_count += 1
                print(f"  ✅ {operation}: {child.token_count} tokens")
            else:
                print(f"  ❌ {operation}: 生成失败")
        
        # 更新种群
        self.population.individuals = new_population[:self.config.population_size]
        self.current_generation += 1
        self.population.generation = self.current_generation
        
        # 计算代统计信息
        stats = self._calculate_generation_stats(elite_count, new_individuals_count)
        self.evolution_history.append(stats)
        
        # 检查停滞
        self._check_stagnation(stats)
        
        print(f"📈 第 {self.current_generation} 代完成:")
        print(f"   最佳适应度: {stats.best_fitness:.4f}")
        print(f"   平均适应度: {stats.avg_fitness:.4f}")
        print(f"   多样性分数: {stats.diversity_score:.4f}")
        
        return stats
    
    def run_evolution(self, 
                     test_samples: List[SegmentationSample],
                     seed_prompts: List[str] = None) -> PromptIndividual:
        """运行完整的进化过程"""
        
        print("🧬 开始LLM Prompt进化实验")
        print("=" * 60)
        print(f"配置: {self.config.population_size} 个体, {self.config.max_generations} 代")
        print(f"模型: {self.model_name}")
        print(f"测试样本: {len(test_samples)} 个")
        
        # 初始化种群
        if seed_prompts:
            self.initialize_population(seed_prompts)
        else:
            self.initialize_population(self._get_default_seed_prompts())
        
        # 初始评估
        self.evaluate_population(test_samples)
        
        # 记录初始状态
        initial_stats = self._calculate_generation_stats(0, len(self.population.individuals))
        self.evolution_history.append(initial_stats)
        
        # 进化循环
        for generation in range(1, self.config.max_generations + 1):
            # 执行一代进化
            stats = self.evolve_generation()
            
            # 评估新个体
            self.evaluate_population(test_samples)
            
            # 更新统计信息
            updated_stats = self._calculate_generation_stats(
                stats.elite_count, stats.new_individuals
            )
            self.evolution_history[-1] = updated_stats
            
            # 记录到实验日志
            self.experiment_log["generations"].append(updated_stats.to_dict())
            
            # 检查收敛条件
            if self._check_convergence():
                print(f"🎯 在第 {generation} 代达到收敛")
                break
            
            # 检查停滞条件
            if self.stagnation_count >= self.config.stagnation_limit:
                print(f"⏹️ 连续 {self.stagnation_count} 代无改进，停止进化")
                break
        
        # 完成进化
        self._finalize_evolution()
        
        print(f"\n🏆 进化完成！")
        print(f"最终最佳适应度: {self.best_individual.fitness:.4f}")
        print(f"总用时: {time.time() - self.start_time:.1f} 秒")
        
        return self.best_individual
    
    def _generate_random_individual(self) -> Optional[PromptIndividual]:
        """使用LLM生成随机个体"""
        generation_prompt = f"""创建一个用于文本分割的prompt，要求：

1. 长度不超过{self.config.max_prompt_tokens} tokens
2. 指导LLM将文本在语义边界处分割
3. 输出格式要求在分割点插入"[SPLIT]"标记
4. 包含清晰的任务描述和质量要求

创新要求：
- 可以使用不同的表达方式和角度
- 可以加入特定的分析策略
- 可以设计独特的指导方法
- 保持简洁高效

请直接输出prompt，不要添加解释："""
        
        try:
            llm = LLMInterface(model=self.model_name, temperature=0.8)
            prompt_text = llm.generate(generation_prompt, max_tokens=400)
            
            if prompt_text and len(prompt_text.strip()) > 10:
                individual = PromptIndividual(
                    prompt=prompt_text.strip(),
                    generation=0,
                    evolution_history=["LLM随机生成"]
                )
                individual.token_count = TokenCounter.count_tokens(prompt_text)
                return individual
            
        except Exception as e:
            print(f"随机生成个体失败: {e}")
        
        return None
    
    def _get_default_seed_prompts(self) -> List[str]:
        """获取默认种子prompt"""
        return [
            """请将以下文本按照主题进行分割。在每个主题转换的地方插入分割标记"[SPLIT]"。

文本: {text}

要求:
1. 保持主题内容的完整性
2. 确保逻辑转换清晰
3. 避免过度分割""",

            """作为文本分析专家，请识别以下文本中的语义边界，并在适当位置插入"[SPLIT]"标记。

分析文本: {text}

分割原则:
- 关注主题变化和逻辑转折
- 保持段落语义完整性
- 考虑上下文连贯性""",

            """## 文本分割任务

**输入文本**: {text}

**任务**: 识别文本中的自然分割点

**输出格式**: 在分割点插入"[SPLIT]"

**质量标准**:
1. 主题一致性: 同一段落内容相关
2. 转换自然性: 分割点符合逻辑
3. 信息完整性: 避免截断关键信息"""
        ]
    
    def _update_best_individual(self) -> None:
        """更新最佳个体"""
        if not self.population.individuals:
            return
        
        current_best = max(self.population.individuals, key=lambda x: x.fitness)
        
        if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
            self.best_individual = current_best
            self.stagnation_count = 0  # 重置停滞计数
            print(f"🌟 发现新的最佳个体! 适应度: {current_best.fitness:.4f}")
        else:
            self.stagnation_count += 1
    
    def _calculate_generation_stats(self, elite_count: int, new_individuals: int) -> GenerationStats:
        """计算代统计信息"""
        if not self.population.individuals:
            return GenerationStats(
                generation=self.current_generation,
                best_fitness=0.0, avg_fitness=0.0, min_fitness=0.0,
                fitness_std=0.0, population_size=0, diversity_score=0.0,
                elite_count=elite_count, new_individuals=new_individuals,
                timestamp=time.time()
            )
        
        fitnesses = [ind.fitness for ind in self.population.individuals]
        avg_fitness = sum(fitnesses) / len(fitnesses)
        fitness_std = (sum((f - avg_fitness)**2 for f in fitnesses) / len(fitnesses))**0.5
        
        # 计算多样性分数（基于prompt的差异性）
        diversity_score = self._calculate_diversity_score()
        
        return GenerationStats(
            generation=self.current_generation,
            best_fitness=max(fitnesses),
            avg_fitness=avg_fitness,
            min_fitness=min(fitnesses),
            fitness_std=fitness_std,
            population_size=len(fitnesses),
            diversity_score=diversity_score,
            elite_count=elite_count,
            new_individuals=new_individuals,
            timestamp=time.time()
        )
    
    def _calculate_diversity_score(self) -> float:
        """计算种群多样性分数"""
        if len(self.population.individuals) < 2:
            return 0.0
        
        # 基于prompt长度和token使用的多样性
        lengths = [len(ind.prompt) for ind in self.population.individuals]
        tokens = [ind.token_count for ind in self.population.individuals]
        
        length_std = (sum((l - sum(lengths)/len(lengths))**2 for l in lengths) / len(lengths))**0.5
        token_std = (sum((t - sum(tokens)/len(tokens))**2 for t in tokens) / len(tokens))**0.5
        
        # 归一化到[0,1]范围
        length_diversity = min(1.0, length_std / 100)
        token_diversity = min(1.0, token_std / 50)
        
        return (length_diversity + token_diversity) / 2
    
    def _check_convergence(self) -> bool:
        """检查收敛条件"""
        if len(self.evolution_history) < 5:
            return False
        
        # 检查最近5代的适应度改进
        recent_bests = [h.best_fitness for h in self.evolution_history[-5:]]
        improvement = recent_bests[-1] - recent_bests[0]
        
        return improvement < self.config.convergence_threshold
    
    def _check_stagnation(self, stats: GenerationStats) -> None:
        """检查停滞状态"""
        if len(self.evolution_history) >= 2:
            prev_best = self.evolution_history[-2].best_fitness
            if stats.best_fitness <= prev_best + self.config.convergence_threshold:
                self.stagnation_count += 1
            else:
                self.stagnation_count = 0
    
    def _finalize_evolution(self) -> None:
        """完成进化实验"""
        # 更新实验记录
        self.experiment_log.update({
            "end_time": time.time(),
            "total_time": time.time() - self.start_time,
            "final_generation": self.current_generation,
            "best_individual": self.best_individual.to_dict() if self.best_individual else None,
            "convergence_info": {
                "converged": self._check_convergence(),
                "stagnation_count": self.stagnation_count,
                "final_diversity": self.evolution_history[-1].diversity_score if self.evolution_history else 0.0
            },
            "population_stats": self.population.get_statistics()
        })
    
    def save_experiment_report(self, filepath: str) -> None:
        """保存实验报告"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_log, f, indent=2, ensure_ascii=False, default=str)
        print(f"📊 实验报告已保存至: {filepath}")
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """获取进化过程摘要"""
        if not self.evolution_history:
            return {"status": "未开始"}
        
        return {
            "总代数": self.current_generation,
            "最佳适应度": self.best_individual.fitness if self.best_individual else 0.0,
            "初始适应度": self.evolution_history[0].best_fitness,
            "适应度提升": (self.best_individual.fitness - self.evolution_history[0].best_fitness) if self.best_individual else 0.0,
            "收敛状态": "已收敛" if self._check_convergence() else "未收敛",
            "停滞代数": self.stagnation_count,
            "最终多样性": self.evolution_history[-1].diversity_score,
            "总用时": time.time() - self.start_time
        }


if __name__ == "__main__":
    # 简单测试
    print("🧪 测试进化引擎")
    
    from .fitness import SimpleFitnessEvaluator
    
    # 创建测试样本
    test_sample = SegmentationSample(
        text="人工智能发展迅速。机器学习是核心技术。深度学习取得突破。",
        ground_truth_boundaries=[9, 20]
    )
    
    # 创建评估器和引擎
    evaluator = SimpleFitnessEvaluator()
    config = EvolutionConfig(population_size=5, max_generations=3)
    engine = PromptEvolutionEngine(evaluator, config)
    
    # 运行简短的进化测试
    best = engine.run_evolution([test_sample])
    
    print(f"最佳prompt: {best.prompt[:100]}...")
    print("✅ 进化引擎测试完成")