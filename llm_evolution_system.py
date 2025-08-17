#!/usr/bin/env python3
"""
LLM驱动的进化算法系统
使用大语言模型来进化出最优的"DNA"解决方案

核心思想：
1. DNA编码：将问题参数编码为"基因"字符串
2. LLM变异：使用LLM理解语义并生成智能变异
3. 适应度评估：多维度评估解决方案质量
4. 精英选择：保留最优个体并传承优秀基因
"""

import json
import time
import random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import litellm
from pathlib import Path


@dataclass
class Individual:
    """个体：包含DNA编码和适应度信息"""
    dna: str                    # DNA编码字符串
    fitness: float = 0.0        # 适应度分数
    generation: int = 0         # 所属代数
    parent_ids: List[str] = None  # 父代ID
    mutation_history: List[str] = None  # 变异历史
    
    def __post_init__(self):
        if self.parent_ids is None:
            self.parent_ids = []
        if self.mutation_history is None:
            self.mutation_history = []
    
    @property
    def id(self) -> str:
        """生成唯一ID"""
        return f"G{self.generation}_{hash(self.dna) % 10000:04d}"


class DNA:
    """DNA编码解码系统"""
    
    @staticmethod
    def encode_parameters(params: Dict[str, Any]) -> str:
        """将参数编码为DNA字符串"""
        # 使用JSON格式作为基础编码
        base_encoding = json.dumps(params, sort_keys=True, ensure_ascii=False)
        
        # 添加校验和确保完整性
        checksum = hash(base_encoding) % 10000
        dna = f"{base_encoding}|CHK:{checksum}"
        
        return dna
    
    @staticmethod
    def decode_parameters(dna: str) -> Dict[str, Any]:
        """将DNA字符串解码为参数"""
        try:
            if "|CHK:" in dna:
                base_part, checksum_part = dna.rsplit("|CHK:", 1)
                expected_checksum = int(checksum_part)
                actual_checksum = hash(base_part) % 10000
                
                if expected_checksum != actual_checksum:
                    raise ValueError(f"DNA校验失败: {expected_checksum} != {actual_checksum}")
                
                return json.loads(base_part)
            else:
                return json.loads(dna)
        except Exception as e:
            raise ValueError(f"DNA解码失败: {e}")
    
    @staticmethod
    def validate_dna(dna: str) -> bool:
        """验证DNA的有效性"""
        try:
            DNA.decode_parameters(dna)
            return True
        except:
            return False


class FitnessEvaluator(ABC):
    """适应度评估器抽象基类"""
    
    @abstractmethod
    def evaluate(self, individual: Individual) -> float:
        """评估个体的适应度"""
        pass


class LLMEvolutionEngine:
    """LLM驱动的进化引擎"""
    
    def __init__(self, 
                 fitness_evaluator: FitnessEvaluator,
                 population_size: int = 20,
                 elite_ratio: float = 0.2,
                 mutation_rate: float = 0.8,
                 crossover_rate: float = 0.6,
                 max_generations: int = 50,
                 model_name: str = "gpt-4o-mini"):
        
        self.fitness_evaluator = fitness_evaluator
        self.population_size = population_size
        self.elite_count = max(1, int(population_size * elite_ratio))
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.model_name = model_name
        
        self.population: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        self.evolution_history: List[Dict] = []
    
    def initialize_population(self, seed_params: List[Dict[str, Any]]) -> None:
        """初始化种群"""
        print(f"🧬 初始化种群 (大小: {self.population_size})")
        
        self.population = []
        
        # 使用种子参数创建初始个体
        for i, params in enumerate(seed_params[:self.population_size]):
            dna = DNA.encode_parameters(params)
            individual = Individual(dna=dna, generation=0)
            individual.fitness = self.fitness_evaluator.evaluate(individual)
            self.population.append(individual)
            print(f"  个体 {i+1}: 适应度 = {individual.fitness:.3f}")
        
        # 如果种子不足，使用LLM生成更多个体
        while len(self.population) < self.population_size:
            new_individual = self._llm_generate_individual()
            if new_individual:
                new_individual.fitness = self.fitness_evaluator.evaluate(new_individual)
                self.population.append(new_individual)
                print(f"  LLM生成个体: 适应度 = {new_individual.fitness:.3f}")
        
        self._update_best_individual()
        print(f"✅ 初始种群完成，最佳适应度: {self.best_individual.fitness:.3f}")
    
    def evolve(self) -> Individual:
        """执行进化过程"""
        print(f"\n🚀 开始进化过程 (最大代数: {self.max_generations})")
        
        for generation in range(1, self.max_generations + 1):
            self.generation = generation
            print(f"\n=== 第 {generation} 代 ===")
            
            # 选择精英个体
            elite_individuals = self._select_elite()
            
            # 生成新一代
            new_population = elite_individuals.copy()
            
            while len(new_population) < self.population_size:
                if random.random() < self.crossover_rate:
                    # 交叉操作
                    parent1, parent2 = self._select_parents()
                    child = self._llm_crossover(parent1, parent2)
                else:
                    # 变异操作
                    parent = self._select_parent()
                    child = self._llm_mutate(parent)
                
                if child:
                    child.generation = generation
                    child.fitness = self.fitness_evaluator.evaluate(child)
                    new_population.append(child)
            
            self.population = new_population[:self.population_size]
            self._update_best_individual()
            
            # 记录进化历史
            generation_stats = self._calculate_generation_stats()
            self.evolution_history.append(generation_stats)
            
            print(f"  最佳适应度: {self.best_individual.fitness:.3f}")
            print(f"  平均适应度: {generation_stats['avg_fitness']:.3f}")
            print(f"  适应度提升: {generation_stats['fitness_improvement']:.3f}")
            
            # 检查收敛条件
            if self._check_convergence():
                print(f"🎯 在第 {generation} 代达到收敛")
                break
        
        print(f"\n🏆 进化完成！最终最佳适应度: {self.best_individual.fitness:.3f}")
        return self.best_individual
    
    def _llm_generate_individual(self) -> Optional[Individual]:
        """使用LLM生成新个体"""
        prompt = f"""
作为进化算法专家，请生成一个新的参数配置用于优化。

当前最佳配置示例：
{DNA.decode_parameters(self.best_individual.dna) if self.best_individual else "无"}

请生成一个新的参数配置，以JSON格式返回：
{{
    "param1": value1,
    "param2": value2,
    ...
}}

要求：
1. 参数应该是合理的数值或字符串
2. 考虑参数间的相互关系
3. 探索新的参数组合空间
"""
        
        try:
            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8
            )
            
            content = response.choices[0].message.content.strip()
            
            # 提取JSON部分
            if "```json" in content:
                json_part = content.split("```json")[1].split("```")[0].strip()
            elif "{" in content and "}" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                json_part = content[start:end]
            else:
                return None
            
            params = json.loads(json_part)
            dna = DNA.encode_parameters(params)
            
            return Individual(dna=dna, generation=self.generation)
            
        except Exception as e:
            print(f"  ⚠️ LLM生成个体失败: {e}")
            return None
    
    def _llm_mutate(self, parent: Individual) -> Optional[Individual]:
        """使用LLM执行智能变异"""
        parent_params = DNA.decode_parameters(parent.dna)
        
        prompt = f"""
作为进化算法专家，请对以下参数配置进行智能变异：

原始配置：
{json.dumps(parent_params, indent=2, ensure_ascii=False)}

适应度分数：{parent.fitness:.3f}

请生成一个变异后的配置，要求：
1. 在原配置基础上进行合理的调整
2. 可以微调数值、添加/删除参数、改变策略等
3. 保持参数的合理性和一致性
4. 探索潜在的改进方向

以JSON格式返回变异后的配置：
"""
        
        try:
            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            
            content = response.choices[0].message.content.strip()
            
            # 提取JSON部分
            if "```json" in content:
                json_part = content.split("```json")[1].split("```")[0].strip()
            elif "{" in content and "}" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                json_part = content[start:end]
            else:
                return None
            
            new_params = json.loads(json_part)
            new_dna = DNA.encode_parameters(new_params)
            
            individual = Individual(
                dna=new_dna,
                generation=self.generation,
                parent_ids=[parent.id],
                mutation_history=[f"LLM变异自 {parent.id}"]
            )
            
            return individual
            
        except Exception as e:
            print(f"  ⚠️ LLM变异失败: {e}")
            return None
    
    def _llm_crossover(self, parent1: Individual, parent2: Individual) -> Optional[Individual]:
        """使用LLM执行智能交叉"""
        params1 = DNA.decode_parameters(parent1.dna)
        params2 = DNA.decode_parameters(parent2.dna)
        
        prompt = f"""
作为进化算法专家，请对以下两个参数配置进行智能交叉融合：

父代1配置（适应度: {parent1.fitness:.3f}）：
{json.dumps(params1, indent=2, ensure_ascii=False)}

父代2配置（适应度: {parent2.fitness:.3f}）：
{json.dumps(params2, indent=2, ensure_ascii=False)}

请生成一个融合后的子代配置，要求：
1. 结合两个父代的优秀特征
2. 可以采用加权平均、选择性继承、创新组合等策略
3. 保持参数的合理性和一致性
4. 追求比父代更优的性能

以JSON格式返回交叉后的配置：
"""
        
        try:
            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6
            )
            
            content = response.choices[0].message.content.strip()
            
            # 提取JSON部分
            if "```json" in content:
                json_part = content.split("```json")[1].split("```")[0].strip()
            elif "{" in content and "}" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                json_part = content[start:end]
            else:
                return None
            
            new_params = json.loads(json_part)
            new_dna = DNA.encode_parameters(new_params)
            
            individual = Individual(
                dna=new_dna,
                generation=self.generation,
                parent_ids=[parent1.id, parent2.id],
                mutation_history=[f"LLM交叉自 {parent1.id} × {parent2.id}"]
            )
            
            return individual
            
        except Exception as e:
            print(f"  ⚠️ LLM交叉失败: {e}")
            return None
    
    def _select_elite(self) -> List[Individual]:
        """选择精英个体"""
        sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        return sorted_population[:self.elite_count]
    
    def _select_parents(self) -> Tuple[Individual, Individual]:
        """选择两个父代个体"""
        # 轮盘赌选择
        total_fitness = sum(ind.fitness for ind in self.population)
        
        def select_one():
            r = random.uniform(0, total_fitness)
            cumsum = 0
            for ind in self.population:
                cumsum += ind.fitness
                if cumsum >= r:
                    return ind
            return self.population[-1]
        
        parent1 = select_one()
        parent2 = select_one()
        while parent2 == parent1:
            parent2 = select_one()
        
        return parent1, parent2
    
    def _select_parent(self) -> Individual:
        """选择一个父代个体"""
        # 锦标赛选择
        tournament_size = min(3, len(self.population))
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def _update_best_individual(self):
        """更新最佳个体"""
        current_best = max(self.population, key=lambda x: x.fitness)
        if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
            self.best_individual = current_best
    
    def _calculate_generation_stats(self) -> Dict:
        """计算当代统计信息"""
        fitnesses = [ind.fitness for ind in self.population]
        
        stats = {
            "generation": self.generation,
            "best_fitness": max(fitnesses),
            "avg_fitness": sum(fitnesses) / len(fitnesses),
            "min_fitness": min(fitnesses),
            "fitness_std": (sum((f - sum(fitnesses)/len(fitnesses))**2 for f in fitnesses) / len(fitnesses))**0.5,
            "population_size": len(self.population)
        }
        
        if len(self.evolution_history) > 0:
            prev_best = self.evolution_history[-1]["best_fitness"]
            stats["fitness_improvement"] = stats["best_fitness"] - prev_best
        else:
            stats["fitness_improvement"] = 0.0
        
        return stats
    
    def _check_convergence(self) -> bool:
        """检查收敛条件"""
        if len(self.evolution_history) < 5:
            return False
        
        # 检查最近5代的适应度改进
        recent_improvements = [h["fitness_improvement"] for h in self.evolution_history[-5:]]
        avg_improvement = sum(recent_improvements) / len(recent_improvements)
        
        return avg_improvement < 0.001  # 改进幅度小于0.001则认为收敛
    
    def save_evolution_report(self, filepath: str):
        """保存进化报告"""
        report = {
            "evolution_config": {
                "population_size": self.population_size,
                "elite_count": self.elite_count,
                "mutation_rate": self.mutation_rate,
                "crossover_rate": self.crossover_rate,
                "max_generations": self.max_generations,
                "model_name": self.model_name
            },
            "best_individual": {
                "id": self.best_individual.id,
                "dna": self.best_individual.dna,
                "parameters": DNA.decode_parameters(self.best_individual.dna),
                "fitness": self.best_individual.fitness,
                "generation": self.best_individual.generation,
                "parent_ids": self.best_individual.parent_ids,
                "mutation_history": self.best_individual.mutation_history
            },
            "evolution_history": self.evolution_history,
            "final_population": [
                {
                    "id": ind.id,
                    "fitness": ind.fitness,
                    "parameters": DNA.decode_parameters(ind.dna)
                }
                for ind in sorted(self.population, key=lambda x: x.fitness, reverse=True)
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📊 进化报告已保存至: {filepath}")


# 示例适应度评估器：文本压缩优化
class CompressionFitnessEvaluator(FitnessEvaluator):
    """文本压缩参数适应度评估器"""
    
    def __init__(self, test_texts: List[str]):
        self.test_texts = test_texts
    
    def evaluate(self, individual: Individual) -> float:
        """评估压缩参数的适应度"""
        try:
            params = DNA.decode_parameters(individual.dna)
            
            # 模拟文本压缩评估
            total_score = 0.0
            
            for text in self.test_texts:
                # 根据参数计算压缩质量分数
                tau_buffer = params.get("tau_buffer", 100)
                tau_slice = params.get("tau_slice", 80)
                batch_size = params.get("batch_size", 4)
                strategy = params.get("strategy", "semantic")
                
                # 模拟评估指标
                compression_ratio = min(1.0, len(text) / (tau_buffer + tau_slice))
                semantic_quality = 0.8 if strategy == "semantic" else 0.6
                processing_efficiency = min(1.0, 10.0 / batch_size)
                
                # 综合评分
                text_score = (
                    compression_ratio * 0.4 +
                    semantic_quality * 0.4 +
                    processing_efficiency * 0.2
                )
                
                total_score += text_score
            
            # 添加参数合理性惩罚
            if tau_buffer < 50 or tau_buffer > 500:
                total_score *= 0.8
            if tau_slice < 30 or tau_slice > 200:
                total_score *= 0.8
            if batch_size < 1 or batch_size > 20:
                total_score *= 0.8
            
            return total_score / len(self.test_texts)
            
        except Exception as e:
            print(f"  ⚠️ 适应度评估失败: {e}")
            return 0.0


if __name__ == "__main__":
    # 演示用法
    print("🧬 LLM进化算法系统演示")
    
    # 创建测试文本
    test_texts = [
        "这是一个测试文本，用于评估文本压缩算法的性能。",
        "线性代数是数学的一个重要分支，研究向量空间和线性变换。",
        "人工智能技术正在快速发展，深度学习模型在各个领域都取得了显著的成果。"
    ]
    
    # 创建适应度评估器
    fitness_evaluator = CompressionFitnessEvaluator(test_texts)
    
    # 创建进化引擎
    evolution_engine = LLMEvolutionEngine(
        fitness_evaluator=fitness_evaluator,
        population_size=10,
        max_generations=5  # 演示用较小的代数
    )
    
    # 初始种子参数
    seed_params = [
        {"tau_buffer": 100, "tau_slice": 80, "batch_size": 4, "strategy": "semantic"},
        {"tau_buffer": 150, "tau_slice": 60, "batch_size": 6, "strategy": "elevation"},
        {"tau_buffer": 80, "tau_slice": 100, "batch_size": 3, "strategy": "conceptual"}
    ]
    
    # 初始化种群
    evolution_engine.initialize_population(seed_params)
    
    # 执行进化
    best_solution = evolution_engine.evolve()
    
    # 保存结果
    evolution_engine.save_evolution_report("evolution_report.json")
    
    print(f"\n🏆 最优解:")
    print(f"  DNA: {best_solution.dna}")
    print(f"  参数: {DNA.decode_parameters(best_solution.dna)}")
    print(f"  适应度: {best_solution.fitness:.3f}")