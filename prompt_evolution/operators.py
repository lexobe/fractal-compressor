#!/usr/bin/env python3
"""
LLM Prompt进化算法遗传操作
实现变异、交叉和选择操作
"""

import random
import time
from typing import List, Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod

from .core import PromptIndividual, LLMInterface, TokenCounter, PromptValidator


class MutationOperator(ABC):
    """变异操作抽象基类"""
    
    @abstractmethod
    def mutate(self, parent: PromptIndividual, mutation_strength: float = 0.3) -> Optional[PromptIndividual]:
        """执行变异操作"""
        pass


class LLMMutationOperator(MutationOperator):
    """LLM驱动的智能变异操作"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = LLMInterface(model=model, temperature=0.7)
        self.mutation_strategies = {
            "style_change": "改变语言风格和表达方式",
            "strategy_evolution": "演化分析策略和方法",
            "format_optimization": "优化输出格式和标记方式",
            "precision_enhancement": "增强精确度和细节要求",
            "efficiency_improvement": "提高效率和简洁性",
            "robustness_boost": "增强鲁棒性和适应性"
        }
    
    def mutate(self, parent: PromptIndividual, mutation_strength: float = 0.3) -> Optional[PromptIndividual]:
        """执行LLM驱动的智能变异"""
        try:
            # 根据变异强度选择策略
            if mutation_strength > 0.7:
                strategy = "major_reconstruction"
            elif mutation_strength > 0.4:
                strategy = "moderate_adjustment"
            else:
                strategy = "minor_refinement"
            
            mutation_prompt = self._create_mutation_prompt(parent.prompt, mutation_strength, strategy)
            mutated_text = self.llm.generate(mutation_prompt, max_tokens=400)
            
            # 提取变异后的prompt
            new_prompt = self._extract_prompt_from_response(mutated_text)
            
            if not new_prompt or new_prompt == parent.prompt:
                return None
            
            # 验证新prompt
            is_valid, reason = PromptValidator.validate_prompt(new_prompt)
            if not is_valid:
                print(f"变异结果无效: {reason}")
                return None
            
            # 创建新个体
            child = PromptIndividual(
                prompt=new_prompt,
                generation=parent.generation + 1,
                parent_ids=[parent.id],
                evolution_history=parent.evolution_history + [f"LLM变异自 {parent.id}"]
            )
            child.token_count = TokenCounter.count_tokens(new_prompt)
            
            return child
            
        except Exception as e:
            print(f"变异操作失败: {e}")
            return None
    
    def _create_mutation_prompt(self, original_prompt: str, strength: float, strategy: str) -> str:
        """创建变异指令prompt"""
        
        strength_desc = {
            "major_reconstruction": "大幅重构和创新",
            "moderate_adjustment": "适度调整和优化", 
            "minor_refinement": "细微改进和完善"
        }
        
        return f"""作为prompt工程专家，请对以下文本分割prompt进行{strength_desc[strategy]}：

原始prompt:
{original_prompt}

变异要求：
1. 保持核心功能（文本分割）不变
2. 变异强度：{strength:.1f} ({strength_desc[strategy]})
3. 确保变异后prompt在300 token以内
4. 输出格式要求在分割点插入"[SPLIT]"标记
5. 可以调整表达方式、指导策略、质量标准等

变异方向：
- 调整任务描述的角度和方式
- 修改质量标准或评估维度
- 改变输出格式说明
- 优化语言表达的清晰度
- 加入新的分析思路

请直接输出变异后的prompt，不要添加解释："""
    
    def _extract_prompt_from_response(self, response: str) -> str:
        """从LLM响应中提取prompt"""
        # 移除常见的前缀和后缀
        response = response.strip()
        
        # 如果包含代码块标记，提取内容
        if "```" in response:
            parts = response.split("```")
            if len(parts) >= 3:
                response = parts[1].strip()
        
        # 移除常见的说明性文字
        prefixes_to_remove = [
            "变异后的prompt：",
            "新的prompt：",
            "优化后的prompt：",
            "以下是变异后的prompt：",
            "变异结果："
        ]
        
        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        return response
    
    def directed_mutation(self, parent: PromptIndividual, mutation_type: str) -> Optional[PromptIndividual]:
        """定向变异操作"""
        if mutation_type not in self.mutation_strategies:
            return self.mutate(parent, 0.5)
        
        direction = self.mutation_strategies[mutation_type]
        
        mutation_prompt = f"""针对以下文本分割prompt，进行{direction}的定向优化：

原始prompt: 
{parent.prompt}

优化方向: {direction}

要求：
1. 重点在{direction}方面进行改进
2. 保持其他功能基本稳定
3. 确保token数量在300以内
4. 提升整体效果

请直接输出优化后的prompt："""
        
        try:
            response = self.llm.generate(mutation_prompt)
            new_prompt = self._extract_prompt_from_response(response)
            
            if new_prompt and new_prompt != parent.prompt:
                is_valid, _ = PromptValidator.validate_prompt(new_prompt)
                if is_valid:
                    child = PromptIndividual(
                        prompt=new_prompt,
                        generation=parent.generation + 1,
                        parent_ids=[parent.id],
                        evolution_history=parent.evolution_history + [f"定向变异({mutation_type})自 {parent.id}"]
                    )
                    child.token_count = TokenCounter.count_tokens(new_prompt)
                    return child
            
            return None
            
        except Exception as e:
            print(f"定向变异失败: {e}")
            return None


class CrossoverOperator(ABC):
    """交叉操作抽象基类"""
    
    @abstractmethod
    def crossover(self, parent1: PromptIndividual, parent2: PromptIndividual) -> Optional[PromptIndividual]:
        """执行交叉操作"""
        pass


class LLMCrossoverOperator(CrossoverOperator):
    """LLM驱动的智能交叉操作"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = LLMInterface(model=model, temperature=0.6)
    
    def crossover(self, parent1: PromptIndividual, parent2: PromptIndividual) -> Optional[PromptIndividual]:
        """执行语义融合交叉"""
        try:
            crossover_prompt = f"""请融合以下两个文本分割prompt的优点，创造一个新的更优prompt：

Parent 1 (适应度: {parent1.fitness:.3f}):
{parent1.prompt}

Parent 2 (适应度: {parent2.fitness:.3f}):
{parent2.prompt}

融合要求：
1. 提取两个prompt的核心优势和有效策略
2. 创新性结合不同的方法和思路
3. 避免简单的文本拼接
4. 生成语义连贯、逻辑清晰的新prompt
5. 控制在300 token以内
6. 追求比父代更优的效果

融合策略：
- 结合不同的分析角度
- 融合各自的质量标准
- 整合有效的指导方法
- 优化整体结构和表达

请直接输出融合后的prompt："""
            
            response = self.llm.generate(crossover_prompt, max_tokens=400)
            new_prompt = self._extract_prompt_from_response(response)
            
            if not new_prompt:
                return None
            
            # 验证新prompt
            is_valid, reason = PromptValidator.validate_prompt(new_prompt)
            if not is_valid:
                print(f"交叉结果无效: {reason}")
                return None
            
            # 创建子代个体
            child = PromptIndividual(
                prompt=new_prompt,
                generation=max(parent1.generation, parent2.generation) + 1,
                parent_ids=[parent1.id, parent2.id],
                evolution_history=parent1.evolution_history + parent2.evolution_history + 
                               [f"LLM交叉自 {parent1.id} × {parent2.id}"]
            )
            child.token_count = TokenCounter.count_tokens(new_prompt)
            
            return child
            
        except Exception as e:
            print(f"交叉操作失败: {e}")
            return None
    
    def _extract_prompt_from_response(self, response: str) -> str:
        """从LLM响应中提取prompt"""
        response = response.strip()
        
        # 处理代码块
        if "```" in response:
            parts = response.split("```")
            if len(parts) >= 3:
                response = parts[1].strip()
        
        # 移除说明性前缀
        prefixes = ["融合后的prompt：", "新prompt：", "结果：", "融合结果："]
        for prefix in prefixes:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        return response
    
    def modular_crossover(self, parent1: PromptIndividual, parent2: PromptIndividual) -> Optional[PromptIndividual]:
        """模块化交叉：分别提取和组合不同模块"""
        try:
            # 分析两个prompt的结构模块
            modules1 = self._analyze_prompt_modules(parent1.prompt)
            modules2 = self._analyze_prompt_modules(parent2.prompt)
            
            crossover_prompt = f"""基于以下prompt模块分析，创造一个新的组合prompt：

Parent 1 模块分析：
- 任务描述: {modules1.get('task_description', '无')}
- 分析方法: {modules1.get('analysis_method', '无')}
- 输出格式: {modules1.get('output_format', '无')}
- 质量标准: {modules1.get('quality_criteria', '无')}

Parent 2 模块分析：
- 任务描述: {modules2.get('task_description', '无')}
- 分析方法: {modules2.get('analysis_method', '无')}
- 输出格式: {modules2.get('output_format', '无')}
- 质量标准: {modules2.get('quality_criteria', '无')}

请选择每个模块的最佳版本并创新性组合，生成新prompt："""
            
            response = self.llm.generate(crossover_prompt)
            new_prompt = self._extract_prompt_from_response(response)
            
            if new_prompt:
                is_valid, _ = PromptValidator.validate_prompt(new_prompt)
                if is_valid:
                    child = PromptIndividual(
                        prompt=new_prompt,
                        generation=max(parent1.generation, parent2.generation) + 1,
                        parent_ids=[parent1.id, parent2.id],
                        evolution_history=[f"模块化交叉自 {parent1.id} × {parent2.id}"]
                    )
                    child.token_count = TokenCounter.count_tokens(new_prompt)
                    return child
            
            return None
            
        except Exception as e:
            print(f"模块化交叉失败: {e}")
            return None
    
    def _analyze_prompt_modules(self, prompt: str) -> Dict[str, str]:
        """分析prompt的模块结构"""
        modules = {}
        
        # 简单的启发式分析
        lines = prompt.split('\n')
        
        current_module = "task_description"
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检测模块分界标志
            if any(keyword in line.lower() for keyword in ['要求', 'requirement', '标准', 'criteria']):
                if current_content:
                    modules[current_module] = '\n'.join(current_content)
                current_module = "quality_criteria"
                current_content = [line]
            elif any(keyword in line.lower() for keyword in ['格式', 'format', '输出', 'output']):
                if current_content:
                    modules[current_module] = '\n'.join(current_content)
                current_module = "output_format"
                current_content = [line]
            elif any(keyword in line.lower() for keyword in ['方法', 'method', '策略', 'strategy']):
                if current_content:
                    modules[current_module] = '\n'.join(current_content)
                current_module = "analysis_method"
                current_content = [line]
            else:
                current_content.append(line)
        
        # 添加最后一个模块
        if current_content:
            modules[current_module] = '\n'.join(current_content)
        
        return modules


class SelectionOperator(ABC):
    """选择操作抽象基类"""
    
    @abstractmethod
    def select(self, population: List[PromptIndividual], count: int) -> List[PromptIndividual]:
        """执行选择操作"""
        pass


class EliteSelection(SelectionOperator):
    """精英选择"""
    
    def select(self, population: List[PromptIndividual], count: int) -> List[PromptIndividual]:
        """选择适应度最高的个体"""
        sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True)
        return sorted_population[:count]


class TournamentSelection(SelectionOperator):
    """锦标赛选择"""
    
    def __init__(self, tournament_size: int = 3):
        self.tournament_size = tournament_size
    
    def select(self, population: List[PromptIndividual], count: int) -> List[PromptIndividual]:
        """锦标赛选择"""
        selected = []
        
        for _ in range(count):
            # 随机选择锦标赛参与者
            tournament = random.sample(
                population, 
                min(self.tournament_size, len(population))
            )
            
            # 选择适应度最高的获胜者
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(winner)
        
        return selected


class RouletteWheelSelection(SelectionOperator):
    """轮盘赌选择"""
    
    def select(self, population: List[PromptIndividual], count: int) -> List[PromptIndividual]:
        """轮盘赌选择"""
        if not population:
            return []
        
        # 计算总适应度
        total_fitness = sum(ind.fitness for ind in population)
        if total_fitness == 0:
            # 如果所有适应度都是0，随机选择
            return random.sample(population, min(count, len(population)))
        
        selected = []
        for _ in range(count):
            # 生成随机数
            r = random.uniform(0, total_fitness)
            
            # 找到对应的个体
            cumsum = 0
            for individual in population:
                cumsum += individual.fitness
                if cumsum >= r:
                    selected.append(individual)
                    break
            else:
                # 如果没有找到，选择最后一个
                selected.append(population[-1])
        
        return selected


class EvolutionaryOperators:
    """进化操作管理器"""
    
    def __init__(self, 
                 mutation_operator: MutationOperator = None,
                 crossover_operator: CrossoverOperator = None,
                 selection_operator: SelectionOperator = None):
        
        self.mutation_op = mutation_operator or LLMMutationOperator()
        self.crossover_op = crossover_operator or LLMCrossoverOperator()
        self.selection_op = selection_operator or TournamentSelection()
    
    def mutate(self, parent: PromptIndividual, strength: float = 0.3) -> Optional[PromptIndividual]:
        """执行变异"""
        return self.mutation_op.mutate(parent, strength)
    
    def crossover(self, parent1: PromptIndividual, parent2: PromptIndividual) -> Optional[PromptIndividual]:
        """执行交叉"""
        return self.crossover_op.crossover(parent1, parent2)
    
    def select(self, population: List[PromptIndividual], count: int) -> List[PromptIndividual]:
        """执行选择"""
        return self.selection_op.select(population, count)
    
    def select_parents(self, population: List[PromptIndividual]) -> Tuple[PromptIndividual, PromptIndividual]:
        """选择两个父代个体"""
        parents = self.selection_op.select(population, 2)
        if len(parents) >= 2:
            return parents[0], parents[1]
        elif len(parents) == 1:
            return parents[0], parents[0]
        else:
            # 随机选择
            return random.choice(population), random.choice(population)


if __name__ == "__main__":
    # 测试进化操作
    print("🧬 测试进化操作")
    
    # 创建测试个体
    parent1 = PromptIndividual(
        prompt="请将文本按主题分割，在分割点插入[SPLIT]标记。",
        fitness=0.7
    )
    
    parent2 = PromptIndividual(
        prompt="作为文本分析专家，请识别语义边界并插入[SPLIT]。要求保持主题完整性。",
        fitness=0.8
    )
    
    # 测试变异
    mutation_op = LLMMutationOperator()
    child1 = mutation_op.mutate(parent1, 0.5)
    if child1:
        print(f"变异成功: {child1.prompt[:50]}...")
    
    # 测试交叉
    crossover_op = LLMCrossoverOperator()
    child2 = crossover_op.crossover(parent1, parent2)
    if child2:
        print(f"交叉成功: {child2.prompt[:50]}...")
    
    # 测试选择
    population = [parent1, parent2]
    selection_op = TournamentSelection()
    selected = selection_op.select(population, 1)
    print(f"选择结果: {len(selected)} 个个体")
    
    print("✅ 进化操作测试完成")