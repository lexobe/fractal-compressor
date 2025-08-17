# LLM Prompt进化算法设计文档

## 项目概述

本文档详细描述了一个基于LLM的文本分割prompt进化优化系统的设计方案。该系统使用进化算法自动优化文本分割任务的prompt，以获得最佳的分割效果。

### 核心理念
- **DNA = Prompt**: 每个个体的"基因"就是一段用于文本分割的prompt文本
- **长度约束**: 每个prompt限制在300 token以内
- **有监督进化**: 基于标准数据集进行适应度评估
- **智能变异**: 使用LLM进行语义理解的prompt变异和交叉

---

## 系统架构

### 1. 整体框架

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   种群初始化     │───▶│   适应度评估     │───▶│   进化操作       │
│                │    │                │    │                │
│ • 种子prompt    │    │ • 边界准确率     │    │ • LLM变异       │
│ • 随机生成      │    │ • 语义一致性     │    │ • LLM交叉       │
│ • 多样性保证    │    │ • 颗粒度合理性   │    │ • 精英选择      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                                               │
         │                                               │
         └───────────────── 迭代收敛 ◀────────────────────┘
```

### 2. 核心组件

#### 2.1 Individual类（个体）
```python
@dataclass
class PromptIndividual:
    prompt: str                    # DNA编码（prompt文本）
    fitness: float = 0.0           # 适应度分数
    generation: int = 0            # 所属代数
    token_count: int = 0           # token计数
    parent_ids: List[str] = None   # 父代ID
    evolution_history: List[str] = None  # 进化历史
    
    @property
    def is_valid(self) -> bool:
        """检查prompt是否在300 token限制内"""
        return self.token_count <= 300
```

#### 2.2 适应度评估器
```python
class TextSegmentationFitnessEvaluator:
    def __init__(self, datasets: List[Dataset], weights: Dict[str, float]):
        self.datasets = datasets
        self.weights = weights
    
    def evaluate(self, individual: PromptIndividual) -> float:
        """综合评估prompt的适应度"""
        pass
```

---

## 数据集分析

### 1. 数据集类型

#### 1.1 CPTS数据集（中文段落主题分割）
- **格式**: JSON格式，包含段落列表和主题标注
- **标注**: 每个段落有`topic_index`，相同值表示同一主题
- **特点**: 中文文本，主题分割任务
- **样例**:
```json
{
    "id": "XIN_CMN_20040829.0158",
    "title": "中国再上台阶世界三强确立",
    "paragraph_list": [
        {
            "id": 0,
            "text": "雅典奥运会上,随着中国取代俄罗斯...",
            "topic_index": 0
        },
        {
            "id": 1,
            "text": "本届奥运会金牌榜前三名的顺序为...",
            "topic_index": 1
        }
    ]
}
```

#### 1.2 WikiSection数据集（维基百科章节分割）
- **格式**: JSON格式，包含文本和章节标注
- **标注**: `annotations`数组标记章节边界和标题
- **特点**: 英文/德文文本，层级结构分割
- **样例**:
```json
{
    "id": "https://en.wikipedia.org/wiki/San_Sebastián",
    "text": "In spite of appearances, both the Basque form...",
    "annotations": [
        {
            "class": "SectionAnnotation",
            "begin": 0,
            "length": 596,
            "sectionHeading": "Etymology",
            "sectionLabel": "city.etymology"
        }
    ]
}
```

### 2. 数据预处理策略

#### 2.1 统一格式转换
```python
@dataclass
class SegmentationSample:
    text: str                      # 原始文本
    ground_truth_boundaries: List[int]  # 标准分割边界位置
    topic_labels: List[int] = None      # 主题标签（如果有）
    metadata: Dict[str, Any] = None     # 元数据
```

#### 2.2 采样策略
- **分层采样**: 按文本长度、语言、领域分层
- **难度分级**: 简单、中等、困难三个级别
- **固定测试集**: 确保不同prompt在相同条件下比较

---

## 适应度函数设计

### 1. 综合评分公式

```
适应度 = α×边界准确率 + β×语义一致性 + γ×颗粒度合理性 + δ×效率奖励 - ε×长度惩罚

默认权重:
α = 0.45  (边界准确率 - 最重要)
β = 0.25  (语义一致性)
γ = 0.20  (颗粒度合理性)
δ = 0.05  (效率奖励)
ε = 0.05  (长度惩罚)
```

### 2. 各项指标详解

#### 2.1 边界准确率（Boundary Accuracy）

**计算方法**:
- **精确匹配**: 预测边界完全对应标准边界 → 1.0分
- **近似匹配**: 预测边界在标准边界±5字符内 → 0.8分
- **部分匹配**: 预测边界在±20字符内 → 0.3-0.6分
- **错误边界**: 超出合理范围 → 0.0分

**评估指标**:
```python
def calculate_boundary_score(predicted_boundaries, true_boundaries, text_length):
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    # 考虑边界质量的加权F1
    weighted_f1 = sum(match_quality * match_weight for match in matches) / len(matches)
    
    return weighted_f1
```

#### 2.2 语义一致性（Semantic Consistency）

**基于标注的主题一致性**:
```python
def calculate_semantic_consistency(predicted_segments, ground_truth_topics):
    consistency_scores = []
    
    for segment in predicted_segments:
        # 计算段内主题纯度
        topic_distribution = get_topic_distribution(segment, ground_truth_topics)
        purity = max(topic_distribution.values()) / sum(topic_distribution.values())
        consistency_scores.append(purity)
    
    # 整体一致性 = 平均纯度 × 主题分离度
    avg_purity = mean(consistency_scores)
    separation_score = calculate_topic_separation(predicted_segments, ground_truth_topics)
    
    return avg_purity * separation_score
```

**主题分离度计算**:
- 检查不同主题是否被正确分开
- 避免同一主题内容被错误分割
- 避免不同主题内容被错误合并

#### 2.3 颗粒度合理性（Granularity Appropriateness）

**统计分布指标**:
```python
def calculate_granularity_score(predicted_segments, ground_truth_segments):
    # 1. 分割数量合理性
    count_ratio = len(predicted_segments) / len(ground_truth_segments)
    count_score = 1.0 - abs(1.0 - count_ratio)  # 越接近1.0越好
    
    # 2. 长度分布合理性
    pred_lengths = [len(seg) for seg in predicted_segments]
    true_lengths = [len(seg) for seg in ground_truth_segments]
    
    length_similarity = calculate_distribution_similarity(pred_lengths, true_lengths)
    
    # 3. 避免极端分割
    extreme_penalty = calculate_extreme_segmentation_penalty(pred_lengths)
    
    return (count_score * 0.4 + length_similarity * 0.4 + extreme_penalty * 0.2)
```

#### 2.4 效率奖励（Efficiency Reward）

**Token使用效率**:
```python
def calculate_efficiency_reward(prompt_token_count, performance_score):
    # 在性能相当的情况下，更短的prompt获得奖励
    efficiency_ratio = min(1.0, 200 / prompt_token_count)  # 200为理想长度
    return efficiency_ratio * performance_score
```

#### 2.5 长度惩罚（Length Penalty）

**超长prompt惩罚**:
```python
def calculate_length_penalty(token_count, max_tokens=300):
    if token_count <= max_tokens:
        return 0.0
    else:
        # 超出部分按比例惩罚
        excess_ratio = (token_count - max_tokens) / max_tokens
        return min(1.0, excess_ratio * 2.0)  # 最大惩罚1.0
```

### 3. 评估流程

#### 3.1 批量评估策略
```python
def evaluate_prompt_fitness(prompt: str, test_samples: List[SegmentationSample]) -> float:
    total_score = 0.0
    valid_evaluations = 0
    
    for sample in test_samples:
        try:
            # 使用prompt对文本进行分割
            predicted_segments = apply_prompt_to_text(prompt, sample.text)
            
            # 计算各项指标
            boundary_score = calculate_boundary_score(predicted_segments, sample.ground_truth_boundaries)
            semantic_score = calculate_semantic_consistency(predicted_segments, sample.topic_labels)
            granularity_score = calculate_granularity_score(predicted_segments, sample.ground_truth_segments)
            
            # 综合评分
            sample_score = (
                0.45 * boundary_score +
                0.25 * semantic_score +
                0.20 * granularity_score
            )
            
            total_score += sample_score
            valid_evaluations += 1
            
        except Exception as e:
            # prompt导致错误，给予低分
            total_score += 0.1
            valid_evaluations += 1
    
    # 计算平均分并加入效率和长度调节
    avg_score = total_score / valid_evaluations if valid_evaluations > 0 else 0.0
    
    # 添加效率奖励和长度惩罚
    token_count = count_tokens(prompt)
    efficiency_reward = calculate_efficiency_reward(token_count, avg_score)
    length_penalty = calculate_length_penalty(token_count)
    
    final_score = avg_score + 0.05 * efficiency_reward - 0.05 * length_penalty
    
    return max(0.0, min(1.0, final_score))
```

---

## 进化操作设计

### 1. 初始化策略

#### 1.1 种子prompt设计
```python
SEED_PROMPTS = [
    # 基础分割prompt
    """请将以下文本按照主题进行分割。在每个主题转换的地方插入分割标记"[SPLIT]"。
    
    文本: {text}
    
    要求:
    1. 保持主题内容的完整性
    2. 确保逻辑转换清晰
    3. 避免过度分割""",
    
    # 语义理解prompt
    """作为文本分析专家，请识别以下文本中的语义边界，并在适当位置插入"[SPLIT]"标记。
    
    分析文本: {text}
    
    分割原则:
    - 关注主题变化和逻辑转折
    - 保持段落语义完整性
    - 考虑上下文连贯性""",
    
    # 结构化prompt
    """## 文本分割任务
    
    **输入文本**: {text}
    
    **任务**: 识别文本中的自然分割点
    
    **输出格式**: 在分割点插入"[SPLIT]"
    
    **质量标准**:
    1. 主题一致性: 同一段落内容相关
    2. 转换自然性: 分割点符合逻辑
    3. 信息完整性: 避免截断关键信息""",
]
```

#### 1.2 随机生成策略
```python
def generate_random_prompt(max_tokens: int = 300) -> str:
    """使用LLM生成随机prompt"""
    generation_prompt = f"""
    请创建一个用于文本分割的prompt，要求：
    1. 长度不超过{max_tokens} tokens
    2. 指导LLM将文本在语义边界处分割
    3. 输出格式要求在分割点插入"[SPLIT]"标记
    4. 包含清晰的任务描述和质量要求
    
    创新要求：
    - 可以使用不同的表达方式
    - 可以加入特定的分析角度
    - 可以设计独特的指导策略
    """
    
    response = llm_generate(generation_prompt)
    return extract_prompt_from_response(response)
```

### 2. 变异操作

#### 2.1 智能变异策略
```python
def llm_mutate_prompt(parent_prompt: str, mutation_strength: float = 0.3) -> str:
    """使用LLM进行智能变异"""
    
    mutation_prompt = f"""
    请对以下文本分割prompt进行变异优化，变异强度为{mutation_strength}：
    
    原始prompt:
    {parent_prompt}
    
    变异要求：
    1. 保持核心功能（文本分割）不变
    2. 可以调整表达方式、指导策略、输出格式等
    3. 变异强度{mutation_strength}: {"大幅" if mutation_strength > 0.5 else "适度" if mutation_strength > 0.2 else "微调"}
    4. 确保变异后prompt仍在300 token限制内
    5. 可以加入新的分析维度或优化策略
    
    变异方向建议：
    - 调整任务描述的角度
    - 修改质量标准或评估维度
    - 改变输出格式或标记方式
    - 加入领域专业知识
    - 优化语言表达的清晰度
    """
    
    response = llm_generate(mutation_prompt, temperature=0.7)
    mutated_prompt = extract_prompt_from_response(response)
    
    return mutated_prompt
```

#### 2.2 定向变异类型
```python
MUTATION_TYPES = {
    "style_change": "改变语言风格和表达方式",
    "strategy_evolution": "演化分析策略和方法",
    "format_optimization": "优化输出格式和标记方式",  
    "precision_enhancement": "增强精确度和细节要求",
    "efficiency_improvement": "提高效率和简洁性",
    "robustness_boost": "增强鲁棒性和适应性"
}

def directed_mutation(parent_prompt: str, mutation_type: str) -> str:
    """定向变异操作"""
    direction = MUTATION_TYPES[mutation_type]
    
    mutation_prompt = f"""
    针对以下文本分割prompt，进行{direction}的定向优化：
    
    原始prompt: {parent_prompt}
    
    优化方向: {direction}
    
    要求：
    1. 重点在{direction}方面进行改进
    2. 保持其他功能基本稳定
    3. 确保token数量在300以内
    4. 提升整体效果
    """
    
    return llm_generate(mutation_prompt)
```

### 3. 交叉操作

#### 3.1 语义融合交叉
```python
def llm_crossover_prompts(parent1: str, parent2: str) -> str:
    """LLM驱动的语义融合交叉"""
    
    crossover_prompt = f"""
    请融合以下两个文本分割prompt的优点，创造一个新的更优prompt：
    
    Parent 1:
    {parent1}
    
    Parent 2: 
    {parent2}
    
    融合要求：
    1. 提取两个prompt的核心优势
    2. 创新性结合不同的方法和策略
    3. 避免简单的文本拼接
    4. 生成语义连贯、逻辑清晰的新prompt
    5. 控制在300 token以内
    6. 追求比父代更优的效果
    
    融合策略：
    - 结合不同的分析角度
    - 融合各自的质量标准
    - 整合有效的指导方法
    - 优化整体结构和表达
    """
    
    response = llm_generate(crossover_prompt, temperature=0.6)
    return extract_prompt_from_response(response)
```

#### 3.2 模块化交叉
```python
def modular_crossover(parent1: str, parent2: str) -> str:
    """模块化交叉：分别提取和组合不同模块"""
    
    # 分析两个prompt的结构模块
    modules1 = analyze_prompt_modules(parent1)
    modules2 = analyze_prompt_modules(parent2)
    
    crossover_prompt = f"""
    基于以下prompt模块分析，创造一个新的组合prompt：
    
    Parent 1 模块：
    - 任务描述: {modules1.get('task_description', '')}
    - 分析方法: {modules1.get('analysis_method', '')}
    - 输出格式: {modules1.get('output_format', '')}
    - 质量标准: {modules1.get('quality_criteria', '')}
    
    Parent 2 模块：
    - 任务描述: {modules2.get('task_description', '')}
    - 分析方法: {modules2.get('analysis_method', '')}
    - 输出格式: {modules2.get('output_format', '')}
    - 质量标准: {modules2.get('quality_criteria', '')}
    
    请选择每个模块的最佳版本并创新性组合，生成新prompt。
    """
    
    return llm_generate(crossover_prompt)
```

### 4. 选择策略

#### 4.1 精英选择
```python
def elite_selection(population: List[PromptIndividual], elite_ratio: float = 0.2) -> List[PromptIndividual]:
    """精英选择：保留最优个体"""
    sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True)
    elite_count = max(1, int(len(population) * elite_ratio))
    return sorted_population[:elite_count]
```

#### 4.2 锦标赛选择
```python
def tournament_selection(population: List[PromptIndividual], tournament_size: int = 3) -> PromptIndividual:
    """锦标赛选择"""
    tournament = random.sample(population, min(tournament_size, len(population)))
    return max(tournament, key=lambda x: x.fitness)
```

---

## 实验设计

### 1. 实验配置

#### 1.1 进化参数
```python
EVOLUTION_CONFIG = {
    "population_size": 20,          # 种群大小
    "max_generations": 50,          # 最大进化代数
    "elite_ratio": 0.2,            # 精英保留比例
    "mutation_rate": 0.7,          # 变异概率
    "crossover_rate": 0.5,         # 交叉概率
    "tournament_size": 3,          # 锦标赛选择大小
    "convergence_threshold": 0.001, # 收敛阈值
    "max_prompt_tokens": 300       # prompt最大长度
}
```

#### 1.2 评估配置
```python
EVALUATION_CONFIG = {
    "test_samples_per_dataset": 50,  # 每个数据集的测试样本数
    "fitness_weights": {
        "boundary_accuracy": 0.45,
        "semantic_consistency": 0.25,
        "granularity_appropriateness": 0.20,
        "efficiency_reward": 0.05,
        "length_penalty": 0.05
    },
    "boundary_tolerance": 5,         # 边界匹配容忍度（字符数）
    "datasets": ["CPTS", "WikiSection_EN", "WikiSection_DE"]
}
```

### 2. 实验流程

#### 2.1 数据准备
1. **数据集加载和预处理**
2. **训练/验证/测试集划分**
3. **样本分层和难度分级**
4. **评估基准建立**

#### 2.2 进化执行
1. **种群初始化**（种子prompt + 随机生成）
2. **适应度评估**（批量并行处理）
3. **选择操作**（精英选择 + 锦标赛选择）
4. **遗传操作**（变异 + 交叉）
5. **种群更新**（新一代形成）
6. **收敛检查**（性能稳定性判断）

#### 2.3 结果分析
1. **最优prompt提取和分析**
2. **进化趋势可视化**
3. **性能对比和基准测试**
4. **prompt特征分析**

### 3. 评估基准

#### 3.1 基准prompt
```python
BASELINE_PROMPTS = [
    "simple_split": "请在语义边界处用[SPLIT]分割文本: {text}",
    "detailed_instruction": """请仔细分析以下文本，在主题变化的地方插入[SPLIT]标记进行分割...""",
    "structured_approach": """## 文本分割任务\n**输入**: {text}\n**要求**: 在语义边界插入[SPLIT]..."""
]
```

#### 3.2 性能指标
- **绝对性能**: 在标准数据集上的分割准确率
- **相对改进**: 相比基准prompt的提升幅度
- **稳定性**: 在不同测试样本上的表现一致性
- **泛化能力**: 在不同领域/语言上的适应性

---

## 技术实现

### 1. 系统架构

```
├── core/
│   ├── individual.py              # PromptIndividual类
│   ├── population.py              # 种群管理
│   ├── evolution_engine.py        # 进化引擎
│   └── fitness_evaluator.py       # 适应度评估器
├── operators/
│   ├── mutation.py                # 变异操作
│   ├── crossover.py               # 交叉操作
│   └── selection.py               # 选择操作
├── evaluation/
│   ├── boundary_metrics.py        # 边界准确率计算
│   ├── semantic_metrics.py        # 语义一致性评估
│   └── granularity_metrics.py     # 颗粒度评估
├── datasets/
│   ├── cpts_loader.py             # CPTS数据集加载器
│   ├── wikisection_loader.py      # WikiSection数据集加载器
│   └── data_processor.py          # 数据预处理
├── utils/
│   ├── token_counter.py           # Token计数工具
│   ├── prompt_parser.py           # Prompt解析工具
│   └── llm_interface.py           # LLM接口封装
└── experiments/
    ├── run_evolution.py           # 主实验脚本
    ├── analyze_results.py         # 结果分析脚本
    └── config.py                  # 配置管理
```

### 2. 关键接口

#### 2.1 LLM接口抽象
```python
class LLMInterface:
    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
        pass
    
    def count_tokens(self, text: str) -> int:
        """计算token数量"""
        pass
    
    def apply_segmentation_prompt(self, prompt: str, text: str) -> List[str]:
        """应用分割prompt并返回分割结果"""
        pass
```

#### 2.2 数据集接口
```python
class SegmentationDataset:
    def load_samples(self, split: str = "train") -> List[SegmentationSample]:
        """加载数据样本"""
        pass
    
    def get_evaluation_samples(self, count: int = 50) -> List[SegmentationSample]:
        """获取评估样本"""
        pass
    
    def preprocess_sample(self, raw_sample: Dict) -> SegmentationSample:
        """预处理样本"""
        pass
```

### 3. 性能优化

#### 3.1 并行处理
- **适应度评估并行化**: 多个prompt同时评估
- **批量LLM调用**: 减少API调用开销
- **缓存机制**: 相同prompt结果缓存

#### 3.2 资源管理
- **内存优化**: 大型数据集分批处理
- **API限制处理**: 请求频率控制和重试机制
- **中间结果保存**: 支持实验中断恢复

---

## 预期成果

### 1. 核心产出

#### 1.1 最优prompt
- **高性能prompt**: 在标准数据集上达到最佳分割效果
- **通用prompt**: 在多个数据集上表现稳定
- **领域特化prompt**: 针对特定领域优化的prompt

#### 1.2 进化分析报告
- **进化趋势分析**: 适应度随代数的变化
- **prompt特征分析**: 高效prompt的共同特征
- **策略效果对比**: 不同进化操作的贡献度

### 2. 方法论贡献

#### 2.1 技术创新
- **LLM驱动的prompt进化**: 首次系统性应用
- **多维适应度设计**: 全面评估prompt质量
- **智能遗传操作**: 基于语义理解的变异和交叉

#### 2.2 应用价值
- **自动prompt优化**: 减少人工调试工作
- **跨领域适应**: 为不同任务优化prompt
- **性能提升**: 显著改进文本分割效果

### 3. 扩展方向

#### 3.1 任务扩展
- **其他NLP任务**: 文本分类、实体识别、情感分析
- **多模态任务**: 图文理解、视频分析
- **代码生成**: 编程prompt优化

#### 3.2 方法扩展
- **多目标优化**: 同时优化多个相互冲突的目标
- **在线学习**: 根据反馈实时调整prompt
- **联邦进化**: 多用户协作的prompt进化

---

## 风险评估与应对

### 1. 技术风险

#### 1.1 LLM依赖风险
- **风险**: API不稳定、成本过高、性能波动
- **应对**: 多LLM支持、本地模型备份、成本控制机制

#### 1.2 评估偏差风险
- **风险**: 适应度函数设计偏差导致局部最优
- **应对**: 多维评估、人工验证、动态调整权重

#### 1.3 过拟合风险
- **风险**: prompt过度适应特定数据集，泛化能力差
- **应对**: 交叉验证、多数据集评估、正则化机制

### 2. 实验风险

#### 2.1 收敛失败风险
- **风险**: 种群过早收敛或无法收敛
- **应对**: 多样性保持机制、自适应参数调整

#### 2.2 计算资源风险
- **风险**: 计算成本过高、时间超预期
- **应对**: 分层评估、早停机制、云计算备份

### 3. 应用风险

#### 3.1 提示注入风险
- **风险**: 进化过程中产生恶意或有害prompt
- **应对**: 内容安全检查、人工审核机制

#### 3.2 知识产权风险
- **风险**: 生成的prompt可能侵犯他人知识产权
- **应对**: 原创性检查、法律风险评估

---

## 总结

本设计文档描述了一个创新的LLM prompt进化算法系统，专门针对文本分割任务进行优化。该系统的核心创新包括：

1. **DNA=Prompt的设计理念**: 将prompt作为进化个体的基因
2. **多维适应度评估**: 综合考虑准确率、一致性、合理性等多个维度
3. **智能遗传操作**: 使用LLM进行语义理解的变异和交叉
4. **标准数据集驱动**: 基于CPTS和WikiSection等标准数据集进行有监督进化

通过系统化的实验设计和严格的评估框架，预期该系统能够：
- 自动发现高效的文本分割prompt
- 显著提升分割任务的性能
- 为prompt工程提供新的方法论
- 为其他NLP任务的prompt优化提供参考

该方案具有重要的理论价值和实用意义，有望在自然语言处理和人工智能领域产生积极影响。