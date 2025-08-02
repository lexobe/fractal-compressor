# CLAUDE.md

本文件为Claude Code (claude.ai/code)在处理本代码库时提供指导。

## 项目概述

这是一个**FACE（分形抽象上下文编码）**的Python实现，是一个具有前缀寻址协议的流式层级抽象系统。FACE将线性文本流转换为分层、可追溯的抽象结构，在保持完整历史的同时实现高效检索和推理。

### 核心架构

FACE系统围绕三个主要阶段构建：

1. **FACE核心引擎** (`src/fractal_compressor/face_encoder.py`)：主要FACE实现，包含流式块处理和层级抽象
2. **节点管理** (`src/fractal_compressor/face_node.py`)：节点数据结构，包含抽象状态和父映射
3. **前缀寻址** (`src/fractal_compressor/face_addressing.py`)：Level-0路径编码，具有稳定、可逆地址
4. **LLM集成** (`src/fractal_compressor/llm_compressor.py`)：使用litellm的基于LLM的内容提升

### 关键FACE概念

- **流式Chunk化**：将文本缓冲区转换为语义Level-0块，处理缓冲区残留
- **部分抽象**：基于批量的提升，具有前沿跟踪和尾部保留（β,k机制）
- **不删除历史**：所有节点永久保留；仅进行abstracted=True标记
- **前缀路径寻址**：支持LCP/LCA检索的稳定、可逆地址
- **单调覆盖**：每个节点最多被抽象一次
- **过程自相似**：所有层级使用相同的"选批→抽象→挂接"操作

## FACE数学模型

### 数据结构

**节点定义**：
```
N = (id, level, content, |content|, t_now, abstracted, π)
```

其中：
- `level ∈ {0,1,...}`：层级位置
- `|content|`：token数量  
- `t_now`：创建时间戳（推荐逻辑时钟）
- `abstracted ∈ {False, True}`：初始为False
- `π`：父映射（抽象前为∅）

**系统层级**：
```
𝒜 = {L₀, L₁, L₂, ..., L_max}
```

### 第一阶段：Chunk化（Level-0构建）

**缓冲区管理**：
```
Buffer ⊂ T, |tokens(Buffer)| ≤ τ_buffer
```

**语义分块**：
```  
Chunker(Buffer, τ_slice) → C = {c₁, ..., cₚ}, |cᵢ| ≤ τ_slice
```

**L₀节点生成**：
```
L₀ ← L₀ ∪ {N₀,ᵢ = (·, 0, cᵢ, |cᵢ|, t_now, False, ∅)}
```

**缓冲区残留**：不完整的语义块保留在缓冲区中等待下次迭代。

### 第二阶段：部分抽象（批量β，尾部保留k）

**前沿定义**：
```
Frontier(Lₖ) = {N ∈ Lₖ | N.abstracted = False}
```

**提升触发**：
```
ElevationTrigger(Lₖ) = True if |Frontier(Lₖ)| ≥ β
```

**批次分割**：
```
Batch_k = (F₁, ..., F_β)
A = (F₁, ..., F_{β-k})    # 抽象集合
H = (F_{β-k+1}, ..., F_β)  # 保留集合（尾部保留）
```

**内容提升**：
```
Combined = Concat{N.content : N ∈ A}
E = LLM_elevator(Combined)
N_elev = (·, k+1, E, |E|, t_now, False, ∅)
```

**父映射**：
```
∀N ∈ A: π(N) = N_elev, N.abstracted ← True
```

**关键特性**：
- **不删除**：所有节点永久保留
- **滑动窗口**：步长 = β-k
- **尾部保留**：H节点保持未抽象状态以维持上下文连续性

### 第三阶段：Level-0路径编码（前缀路径寻址）

**最高祖先函数**：
```
Top(x) = x if π(x) = ∅
         Top(π(x)) otherwise
```

**历史序列**：
```
H_n = (Top(c₀), ..., Top(c_{n-1}))
Hist_n = ρ(H_n)  # 首次出现去重
```

**祖先链**：
```
Anc(c_n) = (a_d, a_{d-1}, ..., a₁)
其中 a_d = Top(c_n), a₁ = π(c_n), a_{i+1} = π(aᵢ)
```

**编码函数**：
```
Encode(c_n) = ρ(Hist_n ++ Anc(c_n) ++ (c_n))
```

## 开发命令

### 安装
```bash
pip install -e .
```

### 测试
```bash
# 运行所有FACE测试
pytest tests/test_face_*.py

# 运行特定FACE组件
python tests/test_face_integration.py
python tests/test_face_mathematical_verification.py
python tests/test_address_formats.py

# 运行FACE示例和演示
python examples/face_demo.py
python examples/math_corpus_encode_demo.py

# 运行温度确定性测试
python tests/test_temperature_simple.py

# 分析分块问题
python debug/analyze_chunking_issue.py
```

### 代码质量
```bash
# 格式化代码
black src/ tests/ examples/

# 排序导入  
isort src/ tests/ examples/

# 类型检查
mypy src/

# 运行所有质量检查
black src/ tests/ examples/ && isort src/ tests/ examples/ && mypy src/
```

### FACE开发
```bash
# 构建FACE编码器
python -m build

# 以开发模式安装
pip install -e .

# 运行FACE基准测试
python benchmarks/face_performance.py
```

## 环境要求

- Python 3.8+
- `OPENAI_API_KEY`在 .env 文件中
- 依赖：`litellm>=1.0.0`, `tiktoken`

## 测试策略

测试按FACE组件组织：
- `test_face_encoder.py`：核心FACE引擎和流式处理
- `test_face_node.py`：节点管理和抽象状态
- `test_face_addressing.py`：前缀路径编码和LCP/LCA操作
- `test_face_integration.py`：端到端FACE系统测试

所有测试使用逻辑时钟确保确定性、可重现的结果。

## FACE代码模式

### 基本用法
```python
from fractal_compressor import FACEEncoder, FACEAddressing

# 创建FACE编码器
encoder = FACEEncoder(
    tau_buffer=100,    # 缓冲区token限制
    tau_slice=80,      # 单个块token限制
    beta=4,            # 抽象批量大小
    k=2,               # 尾部保留数量
    use_logical_clock=True
)

# 流式文本处理
encoder.append_text("您的文本内容...")
encoder.process_buffer()  # 如需要触发分块

# 前缀寻址系统
addressing = FACEAddressing()
for node in encoder.all_nodes.values():
    addressing.register_node(node)

# 多格式地址编码
level0_node = encoder.get_level0_nodes()[0]
list_address = addressing.encode_path(level0_node, format="list")      # 默认格式
json_address = addressing.encode_path(level0_node, format="json")      # JSON格式
text_address = addressing.encode_path(level0_node, format="text")      # 人类可读格式
```

### 高级FACE操作
```python
# 前沿分析
frontier_nodes = encoder.get_frontier(level=1)

# LCP/LCA查询
lcp = addressing.find_longest_common_prefix(addr1, addr2)
lca_id = addressing.find_lowest_common_ancestor_by_addresses(addr1, addr2)

# 地址分析工具
query_tool = FACEAddressQuery(addressing)
patterns = query_tool.analyze_address_patterns()
related = query_tool.find_related_nodes(node, max_distance=3)
```

### 错误处理和质量控制
- 空文本输入得到优雅处理
- 无效参数引发带描述性消息的`ValueError`
- **LLM分块100%冗余机制**：并行2次调用，质量评分选择最佳结果
- **分块质量验证**：长度差异≤20%，词汇保留率≥80%，块长度合理
- **降级机制**：LLM分块失败时自动降级到本地分块
- **温度确定性**：temperature=0提供完全确定性（已验证）

### 关键实现约束

#### **Chunk拆分不变性原则**
⚠️ **绝对关键**：Chunk拆分绝对不能修改内容！
```
所有Level-0节点的content连接 = 完整原文（一模一样）
```
- 分块只能在字符边界处切割，不能增删改任何字符
- 缓冲区处理必须保证字符完整性
- 文本合并过程不能污染内容

#### **已知问题和修复优先级**
1. **HIGH**: 缓冲区字符边界处理错误导致内容截断（"下欧拉"问题）
2. **MEDIUM**: 本地分块算法在中文字符处理上的fallback机制缺陷
3. **LOW**: LLM分块质量验证标准可能过于严格

### FACE节点状态
系统跟踪节点抽象状态：
- `abstracted=False`：节点在前沿，可被抽象
- `abstracted=True`：节点已被抽象，具有父映射  
- `π=∅`：无父节点（根节点或未抽象）
- `π=N_parent`：父节点引用

## 配置文件

- `pyproject.toml`：完整的FACE项目配置
- `FACE.md`：数学规范和正式定义
- `FRACTAL_ENCODE_V2_MATHEMATICS.md`：数学基础
- 测试配置包含pytest、覆盖率和可重现的逻辑时钟

## FACE实现指南

### 节点管理
- 使用逻辑时钟（`t_now`）进行确定性排序
- 永久保留所有节点（不删除）
- 用`abstracted`布尔值标记抽象状态
- 维护父映射（`π`）用于路径重建

### 流式处理
- 在缓冲区块中处理文本，处理残留
- 当`|tokens(Buffer)| ≥ τ_buffer`时触发分块
- 优雅处理不完整的语义边界
- 维护块大小限制`|cᵢ| ≤ τ_slice`

### 抽象策略
- 使用基于前沿的批次选择
- 应用（β,k）机制：抽象β-k个节点，保留k个节点
- 每批生成单个提升节点
- 建立父映射而不删除

### 寻址协议
- 为Level-0块实现前缀路径编码
- **多格式输出**：list（默认）、json（结构化）、text（人类可读）
- 支持LCP（最长公共前缀）操作
- 启用LCA（最低公共祖先）查询
- 保持地址稳定性（仅追加）

#### 地址格式示例
```python
# List格式（默认）
["L2_N16_15", "L1_N5_4", "L0_N1_0"]

# JSON格式
{
  "encoding_metadata": {"target_node_id": "L0_N1_0", ...},
  "address_path": ["L2_N16_15", "L1_N5_4", "L0_N1_0"],
  "components": [{"id": "L2_N16_15", "level": 2, ...}],
  "navigation_info": {"path_string": "L2_N16_15 -> L1_N5_4 -> L0_N1_0"}
}

# Text格式
📍 FACE地址编码 - 节点 L0_N1_0
🎯 目标节点: L0_N1_0 (Level 0)
🔗 路径字符串: L2_N16_15 -> L1_N5_4 -> L0_N1_0
🧩 地址组件详情: [树形展示结构]
```

## FACE系统特性

1. **可追溯性**：任何高层结论都可以通过π(·)追溯到L₀证据
2. **稳定地址**：旧编码永不更改；新祖先只影响未来的Hist
3. **前缀检索**：祖先地址是后代地址的前缀
4. **单调抽象**：每个节点最多抽象一次；π形成树结构
5. **线性存储增长**：通过索引/内容分离和前缀压缩缓解
6. **确定性**：逻辑时钟 + 固定抽象器设置确保可重现性

## 参数建议

- `τ_slice`：优先考虑句子/段落边界，考虑模型上下文限制
- `τ_buffer`：略大于τ_slice以确保至少1-2个块
- `β`：平衡吞吐量与延迟（更大批次=更稳定但延迟更高）
- `k`：控制重叠度（k=0非重叠；k=β-1几乎逐点滑动）
- 批次排序：默认按t_now/id升序，保持时间可解释性

---

**FACE总结**：FACE结合了"部分抽象的流式层级结构"与"稳定可逆的前缀地址"：不删除历史、一次抽象、持续挂接，使用地址做上下文，将检索和推理成本前移到结构设计中。