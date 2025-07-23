# 分形压缩器 Demo 示例

这个目录包含了分形压缩器的完整CLI演示系统，展示了不同类型文本的压缩效果。

## 快速开始

### 1. 环境准备

```bash
# 激活虚拟环境
source ../venv/bin/activate

# 确保API密钥已设置
export OPENAI_API_KEY="your-api-key-here"
```

### 2. 运行演示

#### 交互式模式（推荐）
```bash
python compress_demo.py --interactive
```

#### 批量处理所有文本类型
```bash
python compress_demo.py --batch all --preset balanced
```

#### 处理特定文本类型
```bash
python compress_demo.py --batch novel tech --preset aggressive
```

#### 处理本地文件
```bash
python compress_demo.py --file ../demo.py --preset conservative
```

## 命令行选项

### 基本命令

| 命令 | 说明 |
|------|------|
| `--interactive` | 启动交互式模式，可以逐步选择文本和压缩策略 |
| `--batch TYPE [TYPE ...]` | 批量处理指定类型的文本 |
| `--file PATH` | 处理指定的文本文件 |
| `--preset PRESET` | 选择压缩预设：conservative, balanced, aggressive |

### 信息查询

| 命令 | 说明 |
|------|------|
| `--list-texts` | 列出所有可用的测试文本 |
| `--list-presets` | 列出所有压缩预设配置 |

## 文本类型

### 预设测试文本

1. **novel** - 科幻小说片段
   - 内容：未来太空探索故事
   - 长度：~1200字符
   - 特点：对话、叙述、情节发展

2. **tech** - 技术文档
   - 内容：分布式系统架构设计
   - 长度：~1500字符
   - 特点：技术术语、结构化内容

3. **news** - 科技新闻报道
   - 内容：AI技术发布新闻
   - 长度：~1300字符
   - 特点：事实陈述、引用、时效性

4. **academic** - 学术论文摘要
   - 内容：网络路由优化研究
   - 长度：~1400字符
   - 特点：学术格式、专业术语

## 压缩预设

### 1. Conservative (保守压缩)
- **比例**: 0.7
- **基础门限**: 200
- **最大层级**: 6
- **适用场景**: 重要文档，需要保留更多细节

### 2. Balanced (平衡压缩) - 默认
- **比例**: 0.618 (黄金分割)
- **基础门限**: 100
- **最大层级**: 8
- **适用场景**: 通用压缩，平衡压缩率和质量

### 3. Aggressive (激进压缩)
- **比例**: 0.4
- **基础门限**: 50
- **最大层级**: 10
- **适用场景**: 最大化压缩率，适合预览和索引

## 输出文件

### 文件位置
所有压缩结果保存在 `results/` 目录中：

```
examples/
├── results/
│   ├── compression_novel_balanced_1642345678.json
│   ├── compression_tech_aggressive.json
│   ├── batch_compression_balanced_1642345678.json
│   └── ...
```

### 结果格式

每个结果文件包含：

```json
{
  "success": true,
  "config": {
    "name": "平衡压缩",
    "ratio": 0.618,
    "base_threshold": 100,
    "max_levels": 8
  },
  "original_text": "原文内容...",
  "compressed_levels": [
    "Level 0 压缩内容",
    "Level 1 压缩内容",
    "..."
  ],
  "thresholds": [100, 61, 38, 23, 14, 9, 5, 3],
  "statistics": {
    "original_length": 1200,
    "compressed_length": 245,
    "compression_ratio": 0.204,
    "compression_efficiency": 79.6,
    "processing_time": 15.3,
    "speed_chars_per_sec": 78
  },
  "metadata": {
    "text_type": "科幻小说片段",
    "preset": "balanced",
    "timestamp": "2024-01-15 14:30:25"
  }
}
```

## 使用示例

### 示例1：比较不同压缩策略
```bash
# 用不同策略压缩同一文本
python compress_demo.py --batch novel --preset conservative
python compress_demo.py --batch novel --preset balanced  
python compress_demo.py --batch novel --preset aggressive
```

### 示例2：批量处理技术文档
```bash
# 处理技术相关文本
python compress_demo.py --batch tech academic --preset balanced
```

### 示例3：交互式探索
```bash
# 启动交互模式，逐步体验
python compress_demo.py --interactive
```

在交互模式中：
1. 选择文本类型：`novel`、`tech`、`news`、`academic`
2. 选择压缩预设：`conservative`、`balanced`、`aggressive`
3. 查看实时压缩过程和结果分析
4. 输入 `quit` 退出

## 性能指标说明

- **压缩比**: 压缩后长度/原文长度，越小越好
- **压缩效率**: (1 - 压缩比) × 100%，越高越好
- **处理速度**: 字符数/秒，反映压缩性能
- **活跃层级**: 实际使用的压缩层级数量

## 注意事项

1. **API配置**: 确保正确设置 `OPENAI_API_KEY` 环境变量
2. **网络连接**: 压缩过程需要调用LLM API，确保网络通畅
3. **结果文件**: 批量处理会生成多个文件，注意磁盘空间
4. **处理时间**: 首次运行可能较慢，后续会有缓存优化

## 扩展使用

### 添加自定义文本
编辑 `compress_demo.py` 中的 `DEMO_TEXTS` 字典，添加新的文本类型。

### 自定义压缩配置
修改 `COMPRESSION_PRESETS` 字典，调整压缩参数以适应特定需求。

### 处理大文件
对于大型文档，建议先分段处理或使用 `conservative` 预设以保证质量。