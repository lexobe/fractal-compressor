# FractalEncoder LLM配置系统重构

## 🎯 重构概述

将FractalEncoder的LLM配置从简单的`model`和`language`参数重构为完整的**多供应商配置系统**，支持API密钥、URL等外部配置。

## 🔧 主要改进

### 1. **多供应商支持**
- ✅ OpenAI GPT系列
- ✅ Anthropic Claude系列  
- ✅ Azure OpenAI
- ✅ 本地模型服务

### 2. **完整配置参数**
- `provider`: LLM提供商
- `model`: 模型名称
- `api_key`: API密钥（外部配置）
- `url`: API端点URL（必需）
- `temperature`: 温度参数
- `max_tokens`: 最大token数
- `language`: 语言类型

### 3. **配置验证与默认值**
- 自动验证配置有效性
- 智能填充默认参数
- 错误提示和异常处理

### 4. **运行时配置更新**
- 支持动态切换模型
- 支持参数调整
- 配置合并机制

## 📝 使用示例

### OpenAI配置
```python
llm_config = {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "api_key": "sk-xxx...",
    "url": "https://api.openai.com/v1",
    "temperature": 0.1
}
encoder = FractalEncoder(llm_config=llm_config)
```

### Anthropic配置
```python
llm_config = {
    "provider": "anthropic",
    "model": "claude-3-haiku-20240307",
    "api_key": "sk-ant-xxx...",
    "url": "https://api.anthropic.com",
    "temperature": 0.2,
    "max_tokens": 4000
}
encoder = FractalEncoder(llm_config=llm_config)
```

### Azure OpenAI配置
```python
llm_config = {
    "provider": "azure_openai",
    "model": "gpt-4o",
    "api_key": "azure-key-xxx...",
    "url": "https://xxx.openai.azure.com/",
    "api_version": "2024-02-15-preview"
}
encoder = FractalEncoder(llm_config=llm_config)
```

### 本地模型配置
```python
llm_config = {
    "provider": "local",
    "model": "llama2",
    "url": "http://localhost:11434",
    "temperature": 0.1
}
encoder = FractalEncoder(llm_config=llm_config)
```

## 🔄 运行时配置更新

```python
# 初始配置
encoder = FractalEncoder(llm_config=openai_config)

# 切换模型
encoder.update_llm_config({
    "model": "gpt-4o",
    "temperature": 0.8
})

# 切换提供商
encoder.update_llm_config({
    "provider": "anthropic",
    "model": "claude-3-sonnet-20240229",
    "api_key": "new-anthropic-key"
})
```

## 🛡️ 安全特性

### API密钥管理
```python
import os

# 环境变量方式（推荐）
llm_config = {
    "provider": "openai",
    "api_key": os.getenv("OPENAI_API_KEY"),
    "model": "gpt-4o-mini"
}

# 配置文件方式
with open("llm_config.json") as f:
    llm_config = json.load(f)
```

### 配置验证
- 自动检查必需参数
- 提供商特定验证
- 错误消息指导

## 📊 兼容性

### 向后兼容
- 保持原有API结构
- 便捷函数支持新配置
- 渐进式迁移

### 前向扩展
- 易于添加新提供商
- 参数扩展机制
- 插件化架构

## 🧪 测试覆盖

### 配置验证测试
- ✅ 默认配置处理
- ✅ 自定义配置验证
- ✅ 错误配置捕获

### 多提供商测试
- ✅ OpenAI配置
- ✅ Anthropic配置
- ✅ Azure OpenAI配置
- ✅ 本地模型配置

### 功能集成测试
- ✅ 基本编码功能
- ✅ 配置更新机制
- ✅ 便捷函数支持

## 🔧 便捷函数

### create_fractal_encoder
```python
encoder = create_fractal_encoder(
    ratio=0.7,
    base_threshold=2000,
    llm_config={
        "provider": "openai",
        "api_key": "sk-xxx...",
        "model": "gpt-4o"
    }
)
```

### encode_text_fractal
```python
result = encode_text_fractal(
    "长文本内容...",
    llm_config={
        "provider": "anthropic",
        "api_key": "sk-ant-xxx...",
        "model": "claude-3-sonnet-20240229"
    }
)
```

## 🚀 下一步计划

### 真实LLM集成
- [ ] 实现OpenAI API调用
- [ ] 实现Anthropic API调用
- [ ] 实现Azure OpenAI调用
- [ ] 实现本地模型调用

### 高级功能
- [ ] 配置模板系统
- [ ] 批量配置管理
- [ ] 配置热重载
- [ ] 性能监控

### 安全增强
- [ ] 密钥加密存储
- [ ] 访问权限控制
- [ ] 审计日志
- [ ] 配置备份

## 💡 使用建议

1. **生产环境**: 使用环境变量管理API密钥
2. **开发环境**: 使用配置文件或直接传入
3. **多模型**: 利用运行时配置更新功能
4. **本地测试**: 使用本地模型提供商

---

**重构完成时间**: 2024年
**测试状态**: ✅ 全部通过
**向后兼容**: ✅ 完全兼容 