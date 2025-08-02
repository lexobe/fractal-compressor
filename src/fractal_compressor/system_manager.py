#!/usr/bin/env python3
"""
系统管理器 - 统一处理配置、依赖、错误处理和降级机制
解决分散式配置管理和脆弱的错误处理问题
"""

import os
import sys
import logging
import traceback
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from pathlib import Path

# 尝试导入可选依赖
try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

try:
    import litellm
    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False


@dataclass
class SystemStatus:
    """系统状态"""
    is_healthy: bool
    has_llm_capability: bool
    has_valid_api_key: bool
    issues: List[str]
    warnings: List[str]
    config: Dict[str, Any]


class FallbackCompressor:
    """本地智能压缩器 - 当LLM不可用时的备选方案"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.FallbackCompressor")
    
    def smart_summarize(self, text: str, target_length: int) -> str:
        """智能本地摘要"""
        if len(text) <= target_length:
            return text
            
        # 按句子分割
        sentences = self._split_sentences(text)
        if not sentences:
            return text[:target_length]
        
        # 保留重要句子
        important_sentences = self._select_important_sentences(sentences, target_length)
        summary = " ".join(important_sentences)
        
        # 如果仍然太长，进行智能截断
        if len(summary) > target_length:
            summary = self._smart_truncate(summary, target_length)
        
        self.logger.info(f"本地摘要: {len(text)} -> {len(summary)} 字符")
        return summary
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        分句 - 保持内容完整性
        
        ⚠️ 重要：必须保证所有返回的句子连接后等于原文本
        """
        import re
        
        # 使用finditer找到所有句号位置，保持分隔符
        pattern = r'[。！？.!?]'
        matches = list(re.finditer(pattern, text))
        
        if not matches:
            # 没有句号，返回整个文本
            return [text]
        
        sentences = []
        start = 0
        
        for match in matches:
            end = match.end()
            sentence = text[start:end]
            if sentence:  # 只检查非空，不strip
                sentences.append(sentence)
            start = end
        
        # 添加最后一部分（如果有）
        if start < len(text):
            remaining = text[start:]
            if remaining:
                sentences.append(remaining)
        
        # 验证完整性
        reconstructed = ''.join(sentences)
        if reconstructed != text:
            self.logger.warning(f"句子分割破坏了内容完整性，回退到整文本")
            return [text]
        
        return sentences
    
    def _select_important_sentences(self, sentences: List[str], target_length: int) -> List[str]:
        """选择重要句子"""
        # 简单策略：保留较长的句子，因为它们通常包含更多信息
        sentences_with_length = [(s, len(s)) for s in sentences]
        sentences_with_length.sort(key=lambda x: x[1], reverse=True)
        
        selected = []
        total_length = 0
        
        for sentence, length in sentences_with_length:
            if total_length + length <= target_length * 0.8:  # 留一些缓冲
                selected.append(sentence)
                total_length += length
            else:
                break
        
        # 按原始顺序排序
        original_order = []
        for sentence in sentences:
            if sentence in selected:
                original_order.append(sentence)
        
        return original_order
    
    def _smart_truncate(self, text: str, target_length: int) -> str:
        """智能截断 - 在词边界截断"""
        if len(text) <= target_length:
            return text
        
        # 尝试在句子边界截断
        sentences = self._split_sentences(text)
        result = ""
        for sentence in sentences:
            if len(result + sentence) <= target_length - 3:  # 留空间给省略号
                result += sentence + "。"
            else:
                break
        
        if result:
            return result
        
        # 如果句子都太长，在词边界截断
        words = text.split()
        result = ""
        for word in words:
            if len(result + word) <= target_length - 3:
                result += word + " "
            else:
                break
        
        return result.strip() + "..."


class SystemManager:
    """系统管理器 - 统一管理配置、依赖和错误处理"""
    
    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root) if project_root else self._find_project_root()
        self.logger = self._setup_logging()
        self.fallback_compressor = FallbackCompressor()
        self._status: Optional[SystemStatus] = None
        
    def _find_project_root(self) -> Path:
        """自动查找项目根目录"""
        # 从当前文件向上查找包含.env或pyproject.toml的目录
        current = Path(__file__).parent
        while current != current.parent:
            if (current / ".env").exists() or (current / "pyproject.toml").exists():
                return current
            current = current.parent
        return Path.cwd()
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger(f"{__name__}.SystemManager")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def initialize(self) -> SystemStatus:
        """系统初始化 - 检查所有依赖和配置"""
        self.logger.info("开始系统初始化...")
        
        issues = []
        warnings = []
        config = {}
        
        # 1. 加载环境变量
        env_status = self._load_environment()
        if env_status['issues']:
            issues.extend(env_status['issues'])
        if env_status['warnings']:
            warnings.extend(env_status['warnings'])
        config.update(env_status['config'])
        
        # 2. 检查依赖
        dep_status = self._check_dependencies()
        if dep_status['issues']:
            issues.extend(dep_status['issues'])
        if dep_status['warnings']:
            warnings.extend(dep_status['warnings'])
        config.update(dep_status['config'])
        
        # 3. 验证API配置
        api_status = self._validate_api_config(config)
        if api_status['issues']:
            issues.extend(api_status['issues'])
        if api_status['warnings']:
            warnings.extend(api_status['warnings'])
        config.update(api_status['config'])
        
        # 4. 生成系统状态
        has_llm = config.get('has_litellm', False) and config.get('api_key_valid', False)
        
        self._status = SystemStatus(
            is_healthy=len(issues) == 0,
            has_llm_capability=has_llm,
            has_valid_api_key=config.get('api_key_valid', False),
            issues=issues,
            warnings=warnings,
            config=config
        )
        
        # 5. 输出状态报告
        self._print_status_report()
        
        return self._status
    
    def _load_environment(self) -> Dict[str, Any]:
        """加载环境变量"""
        issues = []
        warnings = []
        config = {}
        
        # 尝试加载.env文件
        env_file = self.project_root / ".env"
        if env_file.exists():
            if HAS_DOTENV:
                load_dotenv(env_file)
                self.logger.info(f"已加载环境变量文件: {env_file}")
                config['env_file_loaded'] = True
            else:
                warnings.append("发现.env文件但python-dotenv未安装，建议运行: pip install python-dotenv")
                config['env_file_loaded'] = False
        else:
            warnings.append(f"未找到.env文件: {env_file}")
            config['env_file_loaded'] = False
        
        # 检查关键环境变量
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            if api_key.startswith('sk-'):
                config['openai_api_key'] = api_key
                config['has_api_key'] = True
                self.logger.info("发现OpenAI API密钥")
            else:
                warnings.append(f"API密钥格式可能不正确: {api_key[:10]}...")
                config['openai_api_key'] = api_key
                config['has_api_key'] = True
        else:
            warnings.append("未设置OPENAI_API_KEY环境变量")
            config['has_api_key'] = False
        
        return {
            'issues': issues,
            'warnings': warnings,
            'config': config
        }
    
    def _check_dependencies(self) -> Dict[str, Any]:
        """检查依赖"""
        issues = []
        warnings = []
        config = {}
        
        # 检查litellm
        if HAS_LITELLM:
            config['has_litellm'] = True
            self.logger.info("LiteLLM可用")
        else:
            issues.append("litellm未安装，LLM功能不可用。运行: pip install litellm")
            config['has_litellm'] = False
        
        # 检查dotenv
        config['has_dotenv'] = HAS_DOTENV
        if not HAS_DOTENV:
            warnings.append("python-dotenv未安装，建议安装: pip install python-dotenv")
        
        return {
            'issues': issues,
            'warnings': warnings,
            'config': config
        }
    
    def _validate_api_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证API配置"""
        issues = []
        warnings = []
        result_config = {}
        
        if config.get('has_litellm') and config.get('has_api_key'):
            # 尝试进行一个简单的API调用来验证密钥
            try:
                import litellm
                # 进行最小的API调用测试
                response = litellm.completion(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1,
                    api_key=config['openai_api_key']
                )
                result_config['api_key_valid'] = True
                self.logger.info("API密钥验证成功")
            except Exception as e:
                error_msg = str(e)
                if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                    issues.append(f"API密钥验证失败: {error_msg}")
                    result_config['api_key_valid'] = False
                else:
                    # 其他错误可能是网络问题，假设密钥有效
                    warnings.append(f"API密钥验证遇到网络错误，假设有效: {error_msg}")
                    result_config['api_key_valid'] = True
        else:
            result_config['api_key_valid'] = False
        
        return {
            'issues': issues,
            'warnings': warnings,
            'config': result_config
        }
    
    def _print_status_report(self):
        """打印状态报告"""
        status = self._status
        print("\n" + "="*60)
        print("🔧 系统状态报告")
        print("="*60)
        
        # 整体状态
        if status.is_healthy:
            print("✅ 系统状态: 健康")
        else:
            print("❌ 系统状态: 有问题")
        
        # LLM能力
        if status.has_llm_capability:
            print("🤖 LLM功能: 可用")
        else:
            print("⚠️  LLM功能: 降级到本地处理")
        
        # API密钥
        if status.has_valid_api_key:
            print("🔑 API密钥: 有效")
        else:
            print("🔓 API密钥: 无效或缺失")
        
        # 问题
        if status.issues:
            print(f"\n❌ 问题 ({len(status.issues)}):")
            for issue in status.issues:
                print(f"   • {issue}")
        
        # 警告
        if status.warnings:
            print(f"\n⚠️  警告 ({len(status.warnings)}):")
            for warning in status.warnings:
                print(f"   • {warning}")
        
        print("="*60 + "\n")
    
    def get_llm_config(self) -> Dict[str, Any]:
        """获取LLM配置"""
        if not self._status:
            raise RuntimeError("系统未初始化，请先调用initialize()")
        
        if self._status.has_llm_capability:
            return {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": self._status.config['openai_api_key']
            }
        else:
            return {}
    
    def robust_llm_compress(self, text: str, target_length: int, 
                           custom_template: Optional[str] = None) -> Dict[str, Any]:
        """健壮的LLM压缩 - 带智能降级"""
        if not self._status:
            raise RuntimeError("系统未初始化，请先调用initialize()")
        
        # 尝试LLM压缩
        if self._status.has_llm_capability:
            try:
                from .llm_compressor import LLMTextCompressor
                compressor = LLMTextCompressor()
                
                result = compressor.compress(
                    text=text,
                    target_length=target_length,
                    llm_config=self.get_llm_config(),
                    custom_template=custom_template,
                    max_attempts=2
                )
                
                result['method'] = 'llm'
                result['fallback_used'] = False
                return result
                
            except Exception as e:
                self.logger.warning(f"LLM压缩失败，降级到本地处理: {e}")
                # 降级到本地处理
        
        # 本地智能压缩
        self.logger.info("使用本地智能压缩")
        compressed = self.fallback_compressor.smart_summarize(text, target_length)
        
        return {
            'text': compressed,
            'original_text': text,
            'original_length': len(text),
            'compressed_length': len(compressed),
            'target_length': target_length,
            'compression_ratio': len(compressed) / len(text) if len(text) > 0 else 0,
            'method': 'local_smart',
            'fallback_used': True,
            'success': True,
            'length_constraint_satisfied': len(compressed) <= target_length
        }
    
    def robust_chunker(self, text: str, max_tokens: int = 100) -> List[str]:
        """健壮的文本分块 - 带智能降级"""
        if not self._status:
            raise RuntimeError("系统未初始化，请先调用initialize()")
        
        # ⚠️ 强制使用本地分块以保证Chunk拆分不变性原则
        # LLM分块会修改内容格式（如添加空格、修改换行符），违反不变性原则
        self.logger.info("使用本地智能分块以保证内容不变性")
        return self._smart_chunk_local(text, max_tokens)
    
    def _llm_chunk_text(self, text: str, max_tokens: int) -> List[str]:
        """使用LLM进行语义分块 - 100%冗余度质量保证"""
        import litellm
        
        # 估算token数量
        estimated_tokens = len(text) * 0.75  # 粗略估算：中文1字符≈0.75token
        
        # 如果文本很短，不需要分块
        if estimated_tokens <= max_tokens:
            return [text]
        
        # 构造LLM分块prompt
        chunk_prompt = f"""你是一个专业的文本分段专家。请将下面的文本按照语义自然边界分成若干段落。

### 要求：
1. **语义完整**：每个段落必须语义完整，不能在句子中间断开
2. **长度限制**：每个段落不超过{max_tokens}个token（约{max_tokens * 1.3:.0f}个字符）
3. **保持原文**：不要修改、总结或重写原文内容
4. **自然边界**：在段落、句子或逻辑单元的自然边界处分段

### 输出格式：
请用"---CHUNK---"分隔每个段落，像这样：
第一段内容
---CHUNK---
第二段内容
---CHUNK---
第三段内容

### 待分段文本：
{text}"""

        # 100%冗余度：并行执行两次相同的LLM调用
        results = []
        errors = []
        
        for attempt in range(2):  # 两次并行尝试
            try:
                response = litellm.completion(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": chunk_prompt}],
                    temperature=0.0,  # 确定性设置
                    max_tokens=len(text) + 500,
                    api_key=self._status.config['openai_api_key']
                )
                
                result_text = response.choices[0].message.content.strip()
                chunks = [chunk for chunk in result_text.split("---CHUNK---") if chunk]
                
                # 基础验证
                if chunks and self._validate_chunks(chunks, text, max_tokens):
                    results.append(chunks)
                    self.logger.debug(f"LLM分块尝试{attempt + 1}成功: {len(chunks)}个块")
                else:
                    self.logger.warning(f"LLM分块尝试{attempt + 1}验证失败")
                    
            except Exception as e:
                errors.append(str(e))
                self.logger.warning(f"LLM分块尝试{attempt + 1}失败: {e}")
        
        # 冗余度质量选择
        if len(results) == 2:
            # 两次都成功：选择质量更好的结果
            result1, result2 = results
            if self._compare_chunk_quality(result1, result2, text):
                chosen_result = result1
                self.logger.info(f"选择结果1: {len(result1)}个块")
            else:
                chosen_result = result2
                self.logger.info(f"选择结果2: {len(result2)}个块")
            
            self.logger.info(f"LLM分段成功(100%冗余): {len(text)}字符 -> {len(chosen_result)}个语义块")
            return chosen_result
            
        elif len(results) == 1:
            # 只有一次成功：使用成功的结果
            self.logger.info(f"LLM分段成功(50%冗余): {len(text)}字符 -> {len(results[0])}个语义块")
            return results[0]
            
        else:
            # 两次都失败：抛出异常让降级机制处理
            error_msg = f"LLM分块两次尝试均失败: {'; '.join(errors)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _validate_chunks(self, chunks: List[str], original_text: str, max_tokens: int) -> bool:
        """验证分块结果质量"""
        if not chunks:
            return False
        
        # 检查长度差异（放宽到20%以适应LLM的变异性）
        total_length = sum(len(chunk) for chunk in chunks)
        length_diff_ratio = abs(total_length - len(original_text)) / len(original_text)
        if length_diff_ratio > 0.2:
            self.logger.warning(f"分块长度差异过大: {length_diff_ratio:.1%}")
            return False
        
        # 检查每个块的长度是否合理
        for i, chunk in enumerate(chunks):
            estimated_tokens = len(chunk) * 0.75
            if estimated_tokens > max_tokens * 1.5:  # 允许50%超出
                self.logger.warning(f"块{i}长度超限: {estimated_tokens:.0f} > {max_tokens * 1.5:.0f}")
                return False
        
        # 检查是否有明显的内容缺失
        original_words = set(original_text.split())
        chunk_words = set(' '.join(chunks).split())
        word_retention = len(chunk_words & original_words) / len(original_words) if original_words else 1.0
        if word_retention < 0.8:  # 至少保留80%的词汇
            self.logger.warning(f"词汇保留率过低: {word_retention:.1%}")
            return False
        
        return True
    
    def _compare_chunk_quality(self, chunks1: List[str], chunks2: List[str], original_text: str) -> bool:
        """比较两个分块结果的质量，返回True表示chunks1更好"""
        # 评分标准1: 长度保真度
        total1 = sum(len(chunk) for chunk in chunks1)
        total2 = sum(len(chunk) for chunk in chunks2)
        orig_len = len(original_text)
        
        score1_length = 1.0 - abs(total1 - orig_len) / orig_len
        score2_length = 1.0 - abs(total2 - orig_len) / orig_len
        
        # 评分标准2: 块数合理性（更少的块通常意味着更好的语义完整性）
        score1_count = 1.0 / (1.0 + len(chunks1) * 0.1)
        score2_count = 1.0 / (1.0 + len(chunks2) * 0.1)
        
        # 评分标准3: 长度均匀性（块长度方差越小越好）
        avg1 = total1 / len(chunks1)
        avg2 = total2 / len(chunks2)
        var1 = sum((len(c) - avg1) ** 2 for c in chunks1) / len(chunks1)
        var2 = sum((len(c) - avg2) ** 2 for c in chunks2) / len(chunks2)
        
        score1_uniformity = 1.0 / (1.0 + var1 * 0.001)
        score2_uniformity = 1.0 / (1.0 + var2 * 0.001)
        
        # 综合评分
        total_score1 = score1_length * 0.5 + score1_count * 0.3 + score1_uniformity * 0.2
        total_score2 = score2_length * 0.5 + score2_count * 0.3 + score2_uniformity * 0.2
        
        self.logger.debug(f"质量评分 - 结果1: {total_score1:.3f}, 结果2: {total_score2:.3f}")
        return total_score1 >= total_score2
    
    def _smart_chunk_local(self, text: str, max_tokens: int) -> List[str]:
        """本地智能分块 - 改进版"""
        # 使用更准确的token估算
        def estimate_tokens(text: str) -> int:
            # 改进的估算：中文字符≈0.75token，英文单词≈1token，标点≈0.5token
            chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
            english_words = len([w for w in text.split() if any(c.isalpha() for c in w)])
            punctuation = len([c for c in text if c in '。！？.,!?;:'])
            other_chars = len(text) - chinese_chars - english_words - punctuation
            
            return int(chinese_chars * 0.75 + english_words * 1.0 + punctuation * 0.5 + other_chars * 0.6)
        
        # 如果文本很短，直接返回
        if estimate_tokens(text) <= max_tokens:
            return [text]
        
        # 基于句子的智能分块
        sentences = self.fallback_compressor._split_sentences(text)
        if not sentences:
            # 如果无法分句，按字符长度分块
            char_limit = int(max_tokens * 1.3)  # 粗略换算为字符数
            chunks = []
            for i in range(0, len(text), char_limit):
                chunk = text[i:i + char_limit]
                # 尝试在句子边界处截断
                if i + char_limit < len(text):
                    last_period = chunk.rfind('。')
                    last_question = chunk.rfind('？')
                    last_exclamation = chunk.rfind('！')
                    last_punct = max(last_period, last_question, last_exclamation)
                    if last_punct > len(chunk) * 0.7:  # 如果在后70%找到句号
                        chunk = chunk[:last_punct + 1]
                chunks.append(chunk)
            return chunks
        
        # 按句子组合成块
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # 检查添加这个句子是否会超过限制
            test_chunk = current_chunk + sentence
            if current_chunk and estimate_tokens(test_chunk) > max_tokens:
                # 超过限制，保存当前块并开始新块
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                # 可以添加，继续累积
                current_chunk = test_chunk
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk)
        
        # 保底方案：如果没有生成任何块，强制分割
        if not chunks:
            char_limit = int(max_tokens * 1.3)
            chunks = [text[i:i + char_limit] for i in range(0, len(text), char_limit)]
        
        self.logger.info(f"本地分块: {len(text)}字符 -> {len(chunks)}个块")
        return chunks


# 全局系统管理器实例
_system_manager: Optional[SystemManager] = None


def get_system_manager() -> SystemManager:
    """获取全局系统管理器实例"""
    global _system_manager
    if _system_manager is None:
        _system_manager = SystemManager()
        _system_manager.initialize()
    return _system_manager


def ensure_system_ready() -> SystemStatus:
    """确保系统就绪"""
    manager = get_system_manager()
    return manager._status