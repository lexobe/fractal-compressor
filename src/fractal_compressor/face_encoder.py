"""
FACE核心引擎

实现FACE系统的主要功能：流式块处理、层级抽象和前沿管理。
遵循FACE三阶段协议：Chunk化 → 部分抽象 → 路径编码。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import tiktoken

from .face_node import FACENode, NodeFactory, create_frontier_collection, validate_tree_structure
from .llm_compressor import LLMTextCompressor
from .system_manager import get_system_manager

logger = logging.getLogger(__name__)


class FACEEncoder:
    """
    FACE (Fractal Abstracted Context Encoding) 编码器
    
    实现流式层级抽象系统：
    1. 流式Chunk化：将文本缓冲区转换为语义Level-0块
    2. 部分抽象：批量提升前沿节点到更高层级
    3. 前缀路径寻址：为Level-0块生成稳定地址
    
    核心特性：
    - 不删除历史：所有节点永久保留
    - 单调抽象：每个节点最多抽象一次
    - 过程自相似：各层使用相同的批量抽象机制
    """
    
    def __init__(
        self,
        tau_buffer: int = 100,
        tau_slice: int = 80,
        beta: int = 4,
        k: int = 2,
        max_levels: int = 10,
        use_logical_clock: bool = True,
        llm_config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化FACE编码器
        
        Args:
            tau_buffer: 缓冲区token上限
            tau_slice: 单个chunk token上限
            beta: 抽象批量阈值
            k: 尾部保留数量 (0 ≤ k < beta)
            max_levels: 最大层级数
            use_logical_clock: 是否使用逻辑时钟
            llm_config: LLM配置
        """
        # 参数验证
        if tau_slice > tau_buffer:
            raise ValueError(f"tau_slice ({tau_slice}) 不能大于 tau_buffer ({tau_buffer})")
        if k >= beta:
            raise ValueError(f"尾部保留数 k ({k}) 必须小于批量大小 beta ({beta})")
        if beta < 2:
            raise ValueError(f"批量大小 beta ({beta}) 必须至少为2")
        
        self.tau_buffer = tau_buffer
        self.tau_slice = tau_slice
        self.beta = beta
        self.k = k
        self.max_levels = max_levels
        self.use_logical_clock = use_logical_clock
        self.llm_config = llm_config or {}
        
        # 初始化组件
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.node_factory = NodeFactory(use_logical_clock=use_logical_clock)
        self.llm_compressor = LLMTextCompressor()
        self.system_manager = get_system_manager()
        
        # 系统状态
        self.buffer = ""  # 流式缓冲区
        self.layers: Dict[int, List[FACENode]] = {i: [] for i in range(max_levels)}
        self.all_nodes: Dict[str, FACENode] = {}  # 所有节点的引用
        
        logger.info(f"初始化FACE编码器: τ_buffer={tau_buffer}, τ_slice={tau_slice}, β={beta}, k={k}")
    
    def count_tokens(self, text: str) -> int:
        """计算文本token数量"""
        if not text:
            return 0
        return len(self.tokenizer.encode(text))
    
    def append_text(self, text: str) -> None:
        """
        向缓冲区追加文本
        
        Args:
            text: 要追加的文本
        """
        if not text:
            return
        
        self.buffer += text
        buffer_tokens = self.count_tokens(self.buffer)
        
        logger.debug(f"追加文本: +{len(text)}字符, 缓冲区: {buffer_tokens}tokens")
        
        # 检查是否需要触发分块
        if buffer_tokens >= self.tau_buffer:
            self._trigger_chunking()
    
    def process_buffer(self) -> None:
        """
        手动处理缓冲区（用于流式处理的显式控制）
        """
        if self.buffer.strip():
            self._trigger_chunking()
    
    def _trigger_chunking(self) -> None:
        """
        触发语义分块（第一阶段：Chunk化）
        
        实现：
        1. 调用语义分块器
        2. 创建Level-0节点
        3. 处理缓冲区残留
        4. 触发抽象检查
        """
        if not self.buffer.strip():
            return
        
        buffer_tokens = self.count_tokens(self.buffer)
        logger.info(f"触发分块: 缓冲区{buffer_tokens}tokens")
        
        # 语义分块
        chunks = self._semantic_chunking(self.buffer)
        
        if not chunks:
            logger.warning("分块结果为空，保持缓冲区不变")
            return
        
        # 创建Level-0节点
        new_nodes = []
        for chunk_text in chunks:
            if chunk_text.strip():
                node = self.node_factory.create_node(level=0, content=chunk_text)
                self.layers[0].append(node)
                self.all_nodes[node.id] = node
                new_nodes.append(node)
                logger.debug(f"创建L0节点: {node.id}, {node.token_count}tokens")
        
        # 根据Chunk拆分不变性原则，processed_content必须等于原buffer内容
        processed_content = "".join(chunks)
        
        # 验证Chunk拆分不变性：所有chunks连接必须等于原始缓冲区内容
        if processed_content != self.buffer:
            logger.error(f"Chunk拆分违反不变性原则!")
            logger.error(f"原始缓冲区: {len(self.buffer)}字符")
            logger.error(f"处理结果: {len(processed_content)}字符")
            logger.error(f"原始内容前100字符: {repr(self.buffer[:100])}")
            logger.error(f"处理结果前100字符: {repr(processed_content[:100])}")
            
            # 严重错误：回退到安全模式，不处理任何内容
            logger.error("检测到内容不一致，回退到安全模式")
            return
        
        # 内容完全匹配，清空缓冲区
        self.buffer = ""
        logger.debug(f"缓冲区已清空（内容一致性验证通过）")
        
        logger.info(f"分块完成: 创建{len(new_nodes)}个L0节点，缓冲区残留{len(self.buffer)}字符")
        
        # 触发抽象检查
        self._check_and_trigger_abstraction()
    
    def _semantic_chunking(self, text: str) -> List[str]:
        """
        语义分块
        
        使用系统管理器的健壮分块器进行语义分割
        
        Args:
            text: 要分块的文本
            
        Returns:
            语义块列表
        """
        token_count = self.count_tokens(text)
        
        # 如果文本不超过限制，不分块
        if token_count <= self.tau_slice:
            return [text]
        
        # 使用系统管理器的健壮分块器
        try:
            chunks = self.system_manager.robust_chunker(text, self.tau_slice)
            logger.debug(f"语义分块: {len(text)}字符 -> {len(chunks)}块")
            return chunks
        except Exception as e:
            logger.error(f"语义分块失败: {e}")
            # 回退到简单分块
            return self._simple_chunking(text)
    
    def _simple_chunking(self, text: str) -> List[str]:
        """
        简单分块回退方案
        
        按句子边界进行基本分割
        """
        import re
        
        # 按句子分割（中英文）
        sentences = re.split(r'[。！？.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            test_chunk = current_chunk + sentence + "。"
            if self.count_tokens(test_chunk) <= self.tau_slice:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence + "。"
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks or [text]  # 确保至少返回原文本
    
    def _check_and_trigger_abstraction(self) -> None:
        """
        检查所有层级并触发必要的抽象操作
        """
        for level in range(self.max_levels - 1):  # 最高层不能再抽象
            if self._should_trigger_abstraction(level):
                self._trigger_abstraction(level)
    
    def _should_trigger_abstraction(self, level: int) -> bool:
        """
        检查是否应该触发层级抽象
        
        实现 ElevationTrigger(Lₖ) = True if |Frontier(Lₖ)| ≥ β
        """
        frontier = self.get_frontier(level)
        return len(frontier) >= self.beta
    
    def _trigger_abstraction(self, level: int) -> None:
        """
        触发层级抽象（第二阶段：部分抽象）
        
        实现批量β、尾部保留k的抽象机制：
        1. 获取前沿节点
        2. 分割为抽象集A和保留集H
        3. 生成提升节点
        4. 建立父映射
        
        Args:
            level: 要抽象的层级
        """
        frontier = self.get_frontier(level)
        if len(frontier) < self.beta:
            logger.debug(f"L{level} 前沿节点不足: {len(frontier)} < {self.beta}")
            return
        
        logger.info(f"触发L{level}抽象: {len(frontier)}个前沿节点")
        
        # 批次分割
        batch = frontier[:self.beta]  # 取前β个节点
        abstraction_set = batch[:self.beta - self.k]  # A: 参与抽象的节点
        hold_set = batch[self.beta - self.k:]  # H: 尾部保留的节点
        
        logger.debug(f"批次分割: 抽象{len(abstraction_set)}个，保留{len(hold_set)}个")
        
        # 合并内容进行抽象
        combined_content = self._combine_node_contents(abstraction_set)
        elevated_content = self._elevate_content(combined_content)
        
        # 创建提升节点
        elevated_node = self.node_factory.create_node(
            level=level + 1,
            content=elevated_content,
            metadata={
                'source_nodes': [node.id for node in abstraction_set],
                'abstraction_batch_size': len(abstraction_set),
                'abstraction_method': 'llm_elevation'
            }
        )
        
        # 添加到上层
        self.layers[level + 1].append(elevated_node)
        self.all_nodes[elevated_node.id] = elevated_node
        
        # 建立父映射并标记为已抽象
        for node in abstraction_set:
            node.set_abstracted(elevated_node)
        
        logger.info(f"L{level}抽象完成: {len(abstraction_set)}个节点 -> L{level+1}节点{elevated_node.id}")
        
        # 递归检查是否需要进一步抽象
        if level + 1 < self.max_levels - 1:
            if self._should_trigger_abstraction(level + 1):
                self._trigger_abstraction(level + 1)
    
    def _combine_node_contents(self, nodes: List[FACENode]) -> str:
        """
        合并节点内容
        
        实现 Combined = Concat{N.content : N ∈ A}
        """
        return "\n".join(node.content for node in nodes)
    
    def _elevate_content(self, content: str) -> str:
        """
        内容提升（LLM抽象）
        
        实现 E = LLM_elevator(Combined)
        """
        try:
            result = self.system_manager.robust_llm_compress(
                text=content,
                target_length=min(200, len(content) // 2),  # 目标长度为原长度的一半，但不超过200字符
                custom_template=self._get_elevation_prompt()
            )
            
            elevated_content = result["text"].strip()
            method = result.get("method", "unknown")
            
            logger.debug(f"内容提升({method}): {len(content)}字符 -> {len(elevated_content)}字符")
            return elevated_content
            
        except Exception as e:
            logger.error(f"LLM内容提升失败: {e}")
            # 简单回退：取内容摘要
            lines = content.split('\n')
            if len(lines) > 3:
                return f"概要：{lines[0][:100]}...等{len(lines)}项内容的高层次抽象。"
            return f"概要：{content[:200]}..."
    
    def _get_elevation_prompt(self) -> str:
        """获取内容提升的提示词模板"""
        return """### 角色
你是一位概念抽象专家，专门将具体内容提升为高层次的抽象概念。

### 任务
将给定的内容进行概念提升，提取出：
- 核心概念和本质规律
- 可重复应用的方法模式
- 概念间的深层联系
- 通用的理论框架

### 要求
1. 从具体细节上升到概念框架层面
2. 识别可推广的方法论和解决思路
3. 去除表象，保留核心的逻辑结构
4. 用简洁精练的语言表达抽象结果

直接输出抽象结果，体现概念的本质和方法的一般性。

### 待抽象内容
{text}"""
    
    def get_frontier(self, level: int) -> List[FACENode]:
        """
        获取指定层级的前沿节点
        
        实现 Frontier(Lₖ) = {N ∈ Lₖ | N.abstracted = False}
        """
        if level not in self.layers:
            return []
        return create_frontier_collection(self.layers[level])
    
    def get_all_layers(self) -> Dict[int, List[Dict[str, Any]]]:
        """
        获取所有层级的节点信息
        
        Returns:
            层级 -> 节点信息列表的字典
        """
        result = {}
        for level, nodes in self.layers.items():
            if nodes:
                result[level] = [node.to_dict() for node in nodes]
        return result
    
    def get_layer_stats(self) -> Dict[int, Dict[str, Any]]:
        """获取各层级统计信息"""
        stats = {}
        for level, nodes in self.layers.items():
            if nodes:
                frontier = self.get_frontier(level)
                total_tokens = sum(node.token_count for node in nodes)
                stats[level] = {
                    'total_nodes': len(nodes),
                    'frontier_nodes': len(frontier),
                    'abstracted_nodes': len(nodes) - len(frontier),
                    'total_tokens': total_tokens,
                    'avg_tokens': total_tokens / len(nodes) if nodes else 0
                }
        return stats
    
    def get_level0_nodes(self) -> List[FACENode]:
        """获取所有Level-0节点"""
        return self.layers[0].copy()
    
    def get_node_by_id(self, node_id: str) -> Optional[FACENode]:
        """根据ID获取节点"""
        return self.all_nodes.get(node_id)
    
    def validate_structure(self) -> bool:
        """
        验证FACE结构的完整性
        
        检查：
        1. 树结构无环
        2. 父子关系正确
        3. 抽象状态一致
        """
        try:
            all_nodes_list = list(self.all_nodes.values())
            validate_tree_structure(all_nodes_list)
            
            # 验证层级结构
            for level, nodes in self.layers.items():
                for node in nodes:
                    if node.level != level:
                        raise ValueError(f"节点{node.id}层级不匹配: 节点层级{node.level} != 层级索引{level}")
                    
                    # 验证抽象状态
                    if node.abstracted and node.parent is None:
                        raise ValueError(f"节点{node.id}标记为已抽象但无父节点")
                    
                    if not node.abstracted and node.parent is not None:
                        raise ValueError(f"节点{node.id}未标记为已抽象但有父节点")
            
            logger.info("FACE结构验证通过")
            return True
            
        except Exception as e:
            logger.error(f"FACE结构验证失败: {e}")
            return False
    
    def get_system_summary(self) -> Dict[str, Any]:
        """获取系统状态摘要"""
        stats = self.get_layer_stats()
        total_nodes = sum(len(nodes) for nodes in self.layers.values())
        
        return {
            'total_nodes': total_nodes,
            'active_levels': len([level for level, nodes in self.layers.items() if nodes]),
            'buffer_chars': len(self.buffer),
            'buffer_tokens': self.count_tokens(self.buffer),
            'layer_stats': stats,
            'parameters': {
                'tau_buffer': self.tau_buffer,
                'tau_slice': self.tau_slice,
                'beta': self.beta,
                'k': self.k,
                'max_levels': self.max_levels
            }
        }
    
    def reset(self) -> None:
        """重置编码器状态"""
        self.buffer = ""
        self.layers = {i: [] for i in range(self.max_levels)}
        self.all_nodes = {}
        self.node_factory.reset_clock()
        logger.info("FACE编码器已重置")


def create_face_encoder(
    tau_buffer: int = 100,
    tau_slice: int = 80,
    beta: int = 4,
    k: int = 2,
    max_levels: int = 10,
    use_logical_clock: bool = True,
    llm_config: Optional[Dict[str, Any]] = None
) -> FACEEncoder:
    """
    创建FACE编码器的便捷函数
    
    Args:
        tau_buffer: 缓冲区token上限
        tau_slice: 单个chunk token上限
        beta: 抽象批量阈值
        k: 尾部保留数量
        max_levels: 最大层级数
        use_logical_clock: 是否使用逻辑时钟
        llm_config: LLM配置
        
    Returns:
        配置好的FACE编码器实例
    """
    return FACEEncoder(
        tau_buffer=tau_buffer,
        tau_slice=tau_slice,
        beta=beta,
        k=k,
        max_levels=max_levels,
        use_logical_clock=use_logical_clock,
        llm_config=llm_config
    )