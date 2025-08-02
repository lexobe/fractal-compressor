"""
FACE节点管理模块

实现FACE系统的核心节点数据结构，包含抽象状态和父映射管理。
节点遵循FACE规范：永久保留、单调抽象、父映射追溯。
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import tiktoken


@dataclass
class FACENode:
    """
    FACE节点数据结构
    
    根据FACE规范实现：N = (id, level, content, |content|, t_now, abstracted, π)
    
    Attributes:
        id: 节点唯一标识符
        level: 层级位置 (0=Level-0 chunks, k>0=抽象层)
        content: 节点内容
        token_count: token数量 |content|
        t_now: 创建时间戳（逻辑时钟）
        abstracted: 抽象状态标记 (False=前沿可抽象, True=已被抽象)
        parent: 父映射 π (None=∅, FACENode=父节点引用)
        children: 子节点列表（用于反向追踪）
        metadata: 可选元数据
    """
    id: str
    level: int
    content: str
    token_count: int
    t_now: Union[float, int]  # 支持逻辑时钟(int)或时间戳(float)
    abstracted: bool = False
    parent: Optional['FACENode'] = None
    children: List['FACENode'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化验证"""
        if self.level < 0:
            raise ValueError(f"节点层级不能为负数: {self.level}")
        if self.token_count < 0:
            raise ValueError(f"token数量不能为负数: {self.token_count}")
        if not self.content.strip():
            raise ValueError("节点内容不能为空")
    
    def is_root(self) -> bool:
        """判断是否为根节点（无父节点）"""
        return self.parent is None
    
    def is_frontier(self) -> bool:
        """判断是否在前沿（未被抽象）"""
        return not self.abstracted
    
    def set_abstracted(self, parent_node: 'FACENode') -> None:
        """
        设置节点为已抽象状态并建立父映射
        
        Args:
            parent_node: 父节点引用
        """
        if self.abstracted:
            raise ValueError(f"节点{self.id}已经被抽象，不能重复抽象")
        
        if parent_node.level != self.level + 1:
            raise ValueError(f"父节点层级{parent_node.level}不等于子节点层级+1 ({self.level+1})")
        
        self.abstracted = True
        self.parent = parent_node
        parent_node.children.append(self)
    
    def get_top_ancestor(self) -> 'FACENode':
        """
        获取最高祖先节点
        
        实现Top(x)函数：
        Top(x) = x if π(x) = ∅
                 Top(π(x)) otherwise
        """
        if self.is_root():
            return self
        return self.parent.get_top_ancestor()
    
    def get_ancestor_chain(self) -> List['FACENode']:
        """
        获取祖先链
        
        返回从当前节点到根节点的路径：[self, parent, grandparent, ..., root]
        """
        chain = [self]
        current = self
        while current.parent is not None:
            current = current.parent
            chain.append(current)
        return chain
    
    def get_ancestor_path(self) -> List['FACENode']:
        """
        获取祖先路径（不包含自身）
        
        实现Anc(c_n) = (a_d, a_{d-1}, ..., a_1)
        其中 a_d = Top(c_n), a_1 = π(c_n)
        """
        if self.is_root():
            return []
        
        path = []
        current = self.parent
        while current is not None:
            path.append(current)
            current = current.parent
        
        # 返回从最高祖先到直接父节点的路径
        return path[::-1]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'id': self.id,
            'level': self.level,
            'content': self.content,
            'token_count': self.token_count,
            't_now': self.t_now,
            'abstracted': self.abstracted,
            'parent_id': self.parent.id if self.parent else None,
            'children_ids': [child.id for child in self.children],
            'metadata': self.metadata
        }
    
    def __repr__(self) -> str:
        parent_id = self.parent.id if self.parent else None
        return f"FACENode(id={self.id}, level={self.level}, abstracted={self.abstracted}, parent={parent_id}, content_preview='{self.content[:50]}...')"


class LogicalClock:
    """
    逻辑时钟实现
    
    确保FACE系统的确定性和可重现性。
    每次调用tick()返回单调递增的整数时间戳。
    """
    
    def __init__(self, initial_time: int = 0):
        """
        初始化逻辑时钟
        
        Args:
            initial_time: 初始时间值
        """
        self._current_time = initial_time
    
    def tick(self) -> int:
        """获取下一个逻辑时间戳"""
        self._current_time += 1
        return self._current_time
    
    def current(self) -> int:
        """获取当前时间（不递增）"""
        return self._current_time
    
    def reset(self, time: int = 0) -> None:
        """重置时钟"""
        self._current_time = time


class NodeFactory:
    """
    FACE节点工厂
    
    负责创建符合FACE规范的节点，管理ID生成和时间戳。
    """
    
    def __init__(self, use_logical_clock: bool = True):
        """
        初始化节点工厂
        
        Args:
            use_logical_clock: 是否使用逻辑时钟（推荐True以确保可重现性）
        """
        self.use_logical_clock = use_logical_clock
        self.logical_clock = LogicalClock()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self._node_counter = 0
    
    def create_node(
        self, 
        level: int, 
        content: str, 
        node_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FACENode:
        """
        创建FACE节点
        
        Args:
            level: 节点层级
            content: 节点内容
            node_id: 可选的节点ID（默认自动生成）
            metadata: 可选元数据
            
        Returns:
            创建的FACE节点
        """
        if node_id is None:
            node_id = self._generate_node_id(level)
        
        # 计算token数量
        token_count = len(self.tokenizer.encode(content))
        
        # 获取时间戳
        if self.use_logical_clock:
            t_now = self.logical_clock.tick()
        else:
            t_now = time.time()
        
        return FACENode(
            id=node_id,
            level=level,
            content=content,
            token_count=token_count,
            t_now=t_now,
            metadata=metadata or {}
        )
    
    def _generate_node_id(self, level: int) -> str:
        """生成节点ID"""
        self._node_counter += 1
        if self.use_logical_clock:
            timestamp = self.logical_clock.current()
        else:
            timestamp = int(time.time() * 1000)
        
        return f"L{level}_N{self._node_counter}_{timestamp}"
    
    def reset_clock(self) -> None:
        """重置逻辑时钟（用于测试）"""
        if self.use_logical_clock:
            self.logical_clock.reset()
        self._node_counter = 0


def create_frontier_collection(nodes: List[FACENode]) -> List[FACENode]:
    """
    从节点列表中提取前沿节点
    
    实现 Frontier(Lₖ) = {N ∈ Lₖ | N.abstracted = False}
    
    Args:
        nodes: 节点列表
        
    Returns:
        前沿节点列表（按t_now排序）
    """
    frontier = [node for node in nodes if node.is_frontier()]
    # 按时间戳排序确保确定性
    frontier.sort(key=lambda n: n.t_now)
    return frontier


def find_common_ancestor(node1: FACENode, node2: FACENode) -> Optional[FACENode]:
    """
    查找两个节点的最低公共祖先（LCA）
    
    Args:
        node1: 第一个节点
        node2: 第二个节点
        
    Returns:
        最低公共祖先节点，如无共同祖先则返回None
    """
    # 获取两个节点的祖先链
    chain1 = node1.get_ancestor_chain()
    chain2 = node2.get_ancestor_chain()
    
    # 构建第一个节点的祖先集合
    ancestors1 = {node.id: node for node in chain1}
    
    # 在第二个节点的祖先链中查找第一个共同祖先
    for node in chain2:
        if node.id in ancestors1:
            return node
    
    return None


def validate_tree_structure(nodes: List[FACENode]) -> bool:
    """
    验证节点集合是否形成有效的树结构
    
    检查：
    1. 无环
    2. 每个节点最多有一个父节点
    3. 父节点层级 = 子节点层级 + 1
    
    Args:
        nodes: 要验证的节点列表
        
    Returns:
        True如果结构有效
        
    Raises:
        ValueError: 如果发现结构问题
    """
    node_dict = {node.id: node for node in nodes}
    
    for node in nodes:
        if node.parent is not None:
            # 检查父节点是否存在
            if node.parent.id not in node_dict:
                raise ValueError(f"节点{node.id}的父节点{node.parent.id}不在节点集合中")
            
            # 检查层级关系
            if node.parent.level != node.level + 1:
                raise ValueError(f"节点{node.id}(L{node.level})的父节点{node.parent.id}(L{node.parent.level})层级关系错误")
            
            # 检查父节点的children列表
            if node not in node.parent.children:
                raise ValueError(f"父节点{node.parent.id}的children列表中缺少子节点{node.id}")
    
    # 检查无环（通过DFS）
    visited = set()
    rec_stack = set()
    
    def has_cycle(node: FACENode) -> bool:
        if node.id in rec_stack:
            return True
        if node.id in visited:
            return False
        
        visited.add(node.id)
        rec_stack.add(node.id)
        
        if node.parent and has_cycle(node.parent):
            return True
        
        rec_stack.remove(node.id)
        return False
    
    for node in nodes:
        if node.id not in visited:
            if has_cycle(node):
                raise ValueError(f"检测到环：节点{node.id}参与了循环引用")
    
    return True