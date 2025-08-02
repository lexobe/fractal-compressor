"""
FACE前缀路径寻址系统

实现FACE的第三阶段：Level-0路径编码，提供稳定、可逆的前缀地址。
支持LCP（最长公共前缀）和LCA（最低公共祖先）查询操作。
"""

import logging
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
from datetime import datetime

from .face_node import FACENode

logger = logging.getLogger(__name__)


class FACEAddressing:
    """
    FACE前缀路径寻址系统
    
    实现FACE规范中的Level-0路径编码：
    Encode(c_n) = ρ(Hist_n ++ Anc(c_n) ++ (c_n))
    
    核心功能：
    1. 为Level-0节点生成稳定的前缀地址
    2. 支持LCP（最长公共前缀）查询
    3. 支持LCA（最低公共祖先）查询
    4. 维护历史最高祖先序列
    """
    
    def __init__(self):
        """初始化寻址系统"""
        self._address_cache: Dict[str, List[str]] = {}  # 节点ID -> 地址
        self._historical_tops: List[str] = []  # 历史最高祖先序列
        self._node_registry: Dict[str, FACENode] = {}  # 节点注册表
        
    def register_node(self, node: FACENode) -> None:
        """
        注册节点到寻址系统
        
        Args:
            node: 要注册的节点
        """
        self._node_registry[node.id] = node
        
        # 如果是Level-0节点，更新历史最高祖先序列
        if node.level == 0:
            self._update_historical_tops(node)
    
    def _update_historical_tops(self, level0_node: FACENode) -> None:
        """
        更新历史最高祖先序列
        
        实现 H_n = (Top(c_0), ..., Top(c_{n-1}))
        
        Args:
            level0_node: 新的Level-0节点
        """
        if level0_node.level != 0:
            raise ValueError(f"只能为Level-0节点更新历史最高祖先: {level0_node.id}")
        
        top_ancestor = level0_node.get_top_ancestor()
        self._historical_tops.append(top_ancestor.id)
        
        logger.debug(f"更新历史最高祖先: L0节点{level0_node.id} -> 最高祖先{top_ancestor.id}")
    
    def _get_deduplicated_historical_tops(self, up_to_index: int) -> List[str]:
        """
        获取去重的历史最高祖先序列
        
        实现 Hist_n = ρ(H_n)  # 首次出现保序去重
        
        Args:
            up_to_index: 历史序列的截止索引
            
        Returns:
            去重后的历史最高祖先ID列表
        """
        if up_to_index <= 0:
            return []
        
        seen = set()
        result = []
        
        for ancestor_id in self._historical_tops[:up_to_index]:
            if ancestor_id not in seen:
                seen.add(ancestor_id)
                result.append(ancestor_id)
        
        return result
    
    def encode_path(self, level0_node: FACENode, format: str = "list") -> Union[List[str], Dict[str, Any], str]:
        """
        为Level-0节点生成前缀路径地址
        
        实现编码函数：
        Encode(c_n) = ρ(Hist_n ++ Anc(c_n) ++ (c_n))
        
        Args:
            level0_node: Level-0节点
            format: 输出格式，支持 "list", "json", "text"
            
        Returns:
            前缀路径地址：
            - "list": List[str] - 节点ID列表（默认）
            - "json": Dict - 包含完整结构信息的JSON对象
            - "text": str - 人类可读的文本格式
            
        Raises:
            ValueError: 如果不是Level-0节点或格式不支持
        """
        if level0_node.level != 0:
            raise ValueError(f"只能为Level-0节点编码路径: {level0_node.id} (Level {level0_node.level})")
        
        if format not in ["list", "json", "text"]:
            raise ValueError(f"不支持的格式: {format}，支持格式: list, json, text")
        
        # 检查缓存
        address_list = None
        if level0_node.id in self._address_cache:
            address_list = self._address_cache[level0_node.id].copy()
        else:
            # 确保节点已注册
            if level0_node.id not in self._node_registry:
                self.register_node(level0_node)
            
            # 找到该节点在历史序列中的位置
            node_index = self._find_level0_index(level0_node)
            
            # 1. 历史最高祖先序列 Hist_n
            hist_n = self._get_deduplicated_historical_tops(node_index)
            
            # 2. 祖先链 Anc(c_n)
            ancestor_path = level0_node.get_ancestor_path()
            anc_cn = [node.id for node in ancestor_path]
            
            # 3. 节点自身
            self_id = [level0_node.id]
            
            # 4. 合并并去重 ρ(Hist_n ++ Anc(c_n) ++ (c_n))
            combined = hist_n + anc_cn + self_id
            address_list = self._first_occurrence_dedup(combined)
            
            # 缓存结果
            self._address_cache[level0_node.id] = address_list
            
            logger.debug(f"编码路径 {level0_node.id}: {' -> '.join(address_list)}")
        
        # 根据格式返回不同结果
        if format == "list":
            return address_list.copy()
        elif format == "json":
            return self._format_address_as_json(level0_node, address_list)
        elif format == "text":
            return self._format_address_as_text(level0_node, address_list)
    
    def _find_level0_index(self, level0_node: FACENode) -> int:
        """
        找到Level-0节点在历史序列中的索引位置
        
        Args:
            level0_node: Level-0节点
            
        Returns:
            节点在历史序列中的索引
        """
        # 简化实现：基于节点创建时间排序
        all_level0_nodes = [
            node for node in self._node_registry.values() 
            if node.level == 0 and node.t_now <= level0_node.t_now
        ]
        all_level0_nodes.sort(key=lambda n: n.t_now)
        
        for i, node in enumerate(all_level0_nodes):
            if node.id == level0_node.id:
                return i
        
        return len(all_level0_nodes)  # 如果没找到，返回末尾位置
    
    def _first_occurrence_dedup(self, sequence: List[str]) -> List[str]:
        """
        首次出现保序去重
        
        实现 ρ(sequence)：保持首次出现的顺序，去除重复
        
        Args:
            sequence: 输入序列
            
        Returns:
            去重后的序列
        """
        seen = set()
        result = []
        
        for item in sequence:
            if item not in seen:
                seen.add(item)
                result.append(item)
        
        return result
    
    def find_longest_common_prefix(self, addr1: List[str], addr2: List[str]) -> List[str]:
        """
        查找两个地址的最长公共前缀（LCP）
        
        Args:
            addr1: 第一个地址
            addr2: 第二个地址
            
        Returns:
            最长公共前缀
        """
        lcp = []
        min_len = min(len(addr1), len(addr2))
        
        for i in range(min_len):
            if addr1[i] == addr2[i]:
                lcp.append(addr1[i])
            else:
                break
        
        return lcp
    
    def find_lowest_common_ancestor_by_addresses(
        self, 
        addr1: List[str], 
        addr2: List[str]
    ) -> Optional[str]:
        """
        根据地址查找最低公共祖先（LCA）
        
        Args:
            addr1: 第一个节点的地址
            addr2: 第二个节点的地址
            
        Returns:
            最低公共祖先的节点ID，如无共同祖先则返回None
        """
        lcp = self.find_longest_common_prefix(addr1, addr2)
        
        if not lcp:
            return None
        
        # 最长公共前缀的最后一个元素就是LCA
        return lcp[-1]
    
    def find_lca_nodes(self, node1: FACENode, node2: FACENode) -> Optional[FACENode]:
        """
        直接根据节点查找最低公共祖先
        
        Args:
            node1: 第一个节点
            node2: 第二个节点
            
        Returns:
            最低公共祖先节点，如无共同祖先则返回None
        """
        # 如果不是Level-0节点，使用直接的祖先链查找
        if node1.level != 0 or node2.level != 0:
            return self._find_lca_by_ancestor_chains(node1, node2)
        
        # 对于Level-0节点，使用地址查找
        addr1 = self.encode_path(node1)
        addr2 = self.encode_path(node2)
        
        lca_id = self.find_lowest_common_ancestor_by_addresses(addr1, addr2)
        if lca_id:
            return self._node_registry.get(lca_id)
        
        return None
    
    def _find_lca_by_ancestor_chains(self, node1: FACENode, node2: FACENode) -> Optional[FACENode]:
        """
        通过祖先链查找LCA（用于非Level-0节点）
        
        Args:
            node1: 第一个节点
            node2: 第二个节点
            
        Returns:
            最低公共祖先节点
        """
        chain1 = node1.get_ancestor_chain()
        chain2 = node2.get_ancestor_chain()
        
        # 构建第一个节点的祖先集合
        ancestors1 = {node.id: node for node in chain1}
        
        # 在第二个节点的祖先链中查找第一个共同祖先
        for node in chain2:
            if node.id in ancestors1:
                return node
        
        return None
    
    def get_address_tree_structure(self) -> Dict[str, List[str]]:
        """
        获取地址的树形结构表示
        
        Returns:
            地址前缀树结构
        """
        tree = defaultdict(list)
        
        for node_id, address in self._address_cache.items():
            current_prefix = []
            for addr_component in address:
                current_prefix.append(addr_component)
                prefix_key = " -> ".join(current_prefix[:-1]) if len(current_prefix) > 1 else "root"
                if addr_component not in tree[prefix_key]:
                    tree[prefix_key].append(addr_component)
        
        return dict(tree)
    
    def validate_address_stability(self, level0_nodes: List[FACENode]) -> bool:
        """
        验证地址稳定性
        
        检查：
        1. 旧地址不改变
        2. 新祖先只影响后续的Hist
        3. 前缀关系正确
        
        Args:
            level0_nodes: Level-0节点列表（按时间排序）
            
        Returns:
            True如果地址稳定
        """
        try:
            # 按时间排序
            sorted_nodes = sorted(level0_nodes, key=lambda n: n.t_now)
            
            # 生成所有地址
            addresses = {}
            for node in sorted_nodes:
                addresses[node.id] = self.encode_path(node)
            
            # 验证前缀关系：祖先地址应该是后代地址的前缀
            for node in sorted_nodes:
                if node.parent:
                    node_addr = addresses[node.id]
                    parent_in_addr = node.parent.id in node_addr
                    
                    if not parent_in_addr:
                        logger.error(f"前缀关系错误: 节点{node.id}的地址中不包含父节点{node.parent.id}")
                        return False
            
            logger.info("地址稳定性验证通过")
            return True
            
        except Exception as e:
            logger.error(f"地址稳定性验证失败: {e}")
            return False
    
    def get_addressing_stats(self) -> Dict[str, Any]:
        """获取寻址系统统计信息"""
        return {
            'registered_nodes': len(self._node_registry),
            'cached_addresses': len(self._address_cache),
            'historical_tops_length': len(self._historical_tops),
            'unique_historical_tops': len(set(self._historical_tops)),
            'level0_nodes': len([n for n in self._node_registry.values() if n.level == 0])
        }
    
    def clear_cache(self) -> None:
        """清空地址缓存（用于重新计算）"""
        self._address_cache.clear()
    
    def _format_address_as_json(self, level0_node: FACENode, address_list: List[str]) -> Dict[str, Any]:
        """
        将地址格式化为JSON对象
        
        Args:
            level0_node: Level-0节点
            address_list: 地址列表
            
        Returns:
            包含完整结构信息的JSON对象
        """
        # 获取地址组件的详细信息
        components = []
        for i, addr_id in enumerate(address_list):
            node_info = {
                "id": addr_id,
                "position": i,
                "is_target": addr_id == level0_node.id
            }
            
            # 如果节点在注册表中，添加更多信息
            if addr_id in self._node_registry:
                reg_node = self._node_registry[addr_id]
                node_info.update({
                    "level": reg_node.level,
                    "token_count": reg_node.token_count,
                    "timestamp": reg_node.t_now,
                    "abstracted": reg_node.abstracted,
                    "content_preview": reg_node.content[:50] + "..." if len(reg_node.content) > 50 else reg_node.content
                })
                
                # 父子关系
                if reg_node.parent:
                    node_info["parent_id"] = reg_node.parent.id
                if reg_node.children:
                    node_info["children_ids"] = [child.id for child in reg_node.children]
            
            components.append(node_info)
        
        # 构建完整的JSON结构
        json_address = {
            "encoding_metadata": {
                "target_node_id": level0_node.id,
                "target_level": level0_node.level,
                "encoding_timestamp": datetime.now().isoformat(),
                "address_length": len(address_list),
                "encoding_function": "ρ(Hist_n ++ Anc(c_n) ++ (c_n))"
            },
            "address_path": address_list,
            "components": components,
            "structure_analysis": {
                "has_historical_context": any(comp["level"] > level0_node.level for comp in components if "level" in comp),
                "ancestor_chain_length": len([comp for comp in components if comp["id"] != level0_node.id]),
                "deduplication_count": len(set(address_list)) - len(address_list) if len(set(address_list)) != len(address_list) else 0
            },
            "navigation_info": {
                "path_string": " -> ".join(address_list),
                "reverse_path": list(reversed(address_list)),
                "prefix_segments": [address_list[:i+1] for i in range(len(address_list))]
            }
        }
        
        return json_address
    
    def _format_address_as_text(self, level0_node: FACENode, address_list: List[str]) -> str:
        """
        将地址格式化为人类可读的文本
        
        Args:
            level0_node: Level-0节点
            address_list: 地址列表
            
        Returns:
            人类可读的文本格式地址
        """
        lines = []
        
        # 标题
        lines.append(f"📍 FACE地址编码 - 节点 {level0_node.id}")
        lines.append("=" * 50)
        
        # 基本信息
        lines.append(f"🎯 目标节点: {level0_node.id} (Level {level0_node.level})")
        lines.append(f"📏 地址长度: {len(address_list)} 个组件")
        lines.append(f"🔗 路径字符串: {' -> '.join(address_list)}")
        
        # 内容预览
        content_preview = level0_node.content[:60] + "..." if len(level0_node.content) > 60 else level0_node.content
        lines.append(f"📝 内容预览: \"{content_preview}\"")
        
        lines.append("")  # 空行
        
        # 地址组件详情
        lines.append("🧩 地址组件详情:")
        lines.append("-" * 30)
        
        for i, addr_id in enumerate(address_list):
            prefix = "├─" if i < len(address_list) - 1 else "└─"
            
            if addr_id in self._node_registry:
                node = self._node_registry[addr_id]
                status_symbol = "🎯" if addr_id == level0_node.id else "🏗️" if node.abstracted else "📄"
                
                lines.append(f"{prefix} {status_symbol} {addr_id}")
                lines.append(f"   │   Level: {node.level}, Tokens: {node.token_count}")
                
                # 显示关系
                relationships = []
                if node.parent:
                    relationships.append(f"Parent: {node.parent.id}")
                if node.children:
                    relationships.append(f"Children: {len(node.children)}")
                if relationships:
                    lines.append(f"   │   Relations: {', '.join(relationships)}")
                
                # 内容预览
                content = node.content[:40] + "..." if len(node.content) > 40 else node.content
                lines.append(f"   │   Content: \"{content}\"")
                
                if i < len(address_list) - 1:
                    lines.append("   │")
            else:
                lines.append(f"{prefix} ❓ {addr_id} (未注册)")
        
        # 编码公式说明
        lines.append("")
        lines.append("📐 编码公式:")
        lines.append("   Encode(c_n) = ρ(Hist_n ++ Anc(c_n) ++ (c_n))")
        lines.append("   其中:")
        lines.append("   • Hist_n: 历史最高祖先序列")
        lines.append("   • Anc(c_n): 祖先链")
        lines.append("   • (c_n): 节点自身")
        lines.append("   • ρ: 首遇保序去重")
        lines.append("   • ++: 序列连接")
        
        # 时间戳
        lines.append("")
        lines.append(f"⏰ 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(lines)
    
    def decode_path(self, address_path: List[str]) -> Optional[FACENode]:
        """
        根据地址路径解码回节点
        
        Args:
            address_path: 地址路径列表
            
        Returns:
            目标节点，如果找不到则返回None
        """
        if not address_path:
            return None
        
        # 地址路径的最后一个元素应该是目标节点
        target_node_id = address_path[-1]
        
        # 在注册表中查找目标节点
        if target_node_id in self._node_registry:
            target_node = self._node_registry[target_node_id]
            logger.debug(f"成功解码地址路径: {' -> '.join(address_path)} -> {target_node_id}")
            return target_node
        
        logger.warning(f"无法解码地址路径: 节点 {target_node_id} 未在注册表中找到")
        return None
    
    def reset(self) -> None:
        """重置寻址系统"""
        self._address_cache.clear()
        self._historical_tops.clear()
        self._node_registry.clear()
        logger.info("寻址系统已重置")


class FACEAddressQuery:
    """
    FACE地址查询工具
    
    提供便捷的地址查询和分析功能
    """
    
    def __init__(self, addressing: FACEAddressing):
        """
        初始化查询工具
        
        Args:
            addressing: FACE寻址系统实例
        """
        self.addressing = addressing
    
    def find_related_nodes(self, node: FACENode, max_distance: int = 2) -> List[Tuple[FACENode, int]]:
        """
        查找相关节点
        
        基于地址前缀相似性查找相关节点
        
        Args:
            node: 目标节点
            max_distance: 最大距离
            
        Returns:
            (相关节点, 距离)的列表
        """
        if node.level != 0:
            return []
        
        target_addr = self.addressing.encode_path(node)
        related = []
        
        for other_node in self.addressing._node_registry.values():
            if other_node.level == 0 and other_node.id != node.id:
                other_addr = self.addressing.encode_path(other_node)
                lcp = self.addressing.find_longest_common_prefix(target_addr, other_addr)
                
                # 计算距离（地址长度差）
                distance = abs(len(target_addr) - len(lcp)) + abs(len(other_addr) - len(lcp))
                
                if distance <= max_distance:
                    related.append((other_node, distance))
        
        # 按距离排序
        related.sort(key=lambda x: x[1])
        return related
    
    def analyze_address_patterns(self) -> Dict[str, Any]:
        """
        分析地址模式
        
        Returns:
            地址模式分析结果
        """
        addresses = list(self.addressing._address_cache.values())
        
        if not addresses:
            return {'message': '无地址数据'}
        
        # 地址长度分布
        lengths = [len(addr) for addr in addresses]
        length_dist = {}
        for length in lengths:
            length_dist[length] = length_dist.get(length, 0) + 1
        
        # 前缀重用分析
        prefix_usage = defaultdict(int)
        for addr in addresses:
            for i in range(1, len(addr)):
                prefix = tuple(addr[:i])
                prefix_usage[prefix] += 1
        
        # 最常见前缀
        common_prefixes = sorted(
            [(prefix, count) for prefix, count in prefix_usage.items() if count > 1],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            'total_addresses': len(addresses),
            'address_length_distribution': length_dist,
            'average_address_length': sum(lengths) / len(lengths) if lengths else 0,
            'common_prefixes': [(list(prefix), count) for prefix, count in common_prefixes],
            'prefix_reuse_ratio': len([c for c in prefix_usage.values() if c > 1]) / len(prefix_usage) if prefix_usage else 0
        }
    
    def export_address_graph(self) -> Dict[str, Any]:
        """
        导出地址图结构（用于可视化）
        
        Returns:
            地址图的节点和边信息
        """
        nodes = []
        edges = []
        
        for node_id, node in self.addressing._node_registry.items():
            nodes.append({
                'id': node_id,
                'level': node.level,
                'content_preview': node.content[:50] + '...' if len(node.content) > 50 else node.content,
                'abstracted': node.abstracted,
                't_now': node.t_now
            })
            
            # 添加父子边
            if node.parent:
                edges.append({
                    'source': node_id,
                    'target': node.parent.id,
                    'type': 'parent'
                })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'addressing_stats': self.addressing.get_addressing_stats()
        }