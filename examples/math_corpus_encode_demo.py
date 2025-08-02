#!/usr/bin/env python3
"""
数学语料FACE编码演示

使用β=4, k=2参数处理费马大定理数学语料，
展示FACE系统的内容编码结果（JSON和TEXT两种格式）
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from fractal_compressor import FACEEncoder, FACEAddressing


class FACEContentEncoder:
    """FACE内容编码器 - 支持多格式输出"""
    
    def __init__(self, encoder: FACEEncoder, addressing: FACEAddressing):
        self.encoder = encoder
        self.addressing = addressing
    
    def encode(self, texts: list, format: str = "json"):
        """
        编码文本内容
        
        Args:
            texts: 输入文本列表
            format: 输出格式 "json" 或 "text"
            
        Returns:
            编码结果（格式依据format参数）
        """
        if format not in ["json", "text"]:
            raise ValueError(f"不支持的格式: {format}，支持: json, text")
        
        # 重置编码器
        self.encoder.reset()
        self.addressing = FACEAddressing()  # 重新初始化寻址系统
        
        # 处理所有文本
        print(f"🔄 开始处理{len(texts)}段数学语料...")
        for i, text in enumerate(texts, 1):
            self.encoder.append_text(text + "\n\n")
            print(f"   {i}. 已添加 {len(text)} 字符")
        
        # 处理缓冲区
        self.encoder.process_buffer()
        
        # 注册节点到寻址系统
        for node in self.encoder.all_nodes.values():
            self.addressing.register_node(node)
        
        # 收集编码结果
        encoding_result = self._collect_encoding_data()
        
        # 根据格式返回结果
        if format == "json":
            return self._format_as_json(encoding_result)
        else:  # text
            return self._format_as_text(encoding_result)
    
    def _collect_encoding_data(self):
        """收集编码数据"""
        # 获取系统状态
        system_summary = self.encoder.get_system_summary()
        
        # 收集各层级节点信息
        layer_contents = {}
        for level in range(system_summary['active_levels']):
            layer_nodes = []
            for node in self.encoder.all_nodes.values():
                if node.level == level:
                    node_info = {
                        'id': node.id,
                        'content': node.content,
                        'token_count': node.token_count,
                        'abstracted': node.abstracted,
                        'timestamp': node.t_now,
                        'parent_id': node.parent.id if node.parent else None,
                        'children_ids': [child.id for child in node.children] if node.children else []
                    }
                    layer_nodes.append(node_info)
            
            # 按时间戳排序
            layer_nodes.sort(key=lambda x: x['timestamp'])
            layer_contents[f"L{level}"] = layer_nodes
        
        # 收集地址信息
        address_info = {}
        level0_nodes = self.encoder.get_level0_nodes()
        for node in level0_nodes[:5]:  # 前5个节点的地址
            try:
                address = self.addressing.encode_path(node)
                address_info[node.id] = address
            except Exception as e:
                address_info[node.id] = f"地址生成失败: {e}"
        
        return {
            'system_summary': system_summary,
            'layer_contents': layer_contents,
            'addressing_info': address_info,
            'parameters': {
                'beta': self.encoder.beta,
                'k': self.encoder.k,
                'tau_buffer': self.encoder.tau_buffer,
                'tau_slice': self.encoder.tau_slice
            }
        }
    
    def _format_as_json(self, data):
        """格式化为JSON"""
        json_result = {
            "face_encoding_result": {
                "metadata": {
                    "encoding_timestamp": datetime.now().isoformat(),
                    "encoder_version": "FACE v1.0",
                    "total_nodes": data['system_summary']['total_nodes'],
                    "active_levels": data['system_summary']['active_levels'],
                    "parameters": data['parameters']
                },
                "layer_hierarchy": {},
                "content_abstraction": [],
                "addressing_map": data['addressing_info'],
                "system_stats": data['system_summary']
            }
        }
        
        # 构建层级内容
        for layer_key, nodes in data['layer_contents'].items():
            level = int(layer_key[1:])  # 提取数字
            json_result["face_encoding_result"]["layer_hierarchy"][layer_key] = {
                "level": level,
                "node_count": len(nodes),
                "nodes": nodes
            }
        
        # 构建抽象链条
        for level in range(data['system_summary']['active_levels']):
            layer_key = f"L{level}"
            if layer_key in data['layer_contents']:
                for node in data['layer_contents'][layer_key]:
                    abstraction_entry = {
                        "source_level": level,
                        "node_id": node['id'],
                        "content_preview": node['content'][:100] + "..." if len(node['content']) > 100 else node['content'],
                        "token_count": node['token_count'],
                        "abstraction_status": "abstracted" if node['abstracted'] else "frontier"
                    }
                    json_result["face_encoding_result"]["content_abstraction"].append(abstraction_entry)
        
        return json_result
    
    def _format_as_text(self, data):
        """格式化为文本"""
        lines = []
        
        # 标题
        lines.append("🧮 FACE数学语料编码结果")
        lines.append("=" * 60)
        
        # 参数信息
        params = data['parameters']
        lines.append(f"📊 编码参数: β={params['beta']}, k={params['k']}, τ_buffer={params['tau_buffer']}, τ_slice={params['tau_slice']}")
        lines.append(f"📈 系统状态: {data['system_summary']['total_nodes']}个节点, {data['system_summary']['active_levels']}个层级")
        lines.append("")
        
        # 层级内容展示
        lines.append("🏗️  层级结构内容:")
        lines.append("-" * 40)
        
        for level in range(data['system_summary']['active_levels']):
            layer_key = f"L{level}"
            if layer_key in data['layer_contents']:
                nodes = data['layer_contents'][layer_key]
                lines.append(f"\n📁 Level {level} ({len(nodes)}个节点):")
                
                for i, node in enumerate(nodes):
                    status_icon = "🏗️" if node['abstracted'] else "📄"
                    lines.append(f"   {status_icon} {node['id']} ({node['token_count']} tokens)")
                    
                    # 内容预览
                    content_preview = node['content'][:80].replace('\n', ' ')
                    if len(node['content']) > 80:
                        content_preview += "..."
                    lines.append(f"      \"{content_preview}\"")
                    
                    # 关系信息
                    if node['parent_id']:
                        lines.append(f"      ↗️  Parent: {node['parent_id']}")
                    if node['children_ids']:
                        lines.append(f"      ↘️  Children: {', '.join(node['children_ids'])}")
                    
                    if i < len(nodes) - 1:
                        lines.append("")
        
        # 地址映射
        lines.append("\n🗺️  前缀地址映射:")
        lines.append("-" * 40)
        
        for node_id, address in data['addressing_info'].items():
            if isinstance(address, list):
                address_str = " → ".join(address)
                lines.append(f"📍 {node_id}: {address_str}")
            else:
                lines.append(f"📍 {node_id}: {address}")
        
        # 统计总结
        lines.append(f"\n📊 编码统计:")
        lines.append("-" * 40)
        
        layer_stats = data['system_summary']['layer_stats']
        for level, stats in layer_stats.items():
            lines.append(f"L{level}: {stats['total_nodes']}节点, {stats['frontier_nodes']}前沿, 平均{stats['avg_tokens']:.1f}tokens")
        
        # 时间戳
        lines.append(f"\n⏰ 编码完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(lines)


def main():
    """主函数"""
    print("🧮 FACE数学语料编码演示 (β=4, k=2)")
    print("=" * 60)
    
    # 费马大定理数学语料
    math_corpus = [
        "费马大定理由皮埃尔·德·费马在1637年提出。该定理表述为：对于大于2的正整数n，不存在正整数x、y、z使得x^n + y^n = z^n成立。",
        
        "这个看似简单的数学陈述困扰了数学家们358年。费马在阅读丢番图的《算术》时写下了这个猜想，声称有一个美妙的证明，但书页边缘太窄写不下。",
        
        "欧拉在1770年证明了n=3的情况，使用了无穷递降法。证明思路是假设存在最小的正整数解，然后构造更小的解导致矛盾。",
        
        "费马本人证明了n=4的情况。这实际上是寻找直角三角形，其中两条直角边都是完全平方数，通过数论分析证明不存在。",
        
        "恩斯特·库默尔发展了理想数理论，研究分圆域中的唯一分解问题。他证明了对于正则素数p，费马大定理成立。",
        
        "谷山丰和志村五郎提出了椭圆曲线和模形式之间联系的深刻猜想。这建立了椭圆曲线理论和模形式理论之间的桥梁。"
    ]
    
    # 创建FACE编码器 (β=4, k=2)
    encoder = FACEEncoder(
        tau_buffer=80,
        tau_slice=60,
        beta=4,        # β=4: 批量大小
        k=2,           # k=2: 尾部保留数量  
        use_logical_clock=True
    )
    
    addressing = FACEAddressing()
    content_encoder = FACEContentEncoder(encoder, addressing)
    
    print(f"📝 数学语料: {len(math_corpus)}段关于费马大定理的文本")
    print(f"⚙️  编码参数: β={encoder.beta}, k={encoder.k}")
    
    # 演示JSON格式
    print(f"\n" + "="*60)
    print(f"📋 格式1: JSON编码结果")
    print(f"="*60)
    
    try:
        json_result = content_encoder.encode(math_corpus, format="json")
        
        # 显示JSON概览
        metadata = json_result["face_encoding_result"]["metadata"]
        print(f"✅ JSON编码成功")
        print(f"   总节点数: {metadata['total_nodes']}")
        print(f"   活跃层级: {metadata['active_levels']}")
        print(f"   编码参数: β={metadata['parameters']['beta']}, k={metadata['parameters']['k']}")
        
        # 保存JSON结果
        json_file = Path(__file__).parent / "math_corpus_encode_result.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, ensure_ascii=False, indent=2)
        print(f"📁 JSON结果已保存: {json_file}")
        
        # 显示部分JSON内容
        print(f"\n🔍 JSON结构概览:")
        for key in json_result["face_encoding_result"].keys():
            if key == "layer_hierarchy":
                layer_count = len(json_result["face_encoding_result"][key])
                print(f"   {key}: {layer_count}个层级")
            elif key == "content_abstraction":
                abstraction_count = len(json_result["face_encoding_result"][key])
                print(f"   {key}: {abstraction_count}个抽象条目")
            else:
                print(f"   {key}: {type(json_result['face_encoding_result'][key]).__name__}")
        
    except Exception as e:
        print(f"❌ JSON编码失败: {e}")
        return False
    
    # 演示TEXT格式
    print(f"\n" + "="*60)
    print(f"📝 格式2: TEXT编码结果")
    print(f"="*60)
    
    try:
        # 重新编码以确保独立性
        text_result = content_encoder.encode(math_corpus, format="text")
        
        print(f"✅ TEXT编码成功")
        print(f"📏 文本长度: {len(text_result)} 字符")
        print(f"📄 文本行数: {len(text_result.splitlines())} 行")
        
        # 显示TEXT内容
        print(f"\n📖 TEXT编码结果:")
        print("-" * 60)
        print(text_result)
        print("-" * 60)
        
        # 保存TEXT结果
        text_file = Path(__file__).parent / "math_corpus_encode_result.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text_result)
        print(f"📁 TEXT结果已保存: {text_file}")
        
    except Exception as e:
        print(f"❌ TEXT编码失败: {e}")
        return False
    
    print(f"\n🎉 FACE数学语料编码演示完成！")
    print(f"✅ JSON格式: 结构化编码数据")
    print(f"✅ TEXT格式: 人类可读编码结果")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)