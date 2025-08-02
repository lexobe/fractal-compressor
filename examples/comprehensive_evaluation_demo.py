#!/usr/bin/env python3
"""
FACE编码结果全面评价系统

设计目标：
1. 显示完整原文
2. 逐个显示编码地址还原的文字
3. 对每个还原结果进行质量评价
4. 提供便于大语料评价的机制
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
from datetime import datetime

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from fractal_compressor import FACEEncoder, FACEAddressing


class ComprehensiveEvaluator:
    """全面的FACE编码结果评价器"""
    
    def __init__(self):
        self.evaluation_results = []
    
    def evaluate_corpus(self, 
                       texts: List[str], 
                       beta: int = 4, 
                       k: int = 2,
                       corpus_name: str = "语料") -> Dict[str, Any]:
        """
        全面评价语料的编码质量
        
        Args:
            texts: 原始文本列表
            beta: 批量大小
            k: 尾部保留数
            corpus_name: 语料名称
            
        Returns:
            完整的评价报告
        """
        print(f"🔍 {corpus_name}全面评价系统")
        print("=" * 80)
        
        # 1. 显示完整原文
        original_content = self._display_original_corpus(texts)
        
        # 2. 执行FACE编码
        encoder, addressing = self._perform_encoding(texts, beta, k)
        
        # 3. 提取所有Level-0节点地址
        level0_addresses = self._extract_level0_addresses(encoder, addressing)
        
        # 4. 逐个还原并评价
        restoration_results = self._evaluate_each_restoration(
            level0_addresses, addressing, original_content, corpus_name
        )
        
        # 5. 整体质量分析
        overall_analysis = self._perform_overall_analysis(
            original_content, restoration_results, encoder
        )
        
        # 6. 生成评价报告
        evaluation_report = self._generate_evaluation_report(
            corpus_name, original_content, restoration_results, 
            overall_analysis, beta, k
        )
        
        return evaluation_report
    
    def _display_original_corpus(self, texts: List[str]) -> str:
        """显示完整原文"""
        print("📖 完整原文语料")
        print("-" * 60)
        
        combined_text = ""
        for i, text in enumerate(texts, 1):
            print(f"{i:2d}. {text}")
            print(f"    长度: {len(text)} 字符")
            print()
            combined_text += text + "\n\n"
        
        print(f"📊 原文统计:")
        print(f"   总段落数: {len(texts)}")
        print(f"   总字符数: {len(combined_text)} (含分隔符)")
        print(f"   纯文本数: {sum(len(t) for t in texts)} 字符")
        print()
        
        return combined_text
    
    def _perform_encoding(self, texts: List[str], beta: int, k: int) -> Tuple[FACEEncoder, FACEAddressing]:
        """执行FACE编码"""
        print("🔄 执行FACE编码")
        print("-" * 60)
        
        # 创建编码器
        encoder = FACEEncoder(
            tau_buffer=80,
            tau_slice=60,
            beta=beta,
            k=k,
            use_logical_clock=True
        )
        
        # 逐步添加文本
        for i, text in enumerate(texts, 1):
            print(f"   添加第{i}段: {len(text)} 字符")
            encoder.append_text(text + "\n\n")
        
        # 处理剩余缓冲区
        encoder.process_buffer()
        
        # 创建地址系统
        addressing = FACEAddressing()
        
        # 注册所有节点到寻址系统
        for node in encoder.all_nodes.values():
            addressing.register_node(node)
        
        # 显示编码统计
        stats = encoder.get_system_summary()
        print(f"   ✅ 编码完成: {stats['total_nodes']} 个节点, {stats['active_levels']} 个层级")
        print()
        
        return encoder, addressing
    
    def _extract_level0_addresses(self, encoder: FACEEncoder, addressing: FACEAddressing) -> List[Tuple[str, List[str]]]:
        """提取所有Level-0节点的地址"""
        print("🗺️  提取Level-0节点地址")
        print("-" * 60)
        
        level0_nodes = encoder.get_level0_nodes()
        level0_nodes.sort(key=lambda n: n.t_now)  # 按时间顺序排序
        
        addresses = []
        for i, node in enumerate(level0_nodes, 1):
            address_path = addressing.encode_path(node, format="list")
            addresses.append((node.id, address_path))
            print(f"   {i:2d}. {node.id}: {' → '.join(address_path)}")
        
        print(f"   ✅ 提取了 {len(addresses)} 个地址")
        print()
        
        return addresses
    
    def _evaluate_each_restoration(self, 
                                 addresses: List[Tuple[str, List[str]]], 
                                 addressing: FACEAddressing,
                                 original_content: str,
                                 corpus_name: str) -> List[Dict[str, Any]]:
        """逐个还原并评价每个地址"""
        print("🔍 逐个地址还原与评价")
        print("=" * 80)
        
        restoration_results = []
        
        for i, (node_id, address_path) in enumerate(addresses, 1):
            print(f"\n📍 地址 {i}: {node_id}")
            print(f"   路径: {' → '.join(address_path)}")
            
            # 还原内容
            try:
                restored_node = addressing.decode_path(address_path)
                if restored_node:
                    restored_content = restored_node.content
                    print(f"   ✅ 还原成功")
                else:
                    restored_content = "[还原失败]"
                    print(f"   ❌ 还原失败")
            except Exception as e:
                restored_content = f"[错误: {e}]"
                print(f"   ❌ 还原错误: {e}")
            
            # 显示还原内容
            print(f"   📝 还原内容:")
            if len(restored_content) <= 100:
                print(f"      \"{restored_content}\"")
            else:
                print(f"      \"{restored_content[:50]}...{restored_content[-50:]}\"")
                print(f"      (完整长度: {len(restored_content)} 字符)")
            
            # 质量评价
            quality_score = self._evaluate_content_quality(restored_content, original_content, corpus_name)
            
            # 记录结果
            result = {
                'index': i,
                'node_id': node_id,
                'address_path': address_path,
                'restored_content': restored_content,
                'content_length': len(restored_content),
                'quality_score': quality_score,
                'is_valid': '[' not in restored_content,  # 简单检查是否有错误标记
            }
            restoration_results.append(result)
            
            print(f"   📊 质量评分: {quality_score['overall_score']:.1f}/10")
            if quality_score['issues']:
                for issue in quality_score['issues']:
                    print(f"      ⚠️  {issue}")
            
            print("-" * 40)
        
        return restoration_results
    
    def _evaluate_content_quality(self, content: str, original_corpus: str, corpus_name: str) -> Dict[str, Any]:
        """评价单个内容的质量"""
        issues = []
        quality_factors = {}
        
        # 1. 长度合理性 (0-2分)
        if len(content) < 10:
            issues.append("内容过短")
            quality_factors['length'] = 0.5
        elif len(content) > 500:
            issues.append("内容过长")
            quality_factors['length'] = 1.5
        else:
            quality_factors['length'] = 2.0
        
        # 2. 语义完整性 (0-3分)
        if content.startswith('[') or content.endswith(']'):
            issues.append("包含错误标记")
            quality_factors['integrity'] = 0.0
        elif not content.strip():
            issues.append("内容为空")
            quality_factors['integrity'] = 0.0
        elif content.count('。') == 0 and len(content) > 20:
            issues.append("可能缺少句子结尾")
            quality_factors['integrity'] = 1.5
        else:
            quality_factors['integrity'] = 3.0
        
        # 3. 字符质量 (0-2分)
        if any(char in content for char in ['下欧', '论分', '式理']):
            issues.append("发现疑似截断字符")
            quality_factors['character'] = 0.5
        elif content.startswith((' ', '\n')) or content.endswith((' ', '\n')):
            issues.append("开头或结尾有多余空白")
            quality_factors['character'] = 1.5
        else:
            quality_factors['character'] = 2.0
        
        # 4. 内容相关性 (0-3分)
        if corpus_name == "费马大定理" or "数学" in corpus_name:
            math_terms = ['费马', '定理', '证明', '数学', '欧拉', '方程', '整数']
            found_terms = sum(1 for term in math_terms if term in content)
            if found_terms >= 2:
                quality_factors['relevance'] = 3.0
            elif found_terms >= 1:
                quality_factors['relevance'] = 2.0
            else:
                issues.append("与数学主题相关性较低")
                quality_factors['relevance'] = 1.0
        else:
            quality_factors['relevance'] = 2.5  # 默认分数
        
        # 计算总分
        overall_score = sum(quality_factors.values())
        
        return {
            'overall_score': overall_score,
            'max_score': 10.0,
            'quality_factors': quality_factors,
            'issues': issues,
            'is_excellent': overall_score >= 9.0,
            'is_good': overall_score >= 7.0,
            'is_acceptable': overall_score >= 5.0
        }
    
    def _perform_overall_analysis(self, 
                                original_content: str, 
                                restoration_results: List[Dict[str, Any]], 
                                encoder: FACEEncoder) -> Dict[str, Any]:
        """执行整体质量分析"""
        print("\n📊 整体质量分析")
        print("=" * 80)
        
        # 统计分析
        total_nodes = len(restoration_results)
        valid_nodes = sum(1 for r in restoration_results if r['is_valid'])
        excellent_nodes = sum(1 for r in restoration_results if r['quality_score']['is_excellent'])
        good_nodes = sum(1 for r in restoration_results if r['quality_score']['is_good'])
        acceptable_nodes = sum(1 for r in restoration_results if r['quality_score']['is_acceptable'])
        
        avg_score = sum(r['quality_score']['overall_score'] for r in restoration_results) / total_nodes
        
        print(f"📈 节点质量分布:")
        print(f"   总节点数: {total_nodes}")
        print(f"   有效节点: {valid_nodes} ({valid_nodes/total_nodes*100:.1f}%)")
        print(f"   优秀节点: {excellent_nodes} ({excellent_nodes/total_nodes*100:.1f}%) [9-10分]")
        print(f"   良好节点: {good_nodes} ({good_nodes/total_nodes*100:.1f}%) [7-9分]")
        print(f"   可接受节点: {acceptable_nodes} ({acceptable_nodes/total_nodes*100:.1f}%) [5-7分]")
        print(f"   平均质量得分: {avg_score:.2f}/10")
        
        # 内容完整性检查
        restored_full_text = ''.join(r['restored_content'] for r in restoration_results if r['is_valid'])
        
        print(f"\n🔍 内容完整性验证:")
        print(f"   原文长度: {len(original_content)} 字符")
        print(f"   还原长度: {len(restored_full_text)} 字符")
        
        if restored_full_text == original_content:
            print(f"   ✅ 内容完全一致")
            completeness_score = 10.0
        else:
            # 计算相似度
            common_chars = sum(1 for i, char in enumerate(original_content) 
                             if i < len(restored_full_text) and char == restored_full_text[i])
            similarity = common_chars / len(original_content) * 100
            print(f"   ⚠️  内容有差异")
            print(f"   相似度: {similarity:.1f}%")
            completeness_score = similarity / 10
        
        # 系统效率分析
        stats = encoder.get_system_summary()
        
        print(f"\n⚙️  系统效率:")
        print(f"   层级数: {stats['active_levels']}")
        print(f"   压缩比: {stats['total_nodes']}/{total_nodes} = {stats['total_nodes']/total_nodes:.2f}")
        
        return {
            'total_nodes': total_nodes,
            'valid_nodes': valid_nodes,
            'quality_distribution': {
                'excellent': excellent_nodes,
                'good': good_nodes,
                'acceptable': acceptable_nodes
            },
            'average_quality_score': avg_score,
            'completeness_score': completeness_score,
            'content_similarity': similarity if 'similarity' in locals() else 100.0,
            'system_efficiency': {
                'levels': stats['active_levels'],
                'compression_ratio': stats['total_nodes']/total_nodes
            }
        }
    
    def _generate_evaluation_report(self, 
                                  corpus_name: str,
                                  original_content: str,
                                  restoration_results: List[Dict[str, Any]],
                                  overall_analysis: Dict[str, Any],
                                  beta: int,
                                  k: int) -> Dict[str, Any]:
        """生成完整的评价报告"""
        print(f"\n📋 生成评价报告")
        print("=" * 80)
        
        # 创建报告
        report = {
            'metadata': {
                'corpus_name': corpus_name,
                'evaluation_time': datetime.now().isoformat(),
                'parameters': {'beta': beta, 'k': k},
                'original_content_length': len(original_content)
            },
            'original_content': original_content,
            'restoration_results': restoration_results,
            'overall_analysis': overall_analysis,
            'summary': self._create_summary(overall_analysis),
            'recommendations': self._create_recommendations(overall_analysis, restoration_results)
        }
        
        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = Path(__file__).parent / f"evaluation_report_{corpus_name}_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"📁 评价报告已保存: {report_file}")
        
        # 显示总结
        summary = report['summary']
        print(f"\n🎯 评价总结:")
        print(f"   整体评级: {summary['overall_grade']}")
        print(f"   质量得分: {summary['quality_percentage']:.1f}%")
        print(f"   内容完整性: {summary['completeness_percentage']:.1f}%")
        print(f"   推荐状态: {summary['recommendation']}")
        
        return report
    
    def _create_summary(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """创建评价总结"""
        avg_score = analysis['average_quality_score']
        completeness = analysis['completeness_score']
        
        # 计算整体评级
        if avg_score >= 9.0 and completeness >= 9.5:
            grade = "A+ (优秀)"
        elif avg_score >= 8.0 and completeness >= 9.0:
            grade = "A (良好)"
        elif avg_score >= 7.0 and completeness >= 8.0:
            grade = "B (一般)"
        elif avg_score >= 5.0 and completeness >= 7.0:
            grade = "C (及格)"
        else:
            grade = "D (不及格)"
        
        # 推荐状态
        if avg_score >= 8.5 and completeness >= 9.0:
            recommendation = "推荐使用"
        elif avg_score >= 7.0 and completeness >= 8.0:
            recommendation = "可以使用，建议优化"
        else:
            recommendation = "需要改进后使用"
        
        return {
            'overall_grade': grade,
            'quality_percentage': avg_score * 10,
            'completeness_percentage': completeness * 10,
            'recommendation': recommendation
        }
    
    def _create_recommendations(self, 
                               analysis: Dict[str, Any], 
                               restoration_results: List[Dict[str, Any]]) -> List[str]:
        """创建改进建议"""
        recommendations = []
        
        if analysis['average_quality_score'] < 8.0:
            recommendations.append("提高内容质量：优化抽象算法或调整参数")
        
        if analysis['completeness_score'] < 9.0:
            recommendations.append("改善内容完整性：检查分块算法的字符边界处理")
        
        invalid_count = sum(1 for r in restoration_results if not r['is_valid'])
        if invalid_count > 0:
            recommendations.append(f"修复{invalid_count}个无效节点的还原错误")
        
        if analysis['quality_distribution']['excellent'] / analysis['total_nodes'] < 0.5:
            recommendations.append("增加优秀质量节点比例")
        
        return recommendations


def run_fermat_evaluation():
    """运行费马大定理语料的全面评价"""
    # 费马大定理完整语料
    fermat_texts = [
        "费马大定理由皮埃尔·德·费马在1637年提出。该定理表述为：对于大于2的正整数n，不存在正整数x、y、z使得x^n + y^n = z^n成立。",
        
        "这个看似简单的数学陈述困扰了数学家们358年。费马在阅读丢番图的《算术》时写下了这个猜想，声称有一个美妙的证明，但书页边缘太窄写不下。",
        
        "欧拉在1770年证明了n=3的情况，使用了无穷递降法。证明思路是假设存在最小的正整数解，然后构造更小的解导致矛盾。",
        
        "费马本人证明了n=4的情况。这实际上是寻找直角三角形，其中两条直角边都是完全平方数，通过数论分析证明不存在。",
        
        "恩斯特·库默尔发展了理想数理论，研究分圆域中的唯一分解问题。他证明了对于正则素数p，费马大定理成立。",
        
        "谷山丰和志村五郎提出了椭圆曲线和模形式之间联系的深刻猜想。这建立了椭圆曲线理论和模形式理论之间的桥梁。"
    ]
    
    evaluator = ComprehensiveEvaluator()
    return evaluator.evaluate_corpus(fermat_texts, beta=4, k=2, corpus_name="费马大定理")


if __name__ == "__main__":
    print("🔍 FACE编码结果全面评价系统")
    print("=" * 80)
    print("设计目标：")
    print("1. 显示完整原文")
    print("2. 逐个显示编码地址还原的文字") 
    print("3. 对每个还原结果进行质量评价")
    print("4. 提供便于大语料评价的机制")
    print("=" * 80)
    
    # 运行费马大定理语料评价
    evaluation_report = run_fermat_evaluation()
    
    print(f"\n🎉 全面评价完成!")