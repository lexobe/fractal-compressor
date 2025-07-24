#!/usr/bin/env python3
"""
结果分析脚本 - 分析所有语料的压缩结果
"""

import json
import os
from pathlib import Path

def analyze_all_results():
    """分析所有压缩结果"""
    
    results_dir = Path("results")
    if not results_dir.exists():
        print("❌ results目录不存在")
        return
    
    # 收集所有结果文件
    result_files = list(results_dir.glob("*_result.json"))
    if not result_files:
        print("❌ 未找到任何结果文件")
        return
    
    print(f"📊 找到 {len(result_files)} 个结果文件")
    print("="*70)
    
    # 分析每个文件
    results_data = []
    
    for result_file in sorted(result_files):
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if data.get('success', False):
                stats = data['statistics']
                corpus_name = data['source'].replace('.txt', '').replace('corpus_', '')
                
                results_data.append({
                    'name': corpus_name,
                    'filename': result_file.name,
                    'original_length': stats['original_length'],
                    'compressed_length': stats['compressed_length'],
                    'compression_efficiency': stats['compression_efficiency'],
                    'processing_time': stats['processing_time'],
                    'space_saved': stats['space_saved']
                })
                
        except Exception as e:
            print(f"❌ 读取文件失败: {result_file} - {str(e)}")
    
    if not results_data:
        print("❌ 没有成功的压缩结果")
        return
    
    # 按压缩效率排序
    results_data.sort(key=lambda x: x['compression_efficiency'], reverse=True)
    
    # 打印详细结果
    print(f"{'语料类型':<15} {'原长度':<8} {'压缩后':<8} {'效率':<8} {'耗时':<8} {'节省':<8}")
    print("-"*70)
    
    total_original = 0
    total_compressed = 0
    total_time = 0
    
    for result in results_data:
        print(f"{result['name']:<15} "
              f"{result['original_length']:<8} "
              f"{result['compressed_length']:<8} "
              f"{result['compression_efficiency']:<8.1f}% "
              f"{result['processing_time']:<8.2f}s "
              f"{result['space_saved']:<8}")
        
        total_original += result['original_length']
        total_compressed += result['compressed_length']
        total_time += result['processing_time']
    
    # 总体统计
    print("-"*70)
    overall_efficiency = (1 - total_compressed / total_original) * 100 if total_original > 0 else 0
    
    print(f"{'总计':<15} "
          f"{total_original:<8} "
          f"{total_compressed:<8} "
          f"{overall_efficiency:<8.1f}% "
          f"{total_time:<8.2f}s "
          f"{total_original - total_compressed:<8}")
    
    print(f"\n📈 压缩效果排名:")
    for i, result in enumerate(results_data, 1):
        print(f"  {i}. {result['name']}: {result['compression_efficiency']:.1f}%")
    
    # 语料类型分析
    print(f"\n📋 语料特性分析:")
    
    # 语料分类
    categories = {
        '数学论文': ['6_mathematics'],
        '技术文档': ['4_technical'], 
        '学术论文': ['2_academic'],
        '文学作品': ['3_novel', '5_literature'],
        '新闻文本': ['1_news']
    }
    
    for category, corpus_ids in categories.items():
        category_results = [r for r in results_data if any(corpus_id in r['name'] for corpus_id in corpus_ids)]
        if category_results:
            avg_efficiency = sum(r['compression_efficiency'] for r in category_results) / len(category_results)
            avg_length = sum(r['original_length'] for r in category_results) / len(category_results)
            print(f"  {category}: 平均效率 {avg_efficiency:.1f}%, 平均长度 {avg_length:.0f} 字符")
    
    # 数学语料的特殊分析
    math_result = next((r for r in results_data if '6_mathematics' in r['name']), None)
    if math_result:
        print(f"\n🔢 数学语料特殊分析:")
        print(f"  原始长度: {math_result['original_length']} 字符")
        print(f"  压缩效率: {math_result['compression_efficiency']:.1f}%")
        print(f"  相比平均效率: {math_result['compression_efficiency'] - overall_efficiency:.1f}% {'高于' if math_result['compression_efficiency'] > overall_efficiency else '低于'}平均")
        
        # 查看数学语料的层级分布
        math_file = results_dir / f"{math_result['filename']}"
        try:
            with open(math_file, 'r', encoding='utf-8') as f:
                math_data = json.load(f)
            
            print(f"  层级分布:")
            for i, level in enumerate(math_data['compressed']['levels'][:5]):  # 只显示前5层
                if level.strip():
                    print(f"    Level {i}: {len(level)} 字符 - {level[:50]}{'...' if len(level) > 50 else ''}")
        except:
            pass

if __name__ == "__main__":
    analyze_all_results()