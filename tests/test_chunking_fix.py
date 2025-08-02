#!/usr/bin/env python3
"""
测试分块修复效果

验证"下欧拉"问题是否已修复
"""

import sys
from pathlib import Path

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from fractal_compressor import FACEEncoder, FACEAddressing


def test_chunking_fix():
    """测试分块修复效果"""
    print("🔧 测试分块修复效果")
    print("=" * 60)
    
    # 使用之前有问题的文本
    problem_texts = [
        "费马大定理由皮埃尔·德·费马在1637年提出。该定理表述为：对于大于2的正整数n，不存在正整数x、y、z使得x^n + y^n = z^n成立。",
        
        "这个看似简单的数学陈述困扰了数学家们358年。费马在阅读丢番图的《算术》时写下了这个猜想，声称有一个美妙的证明，但书页边缘太窄写不下。",
        
        "欧拉在1770年证明了n=3的情况，使用了无穷递降法。证明思路是假设存在最小的正整数解，然后构造更小的解导致矛盾。",
        
        "费马本人证明了n=4的情况。这实际上是寻找直角三角形，其中两条直角边都是完全平方数，通过数论分析证明不存在。",
    ]
    
    print("📝 测试文本:")
    for i, text in enumerate(problem_texts, 1):
        print(f"{i}. {text}")
    
    # 创建编码器
    encoder = FACEEncoder(
        tau_buffer=80,
        tau_slice=60,
        beta=4,
        k=2,
        use_logical_clock=True
    )
    
    print(f"\n🔄 逐步处理...")
    
    # 逐步添加并观察
    for i, text in enumerate(problem_texts, 1):
        print(f"\n📝 处理第{i}段: {text[:30]}...")
        encoder.append_text(text + "\n\n")
        
        # 检查缓冲区状态
        buffer_len = len(encoder.buffer)
        print(f"   缓冲区: {buffer_len} 字符")
        
        # 检查节点数
        stats = encoder.get_system_summary()
        print(f"   节点数: {stats['total_nodes']}")
    
    # 处理剩余缓冲区
    print(f"\n🔄 处理剩余缓冲区...")
    encoder.process_buffer()
    
    # 检查最终结果
    print(f"\n📊 最终结果:")
    final_stats = encoder.get_system_summary()
    print(f"   总节点数: {final_stats['total_nodes']}")
    print(f"   活跃层级: {final_stats['active_levels']}")
    
    # 检查Level-0节点内容
    print(f"\n🔍 检查Level-0节点内容:")
    level0_nodes = encoder.get_level0_nodes()
    level0_nodes.sort(key=lambda n: n.t_now)
    
    problems_found = 0
    
    for i, node in enumerate(level0_nodes, 1):
        print(f"\n节点{i} ({node.id}):")
        print(f"   内容: \"{node.content}\"")
        
        # 检查是否有截断问题
        content = node.content.strip()
        if content.startswith(('下', '论', '式')):
            print(f"   ❌ 发现截断问题: 以'{content[0]}'开头")
            problems_found += 1
        elif '下欧拉' in content:
            print(f"   ❌ 发现'下欧拉'问题")
            problems_found += 1
        else:
            print(f"   ✅ 内容正常")
    
    # 验证Chunk拆分不变性
    print(f"\n🔍 验证Chunk拆分不变性:")
    
    # 重建原文
    all_content = ''.join(node.content for node in level0_nodes)
    original_content = ''.join(text + "\n\n" for text in problem_texts)
    
    if all_content == original_content:
        print(f"   ✅ Chunk拆分不变性验证通过")
        print(f"   原文长度: {len(original_content)} 字符")
        print(f"   重建长度: {len(all_content)} 字符")
    else:
        print(f"   ❌ Chunk拆分不变性验证失败")
        print(f"   原文长度: {len(original_content)} 字符")
        print(f"   重建长度: {len(all_content)} 字符")
        
        # 找出差异
        min_len = min(len(original_content), len(all_content))
        for i in range(min_len):
            if original_content[i] != all_content[i]:
                print(f"   第{i+1}个字符开始不同:")
                print(f"   原文: {repr(original_content[max(0,i-10):i+10])}")
                print(f"   重建: {repr(all_content[max(0,i-10):i+10])}")
                break
        problems_found += 1
    
    # 总结
    print(f"\n" + "="*60)
    print(f"🎯 修复效果总结:")
    
    if problems_found == 0:
        print(f"   ✅ 所有问题已修复!")
        print(f"   ✅ 没有发现内容截断")
        print(f"   ✅ Chunk拆分不变性验证通过")
        print(f"   ✅ '下欧拉'问题已解决")
    else:
        print(f"   ⚠️  发现{problems_found}个问题待解决")
    
    return problems_found == 0


if __name__ == "__main__":
    success = test_chunking_fix()
    sys.exit(0 if success else 1)