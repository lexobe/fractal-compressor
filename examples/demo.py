#!/usr/bin/env python3
"""
分形压缩器Demo - 长文本压缩演示
专门展示小说文本的分形压缩效果
"""

import os
import sys
import time
import json
from typing import Dict, Any, List

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import logging
from fractal_compressor import FractalCompressor

# 全局保守配置 - 只记录重要信息
logging.basicConfig(
    level=logging.WARN,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler('fractal_compression_debug.log', encoding='utf-8')  # 输出到文件
    ]
)

# 只开启我们自己模块的详细日志
logging.getLogger('fractal_compressor').setLevel(logging.DEBUG)
logging.getLogger('fractal_compressor.llm_compressor').setLevel(logging.DEBUG)


def print_section(title: str, char: str = "="):
    """打印章节标题"""
    print(f"\n{char * 60}")
    print(f" {title}")
    print(f"{char * 60}")


def print_level_analysis(fractal_result: List[str], thresholds: List[int]):
    """分析并打印各层级详情"""
    print_section("📊 分形层级分析", "-")
    
    total_compressed = 0
    for i, level_text in enumerate(fractal_result):
        if level_text.strip():
            level_len = len(level_text)
            threshold = thresholds[i] if i < len(thresholds) else "N/A"
            status = "✅ 未超限" if isinstance(threshold, int) and level_len <= threshold else "⚠️  超限"
            
            print(f"Level {i}: {level_len:>4} 字符 | 门限: {threshold:>4} | {status}")
            print(f"   内容预览: {level_text[:50]}{'...' if len(level_text) > 50 else ''}")
            total_compressed += level_len
        else:
            print(f"Level {i}: {0:>4} 字符 | 门限: {thresholds[i] if i < len(thresholds) else 'N/A':>4} | 空")
    
    return total_compressed


def format_compression_stats(original_length: int, compressed_length: int, 
                           processing_time: float) -> Dict[str, Any]:
    """格式化压缩统计信息"""
    compression_ratio = compressed_length / original_length if original_length > 0 else 0
    compression_efficiency = (1 - compression_ratio) * 100
    
    return {
        "original_length": original_length,
        "compressed_length": compressed_length,
        "compression_ratio": compression_ratio,
        "compression_efficiency": compression_efficiency,
        "processing_time": processing_time,
        "space_saved": original_length - compressed_length
    }


def demo_novel_compression():
    """演示小说文本压缩"""
    
    print_section("🌀 分形压缩器 - 小说文本Demo")
    
    # 准备小说测试文本（经典科幻小说风格）
    novel_text = """
在遥远的未来，2157年的地球已经不再是人类唯一的家园。太空殖民地如星星般散布在银河系的各个角落，而人工智能已经发展到了前所未有的高度。

艾莉娅·陈是新东京太空站的首席工程师，她正在为即将到来的深空探索任务做最后的准备工作。这次任务的目标是探索距离地球3000光年的开普勒-442b星球，那里据说有着适宜人类居住的环境。

"量子跃迁引擎的能量输出稳定在98.7%，"艾莉娅对着全息显示屏汇报道，"预计12小时后可以启动跃迁程序。"

指挥官马库斯·约翰逊点了点头，他的脸上写满了担忧。这次任务不仅仅是一次简单的星际探索，更关系到人类文明的未来。地球的资源正在枯竭，人口过剩问题日益严重，找到新的宜居星球已经成为了人类生存的关键。

"艾莉娅，"约翰逊指挥官说道，"我需要你再次检查一遍生命维持系统。这次旅程将持续15个月，我们不能承受任何失误。"

太空站外，无数颗星星在黑暗的宇宙中闪烁着微弱的光芒。在这浩瀚的宇宙面前，人类显得如此渺小，但艾莉娅知道，正是这种渺小让人类变得更加坚强和团结。

当艾莉娅检查生命维持系统时，她注意到了一个异常的能量波动。这个波动虽然微弱，但却具有某种规律性，仿佛是某种智能生命体发出的信号。

"这不可能，"艾莉娅自言自语道，"在这个星系中，除了我们，不应该还有其他的智能生命。"

她立即将这个发现报告给了约翰逊指挥官。经过详细的分析，他们发现这个信号确实来自开普勒-442b星球的方向。这个发现改变了一切——他们即将面对的可能不是一个无人的星球，而是一个已经有智能生命存在的世界。

"我们必须重新评估这次任务的风险，"约翰逊指挥官严肃地说道，"与未知的外星文明接触，这超出了我们原本的计划范围。"

艾莉娅望着星空，心中既兴奋又忐忑。人类历史上第一次与外星智能生命的接触即将开始，而她，将成为这历史性时刻的见证者和参与者。

时间在紧张的准备中流逝，量子跃迁的时刻即将到来。整个太空站的人员都为这次历史性的任务做着最后的准备，因为他们知道，这次任务的结果将决定人类文明的未来走向。
"""
    
    # 显示原文信息
    print_section("📖 原文信息")
    print(f"📝 文本类型: 科幻小说片段")
    print(f"📏 原文长度: {len(novel_text)} 字符")
    print(f"📄 预览:\n{novel_text[:200]}...\n")
    
    # 创建分形压缩器（门限设置为100）
    print_section("⚙️  压缩器配置")
    
    llm_config = {
        "provider": "openai",
        "model": "gpt-4o-mini", 
        "api_key": os.getenv("OPENAI_API_KEY"),
        "temperature": 0.1,
        "max_tokens": 2000,
        "language": "chinese"
    }
    
    compressor = FractalCompressor(
        ratio=0.618,           # 黄金分割比例
        base_threshold=100,    # Level 0 门限设为100
        max_levels=8,          # 最多8层
        llm_config=llm_config
    )
    
    # 显示配置信息
    print(f"🔧 分割比例: {compressor.ratio}")
    print(f"🎯 基础门限: {compressor.base_threshold}")
    print(f"📚 最大层级: {compressor.max_levels}")
    print(f"🤖 LLM模型: {llm_config.get('model', 'default')}")
    
    # 显示各层级门限
    thresholds = [compressor.get_threshold(i) for i in range(compressor.max_levels)]
    print(f"\n📊 各层级门限: {thresholds}")
    
    # 开始压缩
    print_section("🚀 开始压缩处理", "=")
    print("⏳ 正在进行分形压缩...")
    
    start_time = time.time()
    
    try:
        fractal_result = compressor.compress_single_text(novel_text)
        processing_time = time.time() - start_time
        
        print(f"✅ 压缩完成！耗时: {processing_time:.2f} 秒")
        
        # 分析结果
        total_compressed = print_level_analysis(fractal_result, thresholds)
        
        # 压缩统计
        stats = format_compression_stats(len(novel_text), total_compressed, processing_time)
        
        print_section("📈 压缩效果统计")
        print(f"📏 原文长度:     {stats['original_length']:>6} 字符")
        print(f"📦 压缩后长度:   {stats['compressed_length']:>6} 字符") 
        print(f"💾 节省空间:     {stats['space_saved']:>6} 字符")
        print(f"📊 压缩比:       {stats['compression_ratio']:>6.1%}")
        print(f"⚡ 压缩效率:     {stats['compression_efficiency']:>6.1f}%")
        print(f"⏱️  处理时间:     {stats['processing_time']:>6.2f} 秒")
        
        # 详细内容展示
        print_section("📋 压缩结果详情")
        for i, level_text in enumerate(fractal_result):
            if level_text.strip():
                print(f"\n🔸 Level {i} ({len(level_text)} 字符):")
                print(f"   {level_text}")
        
        # 保存结果
        save_result = {
            "metadata": {
                "source": "科幻小说Demo",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config": {
                    "ratio": compressor.ratio,
                    "base_threshold": compressor.base_threshold,
                    "max_levels": compressor.max_levels
                }
            },
            "original": {
                "text": novel_text,
                "length": len(novel_text)
            },
            "compressed": {
                "levels": fractal_result,
                "level_lengths": [len(level) for level in fractal_result],
                "thresholds": thresholds
            },
            "statistics": stats
        }
        
        output_file = "novel_compression_demo.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_result, f, ensure_ascii=False, indent=2)
        
        print_section("💾 结果已保存")
        print(f"📁 输出文件: {output_file}")
        
        return save_result
        
    except Exception as e:
        print(f"❌ 压缩过程中出现错误: {str(e)}")
        return None


if __name__ == "__main__":
    demo_novel_compression()