#!/usr/bin/env python3
"""
分形压缩器CLI Demo
支持多种文本类型的压缩演示，包括交互式和批量处理模式
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fractal_compressor import FractalCompressor


# 预设的测试文本
DEMO_TEXTS = {
    "novel": {
        "title": "科幻小说片段",
        "text": """在遥远的未来，2157年的地球已经不再是人类唯一的家园。太空殖民地如星星般散布在银河系的各个角落，而人工智能已经发展到了前所未有的高度。

艾莉娅·陈是新东京太空站的首席工程师，她正在为即将到来的深空探索任务做最后的准备工作。这次任务的目标是探索距离地球3000光年的开普勒-442b星球，那里据说有着适宜人类居住的环境。

"量子跃迁引擎的能量输出稳定在98.7%，"艾莉娅对着全息显示屏汇报道，"预计12小时后可以启动跃迁程序。"

指挥官马库斯·约翰逊点了点头，他的脸上写满了担忧。这次任务不仅仅是一次简单的星际探索，更关系到人类文明的未来。地球的资源正在枯竭，人口过剩问题日益严重，找到新的宜居星球已经成为了人类生存的关键。

当艾莉娅检查生命维持系统时，她注意到了一个异常的能量波动。这个波动虽然微弱，但却具有某种规律性，仿佛是某种智能生命体发出的信号。她立即将这个发现报告给了约翰逊指挥官。经过详细的分析，他们发现这个信号确实来自开普勒-442b星球的方向。

这个发现改变了一切——他们即将面对的可能不是一个无人的星球，而是一个已经有智能生命存在的世界。艾莉娅望着星空，心中既兴奋又忐忑。人类历史上第一次与外星智能生命的接触即将开始，而她，将成为这历史性时刻的见证者和参与者。"""
    },
    
    "tech": {
        "title": "技术文档",
        "text": """分布式系统架构设计是现代软件工程中的核心挑战之一。微服务架构通过将大型应用拆分为多个独立的小型服务，提供了更好的可扩展性、可维护性和部署灵活性。

在微服务架构中，服务间通信是一个关键考虑点。常用的通信模式包括同步的RESTful API调用和异步的消息队列机制。对于实时性要求高的场景，WebSocket或gRPC协议提供了更高效的解决方案。

容器化技术如Docker和Kubernetes已经成为微服务部署的标准选择。Docker提供了轻量级的应用打包和分发机制，而Kubernetes则负责容器的编排、自动扩缩容和服务发现。通过声明式配置，开发团队可以轻松管理复杂的多服务应用。

在数据管理方面，微服务通常采用数据库分离原则，每个服务维护自己的数据存储。这种模式避免了服务间的紧耦合，但也带来了分布式事务和数据一致性的挑战。常见的解决方案包括事件溯源、CQRS模式和最终一致性策略。

监控和日志管理在分布式环境中尤为重要。分布式追踪系统如Jaeger或Zipkin可以跟踪请求在多个服务间的流转路径，帮助快速定位性能瓶颈和错误源头。集中化的日志聚合平台如ELK Stack提供了统一的日志查询和分析能力。

安全性考虑包括服务间的身份验证和授权、网络隔离、敏感数据加密等多个层面。API网关通常作为系统的统一入口，负责请求路由、认证、限流和协议转换等功能。"""
    },
    
    "news": {
        "title": "科技新闻报道",
        "text": """据路透社最新报道，全球领先的人工智能研究机构OpenAI今日宣布推出其最新的多模态大语言模型GPT-5，该模型在文本理解、图像识别和代码生成方面实现了显著突破。

根据官方发布的技术报告，GPT-5采用了全新的Transformer架构优化，参数规模达到10万亿，比上一代模型提升了近50倍。在标准基准测试中，新模型在阅读理解、数学推理和创意写作等任务上的表现超越了人类专家水平。

特别值得关注的是，GPT-5首次实现了真正的多模态融合理解。该模型不仅能够处理文本和图像，还能理解音频、视频和3D模型数据。在演示中，模型成功完成了复杂的视觉推理任务，包括从建筑图纸生成3D模型、分析医学影像并提供诊断建议等。

行业专家认为，这一突破将对教育、医疗、科研和创意产业产生深远影响。斯坦福大学人工智能实验室主任李飞飞教授表示："GPT-5代表了人工智能发展的新里程碑，其强大的推理能力和创造性将重新定义人机协作的边界。"

然而，这项技术的发布也引发了关于AI安全和伦理的新一轮讨论。多位知名学者呼吁建立更完善的AI治理框架，确保先进AI技术的负责任发展和使用。

据悉，GPT-5将首先向研究机构和企业用户开放API接口，面向普通消费者的服务预计将在今年第四季度推出。OpenAI表示将继续投入大量资源用于AI安全研究，致力于构建有益、无害且诚实的人工智能系统。"""
    },
    
    "academic": {
        "title": "学术论文摘要", 
        "text": """Abstract: 本研究提出了一种基于深度强化学习的自适应网络路由优化算法，旨在解决现有路由协议在动态网络环境中的性能局限性。通过结合深度Q网络(DQN)和多智能体强化学习框架，我们设计了一个能够实时适应网络拓扑变化的智能路由系统。

Introduction: 随着物联网和移动计算的快速发展，网络环境变得越来越复杂和动态。传统的路由协议如OSPF和BGP主要基于静态或半静态的路径计算，难以有效应对网络拓扑的频繁变化、流量模式的动态性以及节点移动性等挑战。近年来，机器学习特别是强化学习在网络优化领域展现出巨大潜力。

Methodology: 我们提出的MADRL-Routing算法采用多智能体深度强化学习架构，其中每个网络节点作为一个独立的智能体，通过与环境交互学习最优的路由决策策略。算法的核心创新包括：（1）设计了综合考虑延迟、丢包率、能耗和负载均衡的多目标奖励函数；（2）引入了基于图神经网络的网络状态表示学习机制；（3）实现了分布式训练框架以提高算法的可扩展性。

Results: 在多种网络场景下的仿真实验表明，相比传统路由协议，本算法在平均端到端延迟、网络吞吐量和能效方面分别提升了23%、31%和18%。特别是在高动态性网络环境中，算法展现出了优异的适应性和鲁棒性。

Conclusion: 本研究验证了深度强化学习在网络路由优化中的有效性，为构建智能化的下一代网络基础设施提供了重要的理论基础和技术支撑。未来工作将focus on算法在实际网络部署中的工程化实现和大规模验证。"""
    }
}

# 压缩预设配置
COMPRESSION_PRESETS = {
    "conservative": {
        "name": "保守压缩",
        "ratio": 0.7,
        "base_threshold": 200,
        "max_levels": 6,
        "description": "适合重要文档，保留更多细节"
    },
    "balanced": {
        "name": "平衡压缩", 
        "ratio": 0.618,
        "base_threshold": 100,
        "max_levels": 8,
        "description": "黄金分割比例，平衡压缩率和质量"
    },
    "aggressive": {
        "name": "激进压缩",
        "ratio": 0.4,
        "base_threshold": 50,
        "max_levels": 10,
        "description": "最大化压缩率，适合预览和索引"
    }
}


def print_section(title: str, char: str = "=", width: int = 70):
    """打印格式化的章节标题"""
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}")


def print_level_analysis(fractal_result: List[str], thresholds: List[int]) -> int:
    """分析并打印各层级详情，返回总压缩长度"""
    print_section("📊 分形层级分析", "-")
    
    total_compressed = 0
    active_levels = 0
    
    for i, level_text in enumerate(fractal_result):
        if level_text.strip():
            level_len = len(level_text)
            threshold = thresholds[i] if i < len(thresholds) else "N/A"
            status = "✅ 未超限" if isinstance(threshold, int) and level_len <= threshold else "⚠️  超限"
            
            print(f"Level {i}: {level_len:>4} 字符 | 门限: {threshold:>4} | {status}")
            preview = level_text.replace('\n', '\\n')[:60]
            print(f"   预览: {preview}{'...' if len(level_text) > 60 else ''}")
            total_compressed += level_len
            active_levels += 1
        else:
            threshold = thresholds[i] if i < len(thresholds) else "N/A"
            print(f"Level {i}: {0:>4} 字符 | 门限: {threshold:>4} | 💤 空")
    
    print(f"\n📈 活跃层级: {active_levels}/{len(fractal_result)}")
    return total_compressed


def format_stats(original_len: int, compressed_len: int, process_time: float) -> Dict[str, Any]:
    """格式化压缩统计信息"""
    ratio = compressed_len / original_len if original_len > 0 else 0
    efficiency = (1 - ratio) * 100
    
    return {
        "original_length": original_len,
        "compressed_length": compressed_len,
        "compression_ratio": ratio,
        "compression_efficiency": efficiency,
        "processing_time": process_time,
        "space_saved": original_len - compressed_len,
        "speed_chars_per_sec": original_len / process_time if process_time > 0 else 0
    }


def compress_text(text: str, text_type: str, preset: str = "balanced") -> Dict[str, Any]:
    """执行文本压缩"""
    
    # 获取压缩配置
    config = COMPRESSION_PRESETS[preset].copy()
    
    # LLM配置
    llm_config = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "temperature": 0.1,
        "max_tokens": 2000,
        "language": "chinese"
    }
    
    # 创建压缩器
    compressor = FractalCompressor(
        ratio=config["ratio"],
        base_threshold=config["base_threshold"],
        max_levels=config["max_levels"],
        llm_config=llm_config
    )
    
    print_section(f"🚀 开始压缩: {text_type}")
    print(f"📊 压缩策略: {config['name']} ({preset})")
    print(f"🔧 配置: 比例={config['ratio']}, 门限={config['base_threshold']}, 层级={config['max_levels']}")
    print(f"📏 原文长度: {len(text)} 字符")
    
    # 显示层级门限
    thresholds = [compressor.get_threshold(i) for i in range(compressor.max_levels)]
    print(f"🎯 层级门限: {thresholds}")
    
    print("\n⏳ 压缩处理中...")
    
    start_time = time.time()
    try:
        fractal_result = compressor.compress_single_text(text)
        process_time = time.time() - start_time
        
        print(f"✅ 压缩完成！耗时: {process_time:.2f} 秒")
        
        # 分析结果
        total_compressed = print_level_analysis(fractal_result, thresholds)
        
        # 统计信息
        stats = format_stats(len(text), total_compressed, process_time)
        
        print_section("📈 压缩效果统计")
        print(f"📏 原文长度:     {stats['original_length']:>6} 字符")
        print(f"📦 压缩后长度:   {stats['compressed_length']:>6} 字符")
        print(f"💾 节省空间:     {stats['space_saved']:>6} 字符")
        print(f"📊 压缩比:       {stats['compression_ratio']:>6.1%}")
        print(f"⚡ 压缩效率:     {stats['compression_efficiency']:>6.1f}%")
        print(f"⏱️  处理时间:     {stats['processing_time']:>6.2f} 秒")
        print(f"🏃 处理速度:     {stats['speed_chars_per_sec']:>6.0f} 字符/秒")
        
        return {
            "success": True,
            "config": config,
            "original_text": text,
            "compressed_levels": fractal_result,
            "thresholds": thresholds,
            "statistics": stats,
            "metadata": {
                "text_type": text_type,
                "preset": preset,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
    except Exception as e:
        print(f"❌ 压缩失败: {str(e)}")
        return {"success": False, "error": str(e)}


def interactive_mode():
    """交互式模式"""
    print_section("🎮 交互式压缩Demo", "=")
    
    while True:
        print("\n📚 可用的测试文本:")
        for key, info in DEMO_TEXTS.items():
            length = len(info["text"])
            print(f"  {key:>8}: {info['title']} ({length} 字符)")
        
        print("\n🔧 可用的压缩预设:")
        for key, config in COMPRESSION_PRESETS.items():
            print(f"  {key:>11}: {config['name']} - {config['description']}")
        
        print("\n🎯 选择选项:")
        print("  输入格式: <文本类型> [压缩预设] (例如: novel balanced)")
        print("  特殊命令: quit (退出), help (帮助)")
        
        user_input = input("\n👉 请输入: ").strip().split()
        
        if not user_input:
            continue
            
        cmd = user_input[0].lower()
        
        if cmd == "quit":
            print("👋 再见！")
            break
        elif cmd == "help":
            print_section("📖 帮助信息", "-")
            print("使用方法:")
            print("1. 选择文本类型: novel, tech, news, academic")
            print("2. 选择压缩预设 (可选): conservative, balanced, aggressive")
            print("3. 默认使用 balanced 预设")
            continue
        
        if cmd not in DEMO_TEXTS:
            print("❌ 无效的文本类型，请重新选择")
            continue
            
        preset = user_input[1] if len(user_input) > 1 else "balanced"
        if preset not in COMPRESSION_PRESETS:
            print("❌ 无效的压缩预设，使用默认的 balanced")
            preset = "balanced"
        
        # 执行压缩
        text_info = DEMO_TEXTS[cmd]
        result = compress_text(text_info["text"], text_info["title"], preset)
        
        if result["success"]:
            # 保存结果
            output_file = f"compression_{cmd}_{preset}_{int(time.time())}.json"
            output_path = Path("examples") / "results" / output_file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"\n💾 结果已保存: {output_path}")
        
        print("\n" + "="*50)


def batch_mode(text_types: List[str], preset: str = "balanced"):
    """批量处理模式"""
    print_section(f"🔄 批量压缩Demo - {preset.upper()}", "=")
    
    results = {}
    
    for text_type in text_types:
        if text_type not in DEMO_TEXTS:
            print(f"❌ 跳过无效的文本类型: {text_type}")
            continue
            
        text_info = DEMO_TEXTS[text_type]
        result = compress_text(text_info["text"], text_info["title"], preset)
        
        if result["success"]:
            results[text_type] = result
            
            # 保存单个结果
            output_file = f"compression_{text_type}_{preset}.json"
            output_path = Path("examples") / "results" / output_file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"💾 已保存: {output_path}")
        else:
            print(f"❌ {text_type} 压缩失败")
    
    # 保存批量结果汇总
    if results:
        summary_file = f"batch_compression_{preset}_{int(time.time())}.json"
        summary_path = Path("examples") / "results" / summary_file
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print_section("📊 批量处理汇总")
        print(f"✅ 成功处理: {len(results)}/{len(text_types)} 个文本")
        print(f"💾 汇总报告: {summary_path}")
        
        # 显示对比统计
        print("\n📈 压缩效果对比:")
        for text_type, result in results.items():
            stats = result["statistics"]
            print(f"  {text_type:>8}: {stats['compression_ratio']:>6.1%} 压缩比, "
                  f"{stats['compression_efficiency']:>5.1f}% 效率")


def file_mode(input_file: str, preset: str = "balanced"):
    """文件处理模式"""
    print_section(f"📁 文件压缩Demo - {preset.upper()}", "=")
    
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"❌ 文件不存在: {input_file}")
        return
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f"📖 读取文件: {input_path}")
        print(f"📏 文件大小: {len(text)} 字符")
        
        result = compress_text(text, f"文件: {input_path.name}", preset)
        
        if result["success"]:
            # 保存结果
            output_file = f"compression_{input_path.stem}_{preset}.json"
            output_path = Path("examples") / "results" / output_file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"💾 结果已保存: {output_path}")
        
    except Exception as e:
        print(f"❌ 文件处理失败: {str(e)}")


def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(
        description="分形压缩器CLI Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s --interactive                    # 交互式模式
  %(prog)s --batch novel tech --preset aggressive  # 批量处理
  %(prog)s --file document.txt             # 文件处理
  %(prog)s --batch all                     # 处理所有预设文本
        """
    )
    
    parser.add_argument('--interactive', action='store_true',
                       help='启动交互式模式')
    
    parser.add_argument('--batch', nargs='+', metavar='TYPE',
                       help='批量处理模式，指定文本类型: novel, tech, news, academic, all')
    
    parser.add_argument('--file', metavar='PATH',
                       help='处理指定文件')
    
    parser.add_argument('--preset', choices=['conservative', 'balanced', 'aggressive'],
                       default='balanced', help='压缩预设 (默认: balanced)')
    
    parser.add_argument('--list-texts', action='store_true',
                       help='列出所有可用的测试文本')
    
    parser.add_argument('--list-presets', action='store_true',
                       help='列出所有压缩预设')
    
    args = parser.parse_args()
    
    # 确保结果目录存在
    results_dir = Path("examples") / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if args.list_texts:
        print_section("📚 可用测试文本")
        for key, info in DEMO_TEXTS.items():
            print(f"{key:>8}: {info['title']} ({len(info['text'])} 字符)")
    
    elif args.list_presets:
        print_section("🔧 压缩预设配置")
        for key, config in COMPRESSION_PRESETS.items():
            print(f"{key:>11}: {config['name']}")
            print(f"{'':>12}  {config['description']}")
            print(f"{'':>12}  比例: {config['ratio']}, 门限: {config['base_threshold']}, 层级: {config['max_levels']}")
    
    elif args.interactive:
        interactive_mode()
    
    elif args.batch:
        if 'all' in args.batch:
            text_types = list(DEMO_TEXTS.keys())
        else:
            text_types = args.batch
        batch_mode(text_types, args.preset)
    
    elif args.file:
        file_mode(args.file, args.preset)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()