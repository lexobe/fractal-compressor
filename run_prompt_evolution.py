#!/usr/bin/env python3
"""
LLM Prompt进化算法完整演示
运行文本分割prompt的自动优化实验
"""

import os
import sys
import json
import time
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

# 加载环境变量
from load_env import load_env_file
load_env_file()

from prompt_evolution import (
    PromptEvolutionEngine, 
    EvolutionConfig,
    TextSegmentationFitnessEvaluator,
    SimpleFitnessEvaluator
)
from prompt_evolution.datasets import DatasetLoader, DatasetAnalyzer


def setup_environment():
    """设置环境"""
    print("🔧 检查环境配置...")
    
    # 检查API密钥
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ 未找到OPENAI_API_KEY环境变量")
        print("请在 .env 文件中设置API密钥，或通过环境变量设置")
        
        # 尝试从.env文件加载
        env_file = Path(".env")
        if env_file.exists():
            print("📝 尝试从 .env 文件加载配置...")
            with open(env_file) as f:
                for line in f:
                    if line.startswith("OPENAI_API_KEY="):
                        key = line.split("=", 1)[1].strip()
                        os.environ["OPENAI_API_KEY"] = key
                        print("✅ 成功加载API密钥")
                        break
        
        if not os.getenv("OPENAI_API_KEY"):
            print("❌ 无法加载API密钥，将使用模拟模式")
            return False
    
    print("✅ 环境配置检查完成")
    return True


def load_test_data(quick_mode: bool = False):
    """加载测试数据"""
    print("📊 加载测试数据...")
    
    loader = DatasetLoader()
    
    # 获取数据集信息
    info = loader.get_dataset_info()
    print(f"可用数据集: {info['available_datasets']}")
    
    if quick_mode:
        # 快速模式：使用少量样本
        samples = loader.sample_for_quick_test(count=8)
        print(f"🚀 快速模式：加载 {len(samples)} 个测试样本")
    else:
        # 完整模式：尝试加载真实数据集
        samples = []
        
        # 尝试加载CPTS数据集
        try:
            cpts_samples = loader.load_samples("cpts", "test", limit=20)
            samples.extend(cpts_samples)
        except Exception as e:
            print(f"❌ 加载CPTS失败: {e}")
        
        # 如果没有真实数据，使用虚拟样本
        if not samples:
            samples = loader.sample_for_quick_test(count=15)
            print(f"📝 使用虚拟样本：{len(samples)} 个")
    
    # 分析样本
    analyzer = DatasetAnalyzer()
    analysis = analyzer.analyze_samples(samples)
    
    print(f"📈 数据集分析:")
    print(f"  样本数量: {analysis['count']}")
    print(f"  平均文本长度: {analysis['text_length']['avg']:.0f} 字符")
    print(f"  平均分割点数: {analysis['boundaries']['avg']:.1f}")
    print(f"  语言分布: {analysis['languages']}")
    
    return samples


def create_seed_prompts():
    """创建种子prompt"""
    return [
        # 基础分割prompt
        """请将以下文本按照主题进行分割。在每个主题转换的地方插入分割标记"[SPLIT]"。

文本: {text}

要求:
1. 保持主题内容的完整性
2. 确保逻辑转换清晰  
3. 避免过度分割
4. 分割点应该在自然的语义边界处""",

        # 专家角色prompt
        """作为文本分析专家，请识别以下文本中的语义边界，并在适当位置插入"[SPLIT]"标记。

分析文本: {text}

分割原则:
- 关注主题变化和逻辑转折
- 保持段落语义完整性
- 考虑上下文连贯性
- 确保每个分割段落都是完整的语义单元""",

        # 结构化prompt
        """## 文本分割任务

**输入**: {text}

**目标**: 识别文本中的自然分割点

**输出格式**: 在分割点插入"[SPLIT]"

**质量标准**:
1. 主题一致性: 同一段落内容相关
2. 转换自然性: 分割点符合逻辑
3. 信息完整性: 避免截断关键信息
4. 长度平衡性: 分割后段落长度相对均衡""",

        # 简洁高效prompt
        """分析文本主题边界，在转换点插入[SPLIT]：

{text}

标准：保持语义完整，转换自然，避免碎片化。""",

        # 详细指导prompt
        """请仔细分析以下文本的结构和内容，识别不同主题或概念之间的边界：

文本内容: {text}

分割指导:
- 寻找主题转换、时间变化、角度切换等关键转折点
- 确保每个分割后的段落都有明确的中心思想
- 保持原文的逻辑结构和完整性
- 在识别的边界处精确插入"[SPLIT]"标记
- 避免在句子中间或关键信息处分割"""
    ]


def run_experiment(quick_mode: bool = False, use_simple_evaluator: bool = False):
    """运行完整的进化实验"""
    
    print("🧬 LLM Prompt进化算法实验")
    print("=" * 60)
    
    # 设置环境
    has_api = setup_environment()
    
    # 加载测试数据
    test_samples = load_test_data(quick_mode)
    
    if not test_samples:
        print("❌ 无法加载测试数据")
        return None
    
    # 创建种子prompt
    seed_prompts = create_seed_prompts()
    print(f"🌱 准备种子prompt: {len(seed_prompts)} 个")
    
    # 配置进化参数
    if quick_mode:
        config = EvolutionConfig(
            population_size=8,
            max_generations=5,
            elite_ratio=0.25,
            mutation_rate=0.8,
            crossover_rate=0.4,
            stagnation_limit=3
        )
        print("🚀 快速模式配置")
    else:
        config = EvolutionConfig(
            population_size=15,
            max_generations=20,
            elite_ratio=0.2,
            mutation_rate=0.7,
            crossover_rate=0.5,
            stagnation_limit=5
        )
        print("🔬 完整模式配置")
    
    print(f"配置: {config.population_size} 个体 × {config.max_generations} 代")
    
    # 创建适应度评估器
    if use_simple_evaluator or not has_api:
        evaluator = SimpleFitnessEvaluator()
        print("📊 使用简化评估器")
    else:
        evaluator = TextSegmentationFitnessEvaluator(
            boundary_weight=0.45,
            semantic_weight=0.25,
            granularity_weight=0.20,
            efficiency_weight=0.05,
            penalty_weight=0.05
        )
        print("📊 使用完整评估器")
    
    # 创建进化引擎
    engine = PromptEvolutionEngine(
        fitness_evaluator=evaluator,
        config=config,
        model_name="gpt-4o-mini"
    )
    
    # 开始进化
    print("\n🚀 开始进化过程...")
    start_time = time.time()
    
    try:
        best_individual = engine.run_evolution(test_samples, seed_prompts)
        
        # 显示结果
        elapsed_time = time.time() - start_time
        summary = engine.get_evolution_summary()
        
        print("\n" + "=" * 60)
        print("🏆 进化实验完成！")
        print("=" * 60)
        
        print(f"📊 实验摘要:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        print(f"\n🎯 最优Prompt:")
        print("-" * 40)
        print(best_individual.prompt)
        print("-" * 40)
        
        print(f"\n📈 进化历程:")
        print(f"  适应度: {summary['初始适应度']:.4f} → {summary['最佳适应度']:.4f}")
        print(f"  提升幅度: {summary['适应度提升']:.4f}")
        print(f"  进化代数: {summary['总代数']}")
        print(f"  总用时: {elapsed_time:.1f} 秒")
        
        # 保存实验报告
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = f"prompt_evolution_report_{timestamp}.json"
        engine.save_experiment_report(report_file)
        
        # 保存最佳prompt
        best_prompt_file = f"best_prompt_{timestamp}.txt"
        with open(best_prompt_file, 'w', encoding='utf-8') as f:
            f.write(f"# 最优Prompt (适应度: {best_individual.fitness:.4f})\n\n")
            f.write(best_individual.prompt)
            f.write(f"\n\n# 进化信息\n")
            f.write(f"代数: {best_individual.generation}\n")
            f.write(f"Token数: {best_individual.token_count}\n")
            f.write(f"父代: {best_individual.parent_ids}\n")
            f.write(f"进化历史: {best_individual.evolution_history}\n")
        
        print(f"\n📄 结果已保存:")
        print(f"  实验报告: {report_file}")
        print(f"  最佳prompt: {best_prompt_file}")
        
        return best_individual, engine
        
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断实验")
        return None, engine
    except Exception as e:
        print(f"\n❌ 实验过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None, engine


def interactive_demo():
    """交互式演示"""
    print("🎮 LLM Prompt进化算法交互式演示")
    print("=" * 50)
    
    while True:
        print("\n选择运行模式:")
        print("1. 快速演示 (5分钟)")
        print("2. 完整实验 (15-30分钟)")
        print("3. 简化测试 (无需API)")
        print("4. 查看数据集信息")
        print("5. 退出")
        
        choice = input("\n请选择 [1-5]: ").strip()
        
        if choice == "1":
            print("\n🚀 启动快速演示模式...")
            result = run_experiment(quick_mode=True, use_simple_evaluator=False)
            if result[0]:
                print("✅ 快速演示完成")
        
        elif choice == "2":
            print("\n🔬 启动完整实验模式...")
            result = run_experiment(quick_mode=False, use_simple_evaluator=False)
            if result[0]:
                print("✅ 完整实验完成")
        
        elif choice == "3":
            print("\n🧪 启动简化测试模式...")
            result = run_experiment(quick_mode=True, use_simple_evaluator=True)
            if result[0]:
                print("✅ 简化测试完成")
        
        elif choice == "4":
            print("\n📊 数据集信息:")
            loader = DatasetLoader()
            info = loader.get_dataset_info()
            print(json.dumps(info, indent=2, ensure_ascii=False))
        
        elif choice == "5":
            print("👋 再见！")
            break
        
        else:
            print("❌ 无效选择，请重新输入")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "quick":
            print("🚀 快速模式")
            run_experiment(quick_mode=True)
        elif mode == "full":
            print("🔬 完整模式")
            run_experiment(quick_mode=False)
        elif mode == "simple":
            print("🧪 简化模式")
            run_experiment(quick_mode=True, use_simple_evaluator=True)
        else:
            print(f"❌ 未知模式: {mode}")
            print("支持的模式: quick, full, simple")
    else:
        # 交互式模式
        interactive_demo()