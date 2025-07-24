#!/usr/bin/env python3
"""
分形压缩器 - 独立压缩程序

功能：
- 文件输入输出的分形文本压缩
- 支持单文件或批量压缩
- 生成详细的压缩报告
- 命令行参数配置

使用方法:
python fractal_compress.py input.txt -o output.json
python fractal_compress.py corpus_*.txt -o results/ --batch
python fractal_compress.py input.txt --ratio 0.7 --threshold 200
"""

import os
import sys
import json
import time 
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import logging

# 全局保守配置 - 只记录重要信息
logging.basicConfig(
    level=logging.WARN,
    format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler('fractal_compression_debug.log', encoding='utf-8')  # 输出到文件
    ]
)

# 只开启我们自己模块的详细日志
logging.getLogger('fractal_compressor').setLevel(logging.DEBUG)
logging.getLogger('fractal_compressor.llm_compressor').setLevel(logging.DEBUG)

from fractal_compressor import FractalCompressor

class FractalCompressApp:
    """分形压缩应用程序"""
    
    def __init__(self, 
                 ratio: float = 0.618,
                 base_threshold: int = 100, 
                 max_levels: int = 8,
                 llm_config: Optional[Dict[str, Any]] = None,
                 verbose: bool = False):
        """
        初始化压缩应用
        
        Args:
            ratio: 分形比例 (默认0.618黄金比例)
            base_threshold: Level 0基础容量 (默认100字符)
            max_levels: 最大层级数 (默认8层)
            llm_config: LLM配置字典
            verbose: 是否显示详细日志
        """
        self.ratio = ratio
        self.base_threshold = base_threshold
        self.max_levels = max_levels
        self.verbose = verbose
        
        # 配置日志
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
        
        # 默认LLM配置
        if llm_config is None:
            llm_config = {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "temperature": 0.1,
                "max_tokens": 2000,
                "language": "chinese"
            }
        
        # 创建压缩器
        self.compressor = FractalCompressor(
            ratio=ratio,
            base_threshold=base_threshold,
            max_levels=max_levels,
            llm_config=llm_config
        )
        
        self.logger.info(f"初始化分形压缩器: ratio={ratio}, threshold={base_threshold}, levels={max_levels}")
    
    def read_input_file(self, file_path: str) -> str:
        """读取输入文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.logger.info(f"读取文件: {file_path} ({len(content)} 字符)")
            return content
        except Exception as e:
            self.logger.error(f"读取文件失败: {file_path} - {str(e)}")
            raise
    
    def compress_text(self, text: str, source_name: str = "unknown") -> Dict[str, Any]:
        """压缩单个文本"""
        self.logger.info(f"开始压缩: {source_name}")
        start_time = time.time()
        
        try:
            # 执行压缩
            fractal_result = self.compressor.compress_single_text(text)
            processing_time = time.time() - start_time
            
            # 计算统计信息
            compressed_length = sum(len(level) for level in fractal_result if level.strip())
            compression_ratio = compressed_length / len(text) if len(text) > 0 else 0
            compression_efficiency = (1 - compression_ratio) * 100
            
            # 层级分析
            thresholds = [self.compressor.get_threshold(i) for i in range(len(fractal_result))]
            level_analysis = []
            
            for i, level_text in enumerate(fractal_result):
                if i < len(thresholds):
                    level_analysis.append({
                        "level": i,
                        "length": len(level_text),
                        "threshold": thresholds[i],
                        "usage_ratio": len(level_text) / thresholds[i] if thresholds[i] > 0 else 0,
                        "is_overflow": len(level_text) > thresholds[i],
                        "content": level_text
                    })
            
            result = {
                "source": source_name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "success": True,
                "original": {
                    "text": text,
                    "length": len(text)
                },
                "compressed": {
                    "levels": fractal_result,
                    "level_lengths": [len(level) for level in fractal_result],
                    "thresholds": thresholds,
                    "level_analysis": level_analysis
                },
                "statistics": {
                    "original_length": len(text),
                    "compressed_length": compressed_length,
                    "compression_ratio": compression_ratio,
                    "compression_efficiency": compression_efficiency,
                    "processing_time": processing_time,
                    "space_saved": len(text) - compressed_length
                },
                "config": {
                    "ratio": self.ratio,
                    "base_threshold": self.base_threshold,
                    "max_levels": self.max_levels
                }
            }
            
            self.logger.info(f"压缩完成: {source_name} - 效率 {compression_efficiency:.1f}%, 耗时 {processing_time:.2f}秒")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"压缩失败: {source_name} - {str(e)}")
            return {
                "source": source_name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "success": False,
                "error": str(e),
                "processing_time": processing_time
            }
    
    def save_result(self, result: Dict[str, Any], output_path: str):
        """保存压缩结果"""
        try:
            # 确保输出目录存在
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"结果已保存: {output_path}")
            
        except Exception as e:
            self.logger.error(f"保存结果失败: {output_path} - {str(e)}")
            raise
    
    def compress_single_file(self, input_file: str, output_file: str):
        """压缩单个文件"""
        # 读取输入
        text = self.read_input_file(input_file)
        
        # 压缩
        result = self.compress_text(text, source_name=os.path.basename(input_file))
        
        # 保存结果
        self.save_result(result, output_file)
        
        # 打印摘要
        if result["success"]:
            stats = result["statistics"]
            print(f"\n✅ 压缩完成!")
            print(f"📁 输入文件: {input_file}")
            print(f"📄 输出文件: {output_file}")
            print(f"📏 原文长度: {stats['original_length']:,} 字符")
            print(f"📦 压缩后长度: {stats['compressed_length']:,} 字符")
            print(f"💾 节省空间: {stats['space_saved']:,} 字符")
            print(f"📊 压缩效率: {stats['compression_efficiency']:.1f}%")
            print(f"⏱️  处理时间: {stats['processing_time']:.2f} 秒")
        else:
            print(f"\n❌ 压缩失败: {result['error']}")
    
    def compress_batch_files(self, input_pattern: List[str], output_dir: str):
        """批量压缩文件"""
        results = {}
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 开始批量压缩，输出目录: {output_dir}")
        
        for input_file in input_pattern:
            if not os.path.exists(input_file):
                self.logger.warning(f"文件不存在: {input_file}")
                continue
                
            # 生成输出文件名
            input_name = Path(input_file).stem
            output_file = output_dir / f"{input_name}_compressed.json"
            
            # 读取和压缩
            try:
                text = self.read_input_file(input_file)
                result = self.compress_text(text, source_name=os.path.basename(input_file))
                
                # 保存单个结果
                self.save_result(result, output_file)
                results[input_file] = result
                
            except Exception as e:
                self.logger.error(f"处理文件失败: {input_file} - {str(e)}")
                results[input_file] = {
                    "success": False,
                    "error": str(e),
                    "source": os.path.basename(input_file)
                }
        
        # 生成批量汇总报告
        self.generate_batch_report(results, output_dir / "batch_summary.json")
        
        # 打印批量摘要
        self.print_batch_summary(results)
    
    def generate_batch_report(self, results: Dict[str, Any], output_file: str):
        """生成批量处理汇总报告"""
        successful_results = {k: v for k, v in results.items() if v.get("success", False)}
        
        if successful_results:
            total_original = sum(r["statistics"]["original_length"] for r in successful_results.values())
            total_compressed = sum(r["statistics"]["compressed_length"] for r in successful_results.values())
            total_time = sum(r["statistics"]["processing_time"] for r in successful_results.values())
            avg_efficiency = sum(r["statistics"]["compression_efficiency"] for r in successful_results.values()) / len(successful_results)
            
            summary = {
                "batch_info": {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_files": len(results),
                    "successful_files": len(successful_results),
                    "failed_files": len(results) - len(successful_results)
                },
                "overall_statistics": {
                    "total_original_chars": total_original,
                    "total_compressed_chars": total_compressed,
                    "total_space_saved": total_original - total_compressed,
                    "overall_compression_ratio": total_compressed / total_original if total_original > 0 else 0,
                    "average_compression_efficiency": avg_efficiency,
                    "total_processing_time": total_time
                },
                "individual_results": results,
                "config": {
                    "ratio": self.ratio,
                    "base_threshold": self.base_threshold, 
                    "max_levels": self.max_levels
                }
            }
        else:
            summary = {
                "batch_info": {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_files": len(results),
                    "successful_files": 0,
                    "failed_files": len(results)
                },
                "individual_results": results
            }
        
        self.save_result(summary, output_file)
    
    def print_batch_summary(self, results: Dict[str, Any]):
        """打印批量处理摘要"""
        successful_results = {k: v for k, v in results.items() if v.get("success", False)}
        
        print(f"\n📊 批量压缩完成!")
        print(f"📁 总文件数: {len(results)}")
        print(f"✅ 成功处理: {len(successful_results)}")
        print(f"❌ 失败数量: {len(results) - len(successful_results)}")
        
        if successful_results:
            total_original = sum(r["statistics"]["original_length"] for r in successful_results.values())
            total_compressed = sum(r["statistics"]["compressed_length"] for r in successful_results.values())
            avg_efficiency = sum(r["statistics"]["compression_efficiency"] for r in successful_results.values()) / len(successful_results)
            
            print(f"📏 总原文长度: {total_original:,} 字符")
            print(f"📦 总压缩长度: {total_compressed:,} 字符")
            print(f"💾 总节省空间: {total_original - total_compressed:,} 字符")
            print(f"⚡ 平均压缩效率: {avg_efficiency:.1f}%")
            
            # 显示各文件效率
            print(f"\n📋 各文件压缩效率:")
            for file_path, result in successful_results.items():
                filename = os.path.basename(file_path)
                efficiency = result["statistics"]["compression_efficiency"]
                print(f"  {filename}: {efficiency:.1f}%")

def create_argument_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="分形文本压缩器 - 独立压缩程序",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 压缩单个文件
  python fractal_compress.py input.txt -o output.json
  
  # 批量压缩
  python fractal_compress.py corpus_*.txt -o results/ --batch
  
  # 自定义配置
  python fractal_compress.py input.txt -o output.json --ratio 0.7 --threshold 200
  
  # 详细日志模式
  python fractal_compress.py input.txt -o output.json --verbose
        """
    )
    
    # 输入文件
    parser.add_argument('input', nargs='+', 
                       help='输入文件路径（支持通配符匹配）')
    
    # 输出选项
    parser.add_argument('-o', '--output', required=True,
                       help='输出文件路径（单文件）或输出目录（批量模式）')
    
    # 批量模式
    parser.add_argument('--batch', action='store_true',
                       help='批量处理模式')
    
    # 压缩参数
    parser.add_argument('--ratio', type=float, default=0.618,
                       help='分形比例 (默认: 0.618)')
    
    parser.add_argument('--threshold', type=int, default=100,
                       help='Level 0基础容量 (默认: 100)')
    
    parser.add_argument('--levels', type=int, default=8,
                       help='最大层级数 (默认: 8)')
    
    # LLM配置
    parser.add_argument('--model', default="gpt-4o-mini",
                       help='LLM模型名称 (默认: gpt-4o-mini)')
    
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='LLM温度参数 (默认: 0.1)')
    
    # 其他选项
    parser.add_argument('--verbose', action='store_true',
                       help='显示详细日志')
    
    return parser

def main():
    """主函数"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # API密钥将在实际使用时检查
    
    # LLM配置
    llm_config = {
        "provider": "openai",
        "model": args.model,
        "api_key": os.getenv("OPENAI_API_KEY"),
        "temperature": args.temperature,
        "max_tokens": 2000,
        "language": "chinese"
    }
    
    # 创建压缩应用
    app = FractalCompressApp(
        ratio=args.ratio,
        base_threshold=args.threshold,
        max_levels=args.levels,
        llm_config=llm_config,
        verbose=args.verbose
    )
    
    try:
        if args.batch or len(args.input) > 1:
            # 批量模式
            app.compress_batch_files(args.input, args.output)
        else:
            # 单文件模式
            if len(args.input) != 1:
                print("❌ 单文件模式只能接受一个输入文件")
                return 1
            app.compress_single_file(args.input[0], args.output)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断操作")
        return 1
    except Exception as e:
        print(f"❌ 程序执行失败: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())