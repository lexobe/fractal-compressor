#!/usr/bin/env python3
"""
语料压缩脚本 - 为每个语料生成对应的结果文件
将所有语料文件压缩，结果保存在results目录中
"""

import os
import sys
import glob
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

def main():
    """主函数：处理所有语料文件"""
    
    # 确保results目录存在
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # 找到所有语料文件
    corpus_files = glob.glob("corpus_*.txt")
    corpus_files.sort()  # 按文件名排序
    
    if not corpus_files:
        print("❌ 未找到任何语料文件 (corpus_*.txt)")
        return
    
    print(f"🎯 找到 {len(corpus_files)} 个语料文件")
    print("📁 结果将保存在 results/ 目录中")
    print("="*50)
    
    # 处理每个语料文件
    for i, corpus_file in enumerate(corpus_files, 1):
        # 生成输出文件名
        corpus_name = Path(corpus_file).stem  # 去掉扩展名
        output_file = results_dir / f"{corpus_name}_result.json"
        
        print(f"\n[{i}/{len(corpus_files)}] 处理: {corpus_file}")
        print(f"          输出: {output_file}")
        
        # 调用独立压缩程序
        cmd = f"python3 fractal_compress.py {corpus_file} -o {output_file}"
        print(f"          执行: {cmd}")
        
        # 执行压缩命令
        exit_code = os.system(cmd)
        
        if exit_code == 0:
            print(f"          ✅ 成功")
        else:
            print(f"          ❌ 失败 (退出码: {exit_code})")
    
    print("\n" + "="*50)
    print("🎉 批量压缩完成！")
    print(f"📂 查看结果: ls results/")
    
    # 显示结果文件列表
    result_files = list(results_dir.glob("*.json"))
    if result_files:
        print(f"\n📋 生成的结果文件 ({len(result_files)}个):")
        for result_file in sorted(result_files):
            file_size = result_file.stat().st_size
            print(f"   {result_file.name} ({file_size} bytes)")

if __name__ == "__main__":
    main()