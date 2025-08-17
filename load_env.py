#!/usr/bin/env python3
"""
环境变量加载器
确保正确加载.env文件中的配置
"""

import os
from pathlib import Path


def load_env_file(env_path: str = ".env") -> bool:
    """加载.env文件"""
    env_file = Path(env_path)
    
    if not env_file.exists():
        print(f"⚠️ 找不到 {env_path} 文件")
        return False
    
    try:
        loaded_count = 0
        with open(env_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # 跳过空行和注释
                if not line or line.startswith('#'):
                    continue
                
                # 解析键值对
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # 移除引号
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    
                    # 设置环境变量
                    os.environ[key] = value
                    loaded_count += 1
                    
                    # 只显示非敏感信息
                    if 'KEY' in key or 'TOKEN' in key or 'PASSWORD' in key:
                        display_value = value[:8] + "..." if len(value) > 8 else "***"
                        print(f"  {key}: {display_value}")
                    else:
                        print(f"  {key}: {value}")
        
        print(f"✅ 成功加载 {loaded_count} 个环境变量")
        return True
        
    except Exception as e:
        print(f"❌ 加载 {env_path} 失败: {e}")
        return False


def verify_api_key() -> bool:
    """验证API key是否可用"""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OPENAI_API_KEY 未设置")
        return False
    
    if len(api_key) < 20:
        print("❌ OPENAI_API_KEY 格式不正确")
        return False
    
    if not api_key.startswith("sk-"):
        print("❌ OPENAI_API_KEY 格式不正确（应以sk-开头）")
        return False
    
    print(f"✅ OPENAI_API_KEY 已设置: {api_key[:8]}...")
    return True


if __name__ == "__main__":
    print("🔧 加载环境配置...")
    
    if load_env_file():
        verify_api_key()
    else:
        print("❌ 环境配置加载失败")