#!/usr/bin/env python3
"""
CPU环境配置脚本
在导入mamba_ssm之前设置环境变量，强制使用fallback实现
"""

import os
import sys

def setup_cpu_environment():
    """设置CPU环境变量，强制使用fallback实现"""
    os.environ["CAUSAL_CONV1D_FORCE_FALLBACK"] = "TRUE"
    os.environ["SELECTIVE_SCAN_FORCE_FALLBACK"] = "TRUE"
    print("CPU环境变量已设置:")
    print("  CAUSAL_CONV1D_FORCE_FALLBACK=TRUE")
    print("  SELECTIVE_SCAN_FORCE_FALLBACK=TRUE")

if __name__ == "__main__":
    setup_cpu_environment()
    print("\n现在可以安全地导入mamba_ssm模块了")
    print("使用方法:")
    print("  python -c \"import setup_cpu_env; import mamba_ssm\"")
    print("  或者在脚本开头添加: exec(open('setup_cpu_env.py').read())") 