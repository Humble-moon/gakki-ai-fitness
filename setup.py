"""
setup.py - 项目安装配置文件

角色：定义包名、版本、依赖和打包规则。支持 pip install -e . 开发模式安装。
被调用者：pip install -e . 或 pip install . 命令执行时读取。
调用者：不调用项目内部模块。
"""
from setuptools import setup, find_packages

setup(
    name="gakki-ai-fitness",    # 包名，pip list 显示的名称
    version="0.1.0",           # 当前版本号
    packages=find_packages(),  # 自动发现 src/ 下所有 Python 包，避免手动罗列
    python_requires=">=3.11",  # Python 版本要求，代码中使用了 3.11+ 的语法特性
)
