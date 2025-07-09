#!/bin/bash

# 基于深度学习的中文分词环境配置脚本（国内加速版）
# 
# 作用：使用国内镜像源快速配置DNN_CWS项目的运行环境
# 创建时间：$(date)

set -e  # 遇到错误时立即退出

echo "=========================================="
echo "  DNN_CWS 环境配置脚本（国内加速版）"
echo "=========================================="

# 检查Python版本
echo "检查Python环境..."
PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "当前Python版本: $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -lt 3 ]; then
    echo "错误: 本项目需要Python 3.x，当前版本为 $PYTHON_VERSION"
    exit 1
fi

# 检查pip
echo "检查pip..."
if ! command -v pip3 &> /dev/null; then
    echo "pip3未找到，正在安装..."
    sudo apt-get update
    sudo apt-get install -y python3-pip
fi

# 升级pip并配置国内源
echo "配置pip国内镜像源（清华大学）..."
pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip3 config set install.trusted-host pypi.tuna.tsinghua.edu.cn
pip3 install --upgrade pip

# 创建虚拟环境（可选）
read -p "是否创建虚拟环境？(y/n): " CREATE_VENV
if [ "$CREATE_VENV" = "y" ] || [ "$CREATE_VENV" = "Y" ]; then
    echo "创建虚拟环境..."
    if ! command -v python3-venv &> /dev/null; then
        sudo apt-get install -y python3-venv
    fi
    
    python3 -m venv dnn_cws_env
    source dnn_cws_env/bin/activate
    
    # 在虚拟环境中也配置国内源
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn
    echo "虚拟环境已创建并激活，已配置国内镜像源"
fi

# 询问GPU支持
echo ""
read -p "是否需要GPU支持？(y/n，如果选择n将安装CPU版本，下载更快): " GPU_SUPPORT

# 安装项目依赖
echo "开始安装项目依赖..."

if [ "$GPU_SUPPORT" = "y" ] || [ "$GPU_SUPPORT" = "Y" ]; then
    echo "安装TensorFlow GPU版本..."
    pip3 install tensorflow
else
    echo "安装TensorFlow CPU版本（更小更快）..."
    pip3 install tensorflow-cpu
fi

# 安装NumPy（通常TensorFlow会自动安装，但确保版本兼容）
echo "安装NumPy..."
pip3 install numpy

# 安装其他依赖（使用国内源）
echo "安装其他依赖..."
pip3 install matplotlib scikit-learn

# 检查必要的目录和文件
echo "检查项目文件结构..."

# 检查语料库目录
if [ ! -d "corpus" ]; then
    echo "警告: corpus目录不存在，正在创建..."
    mkdir -p corpus
fi

# 检查必要的语料文件
REQUIRED_FILES=("corpus/pku_training.utf8" "corpus/msr_training.utf8")
MISSING_FILES=0

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "警告: 缺少必要的语料文件 $file"
        MISSING_FILES=1
    fi
done

if [ $MISSING_FILES -eq 1 ]; then
    echo ""
    echo "注意: 检测到缺少语料文件，请确保从项目原始仓库下载完整的语料库文件"
    echo "主要需要的文件："
    echo "  - corpus/pku_training.utf8 (PKU训练语料)"
    echo "  - corpus/msr_training.utf8 (MSR训练语料)"
fi

# 创建必要的目录
echo "创建必要的目录..."
mkdir -p tmp logs

# 检查模型目录
if [ ! -d "model" ]; then
    echo "创建model目录..."
    mkdir -p model
fi

# 初始化数据（如果语料文件存在）
if [ -f "corpus/pku_training.utf8" ]; then
    echo "初始化训练数据..."
    python3 init.py
    echo "数据初始化完成"
else
    echo "跳过数据初始化（缺少语料文件）"
fi

# 运行简单测试
echo "运行环境测试..."
python3 -c "
import tensorflow as tf
import numpy as np
print('TensorFlow版本:', tf.__version__)
print('NumPy版本:', np.__version__)
print('TensorFlow GPU可用:', tf.config.list_physical_devices('GPU'))
print('环境测试通过！')
"

# 创建示例使用脚本
echo "创建示例使用脚本..."
cat > example_usage.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
DNN_CWS使用示例脚本
"""

from seg_dnn import SegDNN
import constant

def main():
    print("=== DNN中文分词示例 ===")
    
    # 初始化分词器
    cws = SegDNN(constant.VOCAB_SIZE, 50, constant.DNN_SKIP_WINDOW)
    
    # 测试句子
    test_sentences = [
        "我爱北京天安门",
        "小明来自南京师范大学", 
        "迈向充满希望的新世纪",
        "深度学习技术在自然语言处理中的应用"
    ]
    
    print("开始分词测试...")
    
    for sentence in test_sentences:
        try:
            result = cws.seg(sentence)
            print(f"原句: {sentence}")
            print(f"分词: {result[0]}")
            print("-" * 50)
        except Exception as e:
            print(f"分词失败: {sentence}, 错误: {e}")
    
    print("测试完成！")

if __name__ == '__main__':
    main()
EOF

chmod +x example_usage.py

echo ""
echo "=========================================="
echo "  环境配置完成！"
echo "=========================================="
echo ""
echo "配置总结："
echo "✓ Python环境检查完成"
echo "✓ pip国内镜像源配置完成"
if [ "$GPU_SUPPORT" = "y" ] || [ "$GPU_SUPPORT" = "Y" ]; then
    echo "✓ TensorFlow GPU版本安装完成"
else
    echo "✓ TensorFlow CPU版本安装完成"
fi
echo "✓ 其他依赖包安装完成"
echo "✓ 项目目录结构检查完成"
if [ -f "corpus/pku_training.utf8" ]; then
    echo "✓ 训练数据初始化完成"
else
    echo "⚠ 训练数据初始化跳过（缺少语料文件）"
fi
echo "✓ 示例脚本创建完成"
echo ""
echo "使用方法："
echo "1. 测试环境: python3 example_usage.py"
echo "2. 查看详细示例: python3 test.py"
echo "3. 开始训练模型: 参考seg_dnn.py中的train()方法"
echo ""

if [ "$CREATE_VENV" = "y" ] || [ "$CREATE_VENV" = "Y" ]; then
    echo "注意: 如果使用了虚拟环境，下次使用时请先激活:"
    echo "source dnn_cws_env/bin/activate"
    echo ""
fi

echo "镜像源已配置为清华大学源，后续安装将自动使用加速源"
echo "项目文档: ReadMe.md"
echo "" 