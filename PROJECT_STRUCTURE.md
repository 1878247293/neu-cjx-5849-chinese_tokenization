# DNN_CWS 项目结构说明

## 核心文件

### 主要脚本
- `init.py` - 数据初始化脚本
- `train_models.py` - 模型训练脚本  
- `custom_seg.py` - 分词测试脚本
- `seg_dnn.py` - DNN分词核心模块

### 数据处理
- `prepare_data.py` - 数据预处理
- `transform_data_dnn.py` - DNN数据转换
- `transform_data.py` - 基础数据转换
- `utils.py` - 工具函数

### 配置文件
- `constant.py` - 常量配置
- `requirements_fast.txt` - Python依赖

## 目录结构

### `corpus/` - 语料库目录
- `pku_training.utf8` - PKU训练语料
- `msr_training.utf8` - MSR训练语料
- `dict.utf8` - 词典文件
- `pku_training_words.txt` - 训练词序列
- `pku_training_labels.txt` - 训练标签序列
- `pku_validation_words.txt` - 验证词序列
- `pku_validation_labels.txt` - 验证标签序列
- `pku_training_raw.utf8` - 原始训练语料
- `dnn/` - DNN训练数据（.npy文件）
- `patches/` - 改进补丁文件

### `model/` - 模型保存目录
- `best_model.ckpt.*` - 最佳训练模型

### `output/` - 输出结果目录
- `batch_seg_*.txt` - 分词结果文本
- `batch_seg_*.json` - 分词结果JSON
- `dnn_training_log_*.txt` - 训练日志

### `backup_*/` - 备份目录
- 原始文件的备份

## 测试文件
- `test_article_official_style.txt` - 官方风格测试文本

## 文档
- `ReadMe.md` - 项目说明
- `MODEL_CAPABILITIES.md` - 模型能力说明
- `IMPROVEMENT_GUIDE.md` - 改进指南
- `DNN优化详解.md` - DNN优化说明
- `为DNN_CWS项目引入验证集机制的重构方案.md` - 重构方案

## 环境配置
- `setup_environment_fast.sh` - 环境配置脚本
- `dnn_cws_env/` - Python虚拟环境

---
*清理时间: 2025-07-10 14:48:46*
