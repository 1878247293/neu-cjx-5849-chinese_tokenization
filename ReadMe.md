# 基于深度学习的中文分词

## 参考论文：

* [deep learning for chinese word segmentation and pos tagging](www.aclweb.org/anthology/D13-1061) （已完全实现，文件[`seg_dnn.py`](https://github.com/supercoderhawk/DNN_CWS/blob/master/seg_dnn.py)）

## Todo List

1.实现验证集
2.目前优化后的DNN处理后的结果还是不太准


## 项目概述
本项目是一个基于深度学习的中文分词系统，使用TensorFlow实现的神经网络模型：
- DNN（深度神经网络）分词

## 环境要求
- Python 3.6+
- TensorFlow 2.0+
- NumPy 1.18+
- 操作系统：Linux

## 配置方式

使用 `setup_environment.sh` 脚本进行完整的环境配置：

```bash
./setup_environment_fast.sh
```

此脚本会：
- 检查Python环境
- 可选创建虚拟环境
- 安装所有依赖包
- 检查项目文件结构
- 初始化训练数据
- 创建示例使用脚本
- 运行环境测试


## 文件说明

### 配置脚本
- `setup_environment.sh` - 完整环境配置脚本
- `requirements.txt` - Python依赖列表

### 核心代码文件
- `seg_dnn.py` - DNN分词实现
- `prepare_data.py` - 数据预处理
- `init.py` - 数据初始化脚本

### 测试和示例
- `test.py` - 项目测试脚本
- `example_usage.py` - 使用示例（由配置脚本生成）

### 数据目录
- `corpus/` - 语料库文件
- `model/` - 训练好的模型文件
- `tmp/` - 临时文件目录
- `logs/` - 日志文件目录

## 使用方法

### 基本使用
```python
python3 init.py   #生成训练数据

python3 train_models.py  #训练模型，保存到model目录下（这里最好先删除原本的，再进行训练）

python3 custom_seg.py   #运行，可以选择输入文本或者文件
```

## 生成训练数据
**执行流程**：
1. **PrepareData**: 处理PKU语料库，生成字典、词汇和标签文件
2. **TransformDataDNN**: 为DNN模型生成训练数据

### 基础语料文件 (corpus/)

```
corpus/
├── dict.utf8                    # 词典文件 (4001个词汇)
├── pku_training_dict.txt        # PKU训练字典
├── pku_training_labels.txt      # PKU训练标签 (3.5MB)
├── pku_training_words.txt       # PKU训练词汇 (6.1MB) 
├── pku_training_raw.utf8        # PKU原始训练数据 (5.0MB)
└── pku_training.utf8            # PKU训练语料 (7.3MB)
```

### DNN训练数据 (corpus/dnn/)

```
corpus/dnn/
├── words_batch.pkl              # 词汇批次数据 (22MB)
├── words_batch_flat.npy         # 平铺词汇数据 (42MB) 
├── labels_batch.pkl             # 标签批次数据 (7.6MB)
└── labels_batch_flat.npy        # 平铺标签数据 (14MB)
```
### 字典文件格式

**文件**: `corpus/dict.utf8`

```
UNK 0      # 未知词
STRT 1     # 句子开始
END 2      # 句子结束
, 3        # 标点符号
的 4       # 高频词汇
...
```

- **总词汇量**: 4001个
- **编码方式**: 每个词汇对应一个唯一的数字ID
- **特殊符号**: UNK(未知)、STRT(开始)、END(结束)

### SBIE标注标签

训练数据使用SBIE标注法：
- **S (Single)**: 单字词 - 标签值: 0
- **B (Begin)**: 词汇开始 - 标签值: 1  
- **I (Inside)**: 词汇中间 - 标签值: 2
- **E (End)**: 词汇结束 - 标签值: 3

**示例**：
```
输入: 北京大学
字符: 北 京 大 学
标签: B  E  B  E  (1 3 1 3)
分词: 北京 | 大学
```
### 关键参数：
```python
# constant.py中的DNN参数
DNN_SKIP_WINDOW = 2        # 上下文窗口大小
VOCAB_SIZE = 4001          # 词汇表大小
BATCH_SIZE = 64            # 批次大小
EMBED_SIZE = 128           # 词向量维度
```
### DNN模型优化

```python
# 在constant.py中调整参数
DNN_LEARNING_RATE = 0.001      # 学习率
DNN_TRAINING_STEPS = 100000    # 训练步数
DNN_SKIP_WINDOW = 2            # 上下文窗口
EMBED_SIZE = 128               # 词嵌入维度
```



## 训练模型
新训练的模型会自动成为默认模型，整个过程无缝衔接。

如果您保存了多个模型（例如，不同轮次或不同参数的版本），您也可以在分词时**手动指定**要加载的模型。

`seg` 函数接受一个 `model_path` 参数。

**示例**：
假设您在训练DNN时，保存了一个名为 `model/dnn_epoch_1.ckpt` 的模型。

```python
# 使用指定的、非默认的模型
result_custom = cws.seg("我爱北京天安门", model_path='model/dnn_epoch_1.ckpt') 
```
### 训练模型文件结构说明

当TensorFlow保存一个模型时（例如 `model.ckpt`），会生成多个文件：
- `model.ckpt.meta`: 保存了模型的**图结构**。
- `model.ckpt.data-*`: 保存了模型中所有**变量的值**（权重、偏置等）。
- `model.ckpt.index`: 一个索引文件，记录了变量名和其在data文件中的位置。
- `checkpoint`: 一个文本文件，记录了最新的模型检查点文件是哪一个。

在加载时，您只需要提供基础路径 `model.ckpt`，TensorFlow会自动找到所有相关文件。

## DNN_CWS 输出功能使用指南

### 输出目录结构

```
output/
├── dnn_results_20250708_004556.json      # DNN分词结果(JSON格式)
├── dnn_results_20250708_004556.txt       # DNN分词结果(文本格式)
├── test_summary_20250708_004557.md       # 测试总结报告
├── custom_seg_20250708_005000.json       # 自定义分词结果
└── batch_seg_20250708_005100.json        # 批量分词结果
```
### output功能的使用
**脚本**: `custom_seg.py`

**功能**:
- 交互式分词工具
- 支持单句分词和批量分词
- 自动保存结果到文件

**使用方法**:
```bash
python3 custom_seg.py
```

**操作选项**:
1. **输入文本进行分词** - 手动输入要分词的文本
2. **从文件批量分词** - 从文本文件读取多行进行批量处理
3. **退出** - 退出程序

## 以下是test.py样例：

```python
from seg_dnn import SegDNN
import constant

# 初始化分词器
cws = SegDNN(constant.VOCAB_SIZE, 50, constant.DNN_SKIP_WINDOW)

# 进行分词
result = cws.seg('我爱北京天安门')
print(result[0])  # 输出分词结果
```


## 注意事项

1. **语料文件**：项目需要以下语料文件才能完全运行：
   - `corpus/pku_training.utf8` (PKU训练语料)
   - `corpus/msr_training.utf8` (MSR训练语料)

2. **内存要求**：训练过程需要较大内存，建议至少4GB可用内存

3. **GPU支持**：如有NVIDIA GPU，TensorFlow会自动利用GPU加速

4. **虚拟环境**：建议使用虚拟环境避免依赖冲突


