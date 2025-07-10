#-*- coding: UTF-8 -*-
from prepare_data import PrepareData
from transform_data_dnn import TransformDataDNN
import constant
from shutil import copyfile
import os
import glob

def cleanup_cache():
    """清理旧的.npy缓存文件"""
    print("--- 步骤 -1: 清理旧的数据缓存 ---")
    cache_files = glob.glob('corpus/dnn/*.npy')
    if not cache_files:
        print("✅ 没有找到缓存文件，无需清理。")
        return
    
    print("发现以下缓存文件将被删除:")
    for f in cache_files:
        print(f"  - {f}")
        
    for f in cache_files:
        os.remove(f)
    print(f"✅ {len(cache_files)} 个缓存文件已删除。")

def combine_corpora():
    """合并pku和msr语料库"""
    pku_corpus_path = 'corpus/pku_training.utf8'
    msr_corpus_path = 'corpus/msr_training.utf8'
    combined_corpus_path = 'corpus/combined_training.utf8'

    print("--- 步骤 0: 合并PKU和MSR语料库 ---")
    if not os.path.exists(pku_corpus_path) or not os.path.exists(msr_corpus_path):
        print(f"❌ 错误: 缺少 {pku_corpus_path} 或 {msr_corpus_path}。")
        return None

    try:
        with open(combined_corpus_path, 'w', encoding='utf-8') as outfile:
            for corpus_path in [pku_corpus_path, msr_corpus_path]:
                with open(corpus_path, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        outfile.write(line)
        print(f"✅ 语料库已合并到: {combined_corpus_path}")
        return combined_corpus_path
    except Exception as e:
        print(f"❌ 合并语料库失败: {e}")
        return None

def init():
  # 确保dnn目录存在
  os.makedirs('corpus/dnn', exist_ok=True)

  # 新增步骤：清理缓存
  cleanup_cache()

  # 新增步骤：合并语料库
  combined_corpus_path = combine_corpora()
  if not combined_corpus_path:
      return
  
  # 步骤1：分割原始语料为训练集和验证集文件
  print("\n--- 步骤 1: 准备和分割语料库 ---")
  prepare_data = PrepareData(constant.VOCAB_SIZE, combined_corpus_path, 'corpus/training_words.txt',
                            'corpus/training_labels.txt', 'corpus/training_dict.txt',
                            'corpus/training_raw.utf8')
  prepare_data.build_exec()
  
  # 步骤2：复制主词典
  dict_name = 'corpus/training_dict.txt'
  copyfile(dict_name,'corpus/dict.utf8')
  print(f"\n--- 步骤 2: 主词典已复制到 corpus/dict.utf8 ---")

  # 步骤3：为DNN模型生成训练和验证的批处理数据
  print("\n--- 步骤 3: 为DNN生成批处理数据 ---")
  print("正在生成训练数据...")
  trans_dnn_train = TransformDataDNN(constant.DNN_SKIP_WINDOW, dataset_type='training')
  trans_dnn_train.generate_exe()
  
  print("\n正在生成验证数据...")
  trans_dnn_val = TransformDataDNN(constant.DNN_SKIP_WINDOW, dataset_type='validation')
  trans_dnn_val.generate_exe()
  
  print("\n✅ 数据初始化完成!")

if __name__ == '__main__':
  init()