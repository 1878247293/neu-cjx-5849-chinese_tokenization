#-*- coding: UTF-8 -*-
from prepare_data import PrepareData
from transform_data_dnn import TransformDataDNN
import constant
from shutil import copyfile
import os

def init():
  # 确保dnn目录存在
  os.makedirs('corpus/dnn', exist_ok=True)
  
  # 步骤1：分割原始语料为训练集和验证集文件
  print("--- 步骤 1: 准备和分割语料库 ---")
  prepare_pku = PrepareData(constant.VOCAB_SIZE, 'corpus/pku_training.utf8', 'corpus/pku_training_words.txt',
                            'corpus/pku_training_labels.txt', 'corpus/pku_training_dict.txt',
                            'corpus/pku_training_raw.utf8')
  prepare_pku.build_exec()
  
  # 步骤2：复制主词典
  dict_name = 'corpus/pku_training_dict.txt'
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