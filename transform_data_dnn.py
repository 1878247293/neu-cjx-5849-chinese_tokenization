# -*- coding: UTF-8 -*-
import numpy as np
import os
import pickle
from transform_data import TransformData
import constant


class TransformDataDNN(TransformData):
  def __init__(self, skip_window, dataset_type='training'):
    """
    构造函数
    :param skip_window: 上下文窗口大小
    :param dataset_type: 数据集类型, 'training' 或 'validation'
    """
    if dataset_type == 'training':
        corpus_base_name = 'pku_training'
    else:
        corpus_base_name = 'pku_validation'
    
    TransformData.__init__(self, 'corpus/dict.utf8', [corpus_base_name])
    
    self.skip_window = skip_window
    self.window_length = 2 * self.skip_window + 1
    
    # 根据数据集类型动态设置路径
    suffix = '_val' if dataset_type == 'validation' else ''
    self.words_batch_base_path = f'corpus/dnn/words_batch{suffix}'
    self.words_batch_flat_path = f'corpus/dnn/words_batch_flat{suffix}.npy'
    self.labels_batch_base_path = f'corpus/dnn/labels_batch{suffix}'
    self.labels_batch_flat_path = f'corpus/dnn/labels_batch_flat{suffix}.npy'

    # 简化加载/生成逻辑
    if os.path.exists(self.words_batch_flat_path) and os.path.exists(self.labels_batch_flat_path):
        print(f"从缓存加载 {dataset_type} 数据...")
        self.words_batch_flat = np.load(self.words_batch_flat_path)
        self.labels_batch_flat = np.load(self.labels_batch_flat_path)
    else:
        print(f"正在为 {dataset_type} 生成新数据...")
        _, self.labels_batch_flat = self.generate_batch()
        # 注意: generate_batch现在只返回flat数据，因为这是唯一被使用的
        self.words_batch_flat = self.generate_words_batch_flat()

    self.words_count = len(self.labels_batch_flat)
    self.whole_words_batch = self.words_batch_flat.reshape([self.words_count, self.window_length])
    self.whole_labels_batch = self.labels_batch_flat.reshape([self.words_count])

  def generate_words_batch_flat(self):
      """只生成和返回展平的词汇批次数据"""
      words_batch = []
      for words in self.words_index:
          if len(words) < self.skip_window:
              continue
          extend_words = [1] * self.skip_window + words + [2] * self.skip_window
          for i in range(len(words)):
              context = extend_words[i : i + self.window_length]
              words_batch.extend(context)
      return np.array(words_batch, dtype=np.int32)

  def generate_batch(self):
    # 此方法现在只处理标签，词汇由新的专用方法处理
    words_batch = []
    labels_batch = []
    for _,labels in enumerate(self.labels_index):
      labels_batch.extend(labels)

    return np.array(words_batch), np.array(labels_batch)

  def generate_exe(self):
    # 只保存flat数据，因为这是唯一被使用的
    print(f"正在保存展平的数据到 {self.words_batch_flat_path} 和 {self.labels_batch_flat_path}...")
    np.save(self.words_batch_flat_path, self.words_batch_flat)
    np.save(self.labels_batch_flat_path, self.labels_batch_flat)
    print("保存完成。")


if __name__ == '__main__':
  print("正在生成训练数据...")
  trans_dnn_train = TransformDataDNN(constant.DNN_SKIP_WINDOW, dataset_type='training')
  trans_dnn_train.generate_exe()
  
  print("\n正在生成验证数据...")
  trans_dnn_val = TransformDataDNN(constant.DNN_SKIP_WINDOW, dataset_type='validation')
  trans_dnn_val.generate_exe()
