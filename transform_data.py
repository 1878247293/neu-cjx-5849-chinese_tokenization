# -*- coding: UTF-8 -*-


class TransformData:
  def __init__(self, dict_path, corpuses):
    self.dict_path = dict_path
    self.dictionary = self.read_dictionary()
    self.reverse_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))
    self.words_index = []
    self.labels_index = []
    if corpuses is not None or len(corpuses) != 0:
      for _, corpus_base_path in enumerate(corpuses):
        base_path = 'corpus/' + corpus_base_path
        self.read_words(base_path + '_words.txt')
        self.read_labels(base_path + '_labels.txt')

  def read_dictionary(self):
    dict_file = open(self.dict_path, 'r', encoding='utf-8')
    dict_content = dict_file.read().splitlines()
    dictionary = {}
    dict_arr = map(lambda item: item.split(' '), dict_content)
    for _, dict_item in enumerate(dict_arr):
      dictionary[dict_item[0]] = int(dict_item[1])
    dict_file.close()
    return dictionary

  def read_words(self, path):
    file = open(path, 'r', encoding='utf-8')
    words = file.read().splitlines()
    for index, word in enumerate(words):
      if word.strip():  # 只处理非空行
        word_parts = [part for part in word.split(' ') if part.strip()]  # 过滤空字符串
        if word_parts:  # 确保有有效的词索引
          self.words_index.append(list(map(int, word_parts)))
    file.close()

  def read_labels(self, path):
    file = open(path, 'r', encoding='utf-8')
    labels = file.read().splitlines()
    for label in labels:
      if label.strip():  # 只处理非空行
        label_parts = [part for part in label.split(' ') if part.strip()]  # 过滤空字符串
        if label_parts:  # 确保有有效的标签索引
          self.labels_index.append(list(map(int, label_parts)))
    file.close()

  def generate_batch(self):
    raise NotImplementedError('must implement generate batch function')
