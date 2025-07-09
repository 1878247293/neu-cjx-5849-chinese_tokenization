# -*- coding: UTF-8 -*-
import collections
import re
from sklearn.model_selection import train_test_split

import constant
from utils import strQ2B


class PrepareData:
  def __init__(self, vocab_size, input_file, output_words_file, output_labels_file, dict_file, raw_file,
               input_dict=False, validation_split=0.1):
    """
    构造函数
    :param vocab_size: 词汇表的大小
    :param input_file:  输入语料的完整文件路径
    :param output_words_file: 输出的训练集字符索引文件完整路径
    :param output_labels_file: 输出的训练集标签索引文件完整路径
    :param dict_file: 词典文件的完整路径
    :param input_dict: 指定是否输入词典，若为True，则使用dict_file指定的词典，若为False，则根据语料和vocab_size生成词典，并输出至dict_file指定的位置，默认为False
    :param raw_file: 输出的语料库未切分的原始语料文件完整路径
    :param validation_split: 验证集所占的比例，默认为0.1 (10%)
    """
    self.input_file = input_file
    self.output_words_file = output_words_file
    self.output_labels_file = output_labels_file
    self.dict_file = dict_file
    self.input_dict = input_dict
    self.vocab_size = vocab_size
    self.validation_split = validation_split

    if raw_file is None or raw_file == '':
      self.output_raw_file = False
    else:
      self.output_raw_file = True
      self.raw_file = raw_file
      
    self.SPLIT_CHAR = '  '
    all_sentences = self.read_sentences()
    
    # 使用 train_test_split 分割数据集
    self.train_sentences, self.validation_sentences = train_test_split(
        all_sentences, test_size=self.validation_split, random_state=42)
    
    self.count = [['UNK', 0], ['STRT', 0], ['END', 0]]
    
    if self.input_dict:
      self.dictionary = self.read_dictionary(self.dict_file)
    else:
      # 基于训练集构建词典，以防数据泄露
      self.dictionary = self.build_dictionary(self.train_sentences)

  def read_sentences(self):
    file = open(self.input_file, 'r', encoding='utf-8')
    content = file.read()
    # 增加标点符号处理
    # 在标点符号两侧添加空格，以便将它们视为独立的token
    content = re.sub(r'([,.?!;:"\'\(\)\[\]{}<>`~#$@%^&*\-_=+·/\\|《》“”‘’、。，！？；：（）【】])', r' \1 ', content)
    sentences = re.sub('[ ]+', self.SPLIT_CHAR, strQ2B(content)).splitlines()  # 将词分隔符统一为双空格
    sentences = list(filter(None, sentences))  # 去除空行
    file.close()
    return sentences

  def build_raw_corpus(self, sentences, raw_file_path):
    with open(raw_file_path, 'w', encoding='utf-8') as file:
        for sentence in sentences:
            file.write(sentence.replace(' ', '') + '\n')

  def build_dictionary(self, sentences):
    dictionary = {}
    words = ''.join(sentences).replace(' ', '')
    vocab_count = len(collections.Counter(words))
    self.count.extend(collections.Counter(words).most_common(self.vocab_size - 3))

    for word, _ in self.count:
      dictionary[word] = len(dictionary)
    return dictionary

  def read_dictionary(self, dict_path):
    dict_file = open(dict_path, 'r', encoding='utf-8')
    dict_content = dict_file.read().splitlines()
    dictionary = {}
    dict_arr = map(lambda item: item.split(' '), dict_content)
    for _, dict_item in enumerate(dict_arr):
      dictionary[dict_item[0]] = dict_item[1]
    dict_file.close()
    return dictionary

  def build_basic_dataset(self, sentences):
    words_index_list = []
    unk_count = 0
    for sentence in sentences:
        sentence = sentence.replace(' ', '')
        sen_data = []
        for word in sentence:
            if word in self.dictionary:
                index = self.dictionary[word]
            else:
                index = 0
                unk_count += 1
            sen_data.append(index)
        words_index_list.append(sen_data)
    self.count[0][1] += unk_count
    return words_index_list

  def build_corpus_dataset(self, sentences):
    labels_index_list = []
    empty = 0
    for sentence in sentences:
        sentence_label = []
        words = sentence.strip().split(self.SPLIT_CHAR)
        for word in words:
            l = len(word)
            if l == 0:
                empty += 1
                continue
            elif l == 1:
                sentence_label.append(0)
            else:
                sentence_label.append(1)
                sentence_label.extend([2] * (l - 2))
                sentence_label.append(3)
        labels_index_list.append(sentence_label)
    return labels_index_list

  def build_test_corpus(self, filename, sentences, labels_index):
    with open(filename, 'w', encoding='utf-8') as file:
      for _, (sentence, sentence_label) in enumerate(zip(sentences, labels_index)):
        file.write(sentence.replace(' ', '') + '\n')
        file.write(' '.join(map(lambda i: str(i), sentence_label)) + '\n')

  def build_exec(self):
    print("正在处理训练集...")
    train_words_index = self.build_basic_dataset(self.train_sentences)
    train_labels_index = self.build_corpus_dataset(self.train_sentences)

    print("正在处理验证集...")
    validation_words_index = self.build_basic_dataset(self.validation_sentences)
    validation_labels_index = self.build_corpus_dataset(self.validation_sentences)

    def write_data(words_file_path, labels_file_path, words_index, labels_index):
        with open(words_file_path, 'w+', encoding='utf-8') as words_file, \
             open(labels_file_path, 'w+', encoding='utf-8') as labels_file:
            for words, labels in zip(words_index, labels_index):
                words_file.write(' '.join(str(word) for word in words) + '\n')
                labels_file.write(' '.join(str(label) for label in labels) + '\n')

    # 写入训练集文件
    write_data(self.output_words_file, self.output_labels_file, train_words_index, train_labels_index)
    print(f"训练集数据已写入: {self.output_words_file}, {self.output_labels_file}")
    
    # 写入验证集文件
    validation_words_file = self.output_words_file.replace('_training_', '_validation_')
    validation_labels_file = self.output_labels_file.replace('_training_', '_validation_')
    write_data(validation_words_file, validation_labels_file, validation_words_index, validation_labels_index)
    print(f"验证集数据已写入: {validation_words_file}, {validation_labels_file}")

    if not self.input_dict:
      with open(self.dict_file, 'w+', encoding='utf-8') as dict_file:
          for (word, index) in self.dictionary.items():
              dict_file.write(word + ' ' + str(index) + '\n')
      print(f"词典文件已写入: {self.dict_file}")
      
    if self.output_raw_file:
      self.build_raw_corpus(self.train_sentences, self.raw_file)

if __name__ == '__main__':
  prepare_pku = PrepareData(constant.VOCAB_SIZE, 'corpus/pku_training.utf8', 'corpus/pku_training_words.txt',
                            'corpus/pku_training_labels.txt', 'corpus/pku_training_dict.txt',
                            'corpus/pku_training_raw.utf8')
  prepare_pku.build_exec()
  # prepare_msr = PrepareData(constant.VOCAB_SIZE,'corpus/msr_training.utf8', 'corpus/msr_training_words.txt',
  #                           'corpus/msr_training_labels.txt', 'corpus/msr_training_dict.txt')
  # prepare_msr.build_exec()
