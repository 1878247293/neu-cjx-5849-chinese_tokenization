# -*- coding: UTF-8 -*-
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import math
import numpy as np
import time
import os
import traceback
from transform_data_dnn import TransformDataDNN
import constant

class SegDNN:
    """
    优化版本的DNN中文分词器 - TensorFlow 2.x兼容
    """
    def __init__(self, vocab_size, embed_size, skip_window):
        self.vocab_size = vocab_size
        self.embed_size = 100 # <--- 增加嵌入维度
        self.skip_window = skip_window
        
        # Hyperparameters
        self.alpha = 0.001
        self.h1 = 600 # <--- 增加隐藏层神经元
        self.h2 = 300 # <--- 增加隐藏层神经元
        self.tags_count = 4
        self.window_length = 2 * self.skip_window + 1
        self.concat_embed_size = self.embed_size * self.window_length
        self.dropout_rate = 0.5
        self.batch_size = 32
        
        # Data loader
        self.tran = TransformDataDNN(self.skip_window, vocab_size) # 传递vocab_size
        self.dictionary = self.tran.dictionary
        
        # Build graph
        self._build_graph()

    def viterbi(self, emission, A, init_A, return_score=False):
        """
        维特比算法的实现，所有输入和返回参数均为numpy数组对象
        :param emission: 发射概率矩阵，对应于本模型中的分数矩阵，4*length
        :param A: 转移概率矩阵，4*4
        :param init_A: 初始转移概率矩阵，4
        :param return_score: 是否返回最优路径的分值，默认为False
        :return: 最优路径，若return_score为True，返回最优路径及其对应分值
        """

        length = emission.shape[1]
        path = np.ones([self.tags_count, length], dtype=np.int32) * -1
        corr_path = np.zeros([length], dtype=np.int32)
        path_score = np.ones([self.tags_count, length], dtype=np.float64) * (np.finfo('f').min / 2)
        path_score[:, 0] = init_A + emission[:, 0]

        for pos in range(1, length):
          for t in range(self.tags_count):
            for prev in range(self.tags_count):
              temp = path_score[prev][pos - 1] + A[prev][t] + emission[t][pos]
              if temp >= path_score[t][pos]:
                path[t][pos] = prev
                path_score[t][pos] = temp

        max_index = np.argmax(path_score[:, -1])
        corr_path[length - 1] = max_index
        for i in range(length - 1, 0, -1):
          max_index = path[max_index][i]
          corr_path[i - 1] = max_index
        if return_score:
          return corr_path, path_score[max_index, -1]
        else:
          return corr_path

    def _build_graph(self):
        """构建优化的TensorFlow计算图"""
        # Placeholders
        self.y = tf.compat.v1.placeholder(tf.int32, [None], name='y')
        self.is_training = tf.compat.v1.placeholder(tf.bool, name='is_training')
        self.x = tf.compat.v1.placeholder(tf.int32, [None, self.window_length], name='x')
        
        # Embedding layer
        with tf.device('/cpu:0'):
            self.embeddings = tf.Variable(
                tf.random.uniform([self.vocab_size, self.embed_size], -1.0, 1.0, dtype=tf.float32), 
                name='embeddings')
            embed = tf.nn.embedding_lookup(self.embeddings, self.x)
        
        concat_embed = tf.reshape(embed, [-1, self.concat_embed_size])
        
        # Hidden layer 1
        self.w1 = tf.Variable(tf.compat.v1.truncated_normal([self.concat_embed_size, self.h1], stddev=np.sqrt(2.0/self.concat_embed_size)), name='w1')
        self.b1 = tf.Variable(tf.zeros([self.h1]), name='b1')
        h1_out = tf.nn.relu(tf.matmul(concat_embed, self.w1) + self.b1)
        
        # Hidden layer 2
        self.w2 = tf.Variable(tf.compat.v1.truncated_normal([self.h1, self.h2], stddev=np.sqrt(2.0/self.h1)), name='w2')
        self.b2 = tf.Variable(tf.zeros([self.h2]), name='b2')
        h2_out = tf.nn.relu(tf.matmul(h1_out, self.w2) + self.b2)
        
        # Dropout
        h2_dropout = tf.cond(self.is_training,
                             lambda: tf.nn.dropout(h2_out, rate=self.dropout_rate),
                             lambda: h2_out)
        
        # Output layer
        self.w3 = tf.Variable(tf.compat.v1.truncated_normal([self.h2, self.tags_count], stddev=np.sqrt(2.0/self.h2)), name='w3')
        self.b3 = tf.Variable(tf.zeros([self.tags_count]), name='b3')
        
        # Loss function with class weights
        self.logits = tf.matmul(h2_dropout, self.w3) + self.b3
        self.word_scores = tf.nn.softmax(self.logits, axis=-1)
        
        class_weights = tf.constant([1.0, 1.0, 4.5, 1.0])
        ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
        sample_weights = tf.gather(class_weights, self.y)
        weighted_loss = ce_loss * sample_weights
        self.loss = tf.reduce_mean(weighted_loss)
        
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in [self.w1, self.w2, self.w3]]) * 0.001
        self.total_loss = self.loss + l2_loss
        
        # Optimizer
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.compat.v1.train.exponential_decay(self.alpha, global_step, 1000, 0.95, staircase=True)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
        self.train_op = self.optimizer.minimize(self.total_loss, global_step=global_step)
        
        # CRF parameters
        self.A = tf.Variable(tf.compat.v1.truncated_normal([4, 4], stddev=0.1), name='A')
        self.init_A = tf.Variable(tf.compat.v1.truncated_normal([4], stddev=0.1), name='init_A')
        
        # Prediction and Accuracy
        self.predictions = tf.argmax(self.logits, axis=-1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predictions, tf.cast(self.y, tf.int64)), tf.float32))
        self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

    def train_optimized(self, train_data, validation_data, epochs=10, early_stopping_patience=3):
        x_data, y_data = train_data
        x_val, y_val = validation_data
        
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            
            saver = tf.compat.v1.train.Saver(self.params + [self.embeddings, self.A, self.init_A], max_to_keep=1)
            
            print("🚀 开始优化训练 (包含验证)...")
            
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(epochs):
                epoch_start = time.time()
                
                # --- 训练阶段 ---
                total_train_loss, total_train_acc, train_batch_count = 0, 0, 0
                num_batches = math.ceil(len(x_data) / self.batch_size)

                print(f"\nEpoch {epoch+1}/{epochs}")
                print("  Training...")
                for i, (batch_x, batch_y) in enumerate(self._get_batches((x_data, y_data))):
                    feed_dict = {self.x: batch_x, self.y: batch_y, self.is_training: True}
                    _, loss_val, acc_val = sess.run([self.train_op, self.total_loss, self.accuracy], feed_dict)
                    total_train_loss += loss_val
                    total_train_acc += acc_val
                    train_batch_count += 1
                    
                    progress = (i + 1) / num_batches
                    bar_len = 30
                    filled_len = int(round(bar_len * progress))
                    bar = '█' * filled_len + '-' * (bar_len - filled_len)
                    print(f"    [{bar}] {i+1}/{num_batches} - loss: {total_train_loss/train_batch_count:.4f}", end='\r')
                print() 

                avg_train_loss = total_train_loss / train_batch_count
                avg_train_acc = total_train_acc / train_batch_count

                # --- 验证阶段 ---
                print("  Validating...")
                total_val_loss, total_val_acc, val_batch_count = 0, 0, 0
                num_val_batches = math.ceil(len(x_val) / self.batch_size)
                for i, (batch_x, batch_y) in enumerate(self._get_batches((x_val, y_val), shuffle=False)):
                    feed_dict = {self.x: batch_x, self.y: batch_y, self.is_training: False}
                    loss_val, acc_val = sess.run([self.total_loss, self.accuracy], feed_dict)
                    total_val_loss += loss_val
                    total_val_acc += acc_val
                    val_batch_count += 1
                    
                    progress = (i + 1) / num_val_batches
                    bar_len = 30
                    filled_len = int(round(bar_len * progress))
                    bar = '█' * filled_len + '-' * (bar_len - filled_len)
                    print(f"    [{bar}] {i+1}/{num_val_batches}", end='\r')
                print()

                avg_val_loss = total_val_loss / val_batch_count
                avg_val_acc = total_val_acc / val_batch_count
                
                epoch_time = time.time() - epoch_start
                
                print("-" * 50)
                print(f'Epoch {epoch+1:2d}/{epochs} 总结:')
                print(f'  Time: {epoch_time:.1f}s')
                print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}')
                print(f'  Valid Loss: {avg_val_loss:.4f}, Valid Acc: {avg_val_acc:.4f}')
                print("-" * 50)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    print(f"🎉 新的最佳验证损失: {best_val_loss:.4f}。正在保存模型...")
                    saver.save(sess, 'model/best_model.ckpt')
                else:
                    patience_counter += 1
                    print(f"验证损失未改善。容忍计数: {patience_counter}/{early_stopping_patience}")
                    if patience_counter >= early_stopping_patience:
                        print(f'Early stopping at epoch {epoch+1}')
                        break
            
            print(f'✅ 训练完成! 最佳验证Loss: {best_val_loss:.4f}')

    def _get_batches(self, data, shuffle=True):
        x_data, y_data = data
        num_samples = len(x_data)
        
        indices = np.arange(num_samples)
        if shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, num_samples, self.batch_size):
            end_idx = min(i + self.batch_size, num_samples)
            batch_indices = indices[i:end_idx]
            yield x_data[batch_indices], y_data[batch_indices]

    def seg(self, sentence, model_path='model/best_model.ckpt', debug=False):
        try:
            if debug: print(f"--- 开始分词: '{sentence}' ---")
            
            if not os.path.exists(model_path + '.index'):
                if debug: print(f"[Fallback] 模型文件不存在: {model_path}")
                return self.seg_simple(sentence), [0] * len(sentence)

            feature_sequences = self.index2seq(self.sentence2index(sentence))
            
            # 修正: 对Numpy数组的空值判断
            if feature_sequences.size == 0:
                if debug: print("[Fallback] 无法为输入生成特征序列。")
                return self.seg_simple(sentence), [0] * len(sentence)
            
            if debug: print(f"已生成 {len(feature_sequences)} 个特征向量，维度: {np.array(feature_sequences).shape}。")

            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                
                try:
                    vars_to_restore = dict(self.params_as_dict(), **self.other_params_as_dict())
                    saver = tf.compat.v1.train.Saver(vars_to_restore)
                    saver.restore(sess, model_path)
                    if debug: print(f"✅ 模型从 {model_path} 加载成功。")
                except Exception as e:
                    if debug:
                        print(f"❌ 模型加载失败: {e}")
                        print("[Fallback] 因模型加载失败，使用简化分词。")
                        traceback.print_exc()
                    return self.seg_simple(sentence), [0] * len(sentence)

                features = np.array(feature_sequences, dtype=np.int32)
                feed_dict = {self.x: features, self.is_training: False}
                
                word_scores_val, A_val, init_A_val = sess.run([self.word_scores, self.A, self.init_A], feed_dict)
                if debug: print(f"模型预测完成，分数矩阵形状: {word_scores_val.shape}")
                
                tags = self.viterbi(word_scores_val.T, A_val, init_A_val)
                if debug: print(f"Viterbi解码完成，标签序列: {tags}")

                words = self.tags2words(sentence, tags)
                if debug: print(f"分词结果: {' | '.join(words)}")
                
                return words, tags
                
        except Exception as e:
            if debug:
                print(f"--- 分词过程中发生未知错误 ---")
                print(f"错误: {e}")
                traceback.print_exc()
                print("[Fallback] 因未知错误，使用简化分词。")
            return self.seg_simple(sentence), [0] * len(sentence)

    # ==================================================================
    # 以下是原本在SegBase中，但逻辑不匹配的辅助函数，现在移入SegDNN并修正
    # ==================================================================

    def sentence2index(self, sentence):
        """将句子转换为词汇表索引序列"""
        indices = []
        for char in sentence:
            indices.append(self.dictionary.get(char, 0)) # 使用 .get() 更安全
        return indices

    def index2seq(self, indices):
        """将索引序列转换为特征序列（使用正确的对称窗口）"""
        if not indices:
            return []
        # 使用 self.skip_window 来确保对称
        padded = [1] * self.skip_window + indices + [2] * self.skip_window
        sequences = []
        # 循环的边界条件也要正确
        for i in range(len(indices)):
            # 从填充后的序列中提取窗口
            start = i
            end = i + 2 * self.skip_window + 1
            sequences.append(padded[start:end])
        return np.array(sequences)

    def tags2words(self, sentence, tags):
        """将标签序列转换为分词结果"""
        words = []
        current_word = ''
        if not sentence:
            return []
        for char, tag in zip(sentence, tags):
            if tag == 0:  # S
                if current_word: words.append(current_word)
                words.append(char)
                current_word = ''
            elif tag == 1:  # B
                if current_word: words.append(current_word)
                current_word = char
            elif tag == 2:  # I
                current_word += char
            elif tag == 3:  # E
                current_word += char
                words.append(current_word)
                current_word = ''
        if current_word:
            words.append(current_word)
        return words

    def seg_simple(self, sentence):
        """修复后的简化分词方法，作为最后的安全回退"""
        return list(sentence)

    def params_as_dict(self):
        """获取核心网络参数的字典"""
        return {p.name.split(':')[0]: p for p in self.params}

    def other_params_as_dict(self):
        """获取其他参数（如embeddings, CRF矩阵）的字典"""
        return {'embeddings': self.embeddings, 'A': self.A, 'init_A': self.init_A} 