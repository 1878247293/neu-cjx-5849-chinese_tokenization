#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
DNN_CWS 模型训练脚本
使用init.py生成的训练数据来训练DNN模型
"""

import os
import time
from datetime import datetime
import numpy as np
import traceback

import constant
from seg_dnn import SegDNN


def ensure_output_dir():
    """确保输出目录存在"""
    if not os.path.exists('output'):
        os.makedirs('output')
    if not os.path.exists('model'):
        os.makedirs('model')
    if not os.path.exists('tmp'):
        os.makedirs('tmp')

def check_training_data():
    """检查训练和验证数据文件是否存在"""
    required_files = {
        '训练集': [
            'corpus/dnn/words_batch_flat.npy',
            'corpus/dnn/labels_batch_flat.npy'
        ],
        '验证集': [
            'corpus/dnn/words_batch_flat_val.npy',
            'corpus/dnn/labels_batch_flat_val.npy'
        ],
        '词典': ['corpus/dict.utf8']
    }
    
    print("🔍 检查数据文件:")
    all_found = True
    
    for set_name, files in required_files.items():
        print(f"  - {set_name}:")
        for file_path in files:
            if os.path.exists(file_path):
                size_mb = os.path.getsize(file_path) / 1024 / 1024
                print(f"    ✅ {file_path} ({size_mb:.1f}MB)")
            else:
                print(f"    ❌ {file_path}")
                all_found = False
    
    if not all_found:
        print(f"\n⚠️  缺少数据文件!")
        print("请先运行: python3 init.py")
        return False
    
    print("✅ 所有数据文件就绪!")
    return True

def train_dnn_model(epochs=10):
    """训练优化版DNN模型"""
    print("\n🚀 开始训练DNN模型...")
    print("=" * 50)
    
    try:
        from transform_data_dnn import TransformDataDNN
        import tensorflow as tf
        tf.compat.v1.reset_default_graph()

        print("正在加载数据...")
        train_loader = TransformDataDNN(constant.DNN_SKIP_WINDOW, dataset_type='training')
        val_loader = TransformDataDNN(constant.DNN_SKIP_WINDOW, dataset_type='validation')

        train_data = (train_loader.whole_words_batch, train_loader.whole_labels_batch)
        validation_data = (val_loader.whole_words_batch, val_loader.whole_labels_batch)

        print("正在初始化DNN模型...")
        # 注意：这里的embed_size需要和seg_dnn.py中一致
        dnn_model = SegDNN(constant.VOCAB_SIZE, 100, constant.DNN_SKIP_WINDOW)
        
        print(f"模型参数:")
        print(f"  - 词汇表大小: {dnn_model.vocab_size}")
        print(f"  - 嵌入维度: {dnn_model.embed_size}")
        print(f"  - 网络结构: {dnn_model.h1} -> {dnn_model.h2} -> {dnn_model.tags_count}")
        print(f"  - 批大小: {dnn_model.batch_size}")
        print(f"  - 训练轮数: {epochs}")
        
        start_time = time.time()
        dnn_model.train_optimized(
            train_data=train_data, 
            validation_data=validation_data, 
            epochs=epochs, 
            early_stopping_patience=3
        )
        training_time = time.time() - start_time
        
        print(f"\n✅ DNN模型训练完成!")
        print(f"总训练时间: {training_time:.1f}秒")
        
        log_path = f'output/dnn_training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"DNN模型训练日志\n"
                    f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"总耗时: {training_time:.1f}秒\n")
        print(f"训练日志保存到: {log_path}")
        return True
        
    except Exception as e:
        print(f"❌ DNN模型训练失败: {e}")
        traceback.print_exc()
        return False

def test_trained_models():
    """测试训练完成的模型"""
    print("\n🧪 测试训练完成的模型...")
    print("=" * 50)
    
    test_sentences = [
        "我爱北京天安门",
        "深度学习技术发展迅速",
        "自然语言处理很有趣"
    ]
    
    if not os.path.exists('model/best_model.ckpt.index'):
        print("  🤷 没有找到训练好的模型 (model/best_model.ckpt)，跳过测试")
        return

    print("测试DNN模型 (model/best_model.ckpt):")
    try:
        import tensorflow as tf
        tf.compat.v1.reset_default_graph()
        
        # 注意：这里的embed_size需要和seg_dnn.py中一致
        cws = SegDNN(constant.VOCAB_SIZE, 100, constant.DNN_SKIP_WINDOW)
        
        for sentence in test_sentences:
            print("-" * 40)
            result, _ = cws.seg(sentence, debug=True)
            print(f"  Input:  {sentence}")
            print(f"  Output: {' | '.join(result)}")
            print("-" * 40)
            
    except Exception as e:
        print(f"  DNN测试失败: {e}")
        traceback.print_exc()

def delete_all_models():
    """一键删除所有历史模型"""
    print("\n🗑️  一键删除所有历史模型...")
    print("=" * 50)
    
    model_patterns = ['model/*.ckpt*', 'tmp/*.ckpt*']
    all_files = []
    
    import glob
    for pattern in model_patterns:
        all_files.extend(glob.glob(pattern))
    
    checkpoint_files = ['model/checkpoint', 'tmp/checkpoint']
    for cp_file in checkpoint_files:
        if os.path.exists(cp_file):
            all_files.append(cp_file)
            
    if not all_files:
        print("✅ 没有找到历史模型文件")
        return
    
    total_size = sum(os.path.getsize(f) for f in all_files)
    print("🔍 发现以下模型文件将被删除:")
    for f in all_files:
        print(f"  - {f} ({os.path.getsize(f)/1024/1024:.2f}MB)")
    
    print(f"\n📊 统计: {len(all_files)} 个文件, 总大小: {total_size/1024/1024:.2f}MB")
    
    try:
        confirm = input(f"\n⚠️  确认删除这 {len(all_files)} 个文件? (y/N): ").strip().lower()
    except EOFError: # Non-interactive mode
        confirm = 'y'

    if confirm in ['y', 'yes']:
        deleted_count = 0
        print("\n🗑️  正在删除...")
        for file_path in all_files:
            try:
                os.remove(file_path)
                deleted_count += 1
            except Exception: pass
        print(f"\n🎉 {deleted_count} 个文件删除完成!")
    else:
        print("❌ 取消删除操作")

def main():
    """主函数"""
    print("🎯 DNN_CWS 模型训练与管理脚本")
    print("=" * 60)
    
    ensure_output_dir()
    
    if not check_training_data():
        return
    
    while True:
        try:
            print("\n请选择操作:")
            print("1. 训练DNN模型")
            print("2. 测试已有模型")
            print("3. 清理所有历史模型")
            print("4. 退出")
            
            choice = input("\n请输入选择 (1-4): ").strip()
            
            if choice == '1':
                epochs_str = input("请输入训练轮数 (建议5-15, 默认10): ").strip()
                epochs = int(epochs_str) if epochs_str.isdigit() else 10
                if train_dnn_model(epochs):
                    test_trained_models()
            elif choice == '2':
                test_trained_models()
            elif choice == '3':
                delete_all_models()
            elif choice == '4':
                print("退出脚本。")
                break
            else:
                print("❌ 无效选择，请重新输入!")
        except EOFError:
            break
    
    print("\n" + "=" * 60)
    print("✅ 任务完成!")

if __name__ == '__main__':
    main()
