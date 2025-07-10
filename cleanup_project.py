#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
DNN_CWS 项目清理脚本
删除临时文件、重复文件和不必要的文件，整理项目结构
"""

import os
import shutil
import glob
from datetime import datetime

class ProjectCleaner:
    def __init__(self):
        self.deleted_files = []
        self.deleted_dirs = []
        self.kept_files = []
        
    def delete_file_safe(self, file_path, reason=""):
        """安全删除文件"""
        try:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                os.remove(file_path)
                self.deleted_files.append((file_path, size, reason))
                print(f"✅ 删除文件: {file_path} ({size/1024:.1f}KB) - {reason}")
                return True
        except Exception as e:
            print(f"❌ 删除失败: {file_path} - {e}")
        return False
    
    def delete_dir_safe(self, dir_path, reason=""):
        """安全删除目录"""
        try:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                self.deleted_dirs.append((dir_path, reason))
                print(f"✅ 删除目录: {dir_path} - {reason}")
                return True
        except Exception as e:
            print(f"❌ 删除目录失败: {dir_path} - {e}")
        return False
    
    def clean_temporary_files(self):
        """清理临时文件"""
        print("\n🧹 清理临时文件...")
        print("=" * 50)
        
        # 1. 删除Python缓存
        cache_patterns = [
            '__pycache__',
            '*.pyc',
            '*.pyo', 
            '.pytest_cache'
        ]
        
        for pattern in cache_patterns:
            if pattern.startswith('.') or not pattern.startswith('*'):
                # 目录
                for cache_dir in glob.glob(pattern):
                    self.delete_dir_safe(cache_dir, "Python缓存")
            else:
                # 文件模式
                for cache_file in glob.glob(pattern):
                    self.delete_file_safe(cache_file, "Python缓存")
        
        # 2. 删除测试临时文件
        test_files = [
            'tt1.py',  # 临时测试文件
            'test.txt',  # 临时测试文件
        ]
        
        for test_file in test_files:
            self.delete_file_safe(test_file, "临时测试文件")
    
    def clean_improvement_scripts(self):
        """清理改进过程中产生的脚本"""
        print("\n📝 清理改进脚本...")
        print("=" * 50)
        
        # 改进过程中创建的脚本（保留指南文档）
        improvement_scripts = [
            'expand_corpus.py',
            'improve_dictionary.py', 
            'improve_labeling.py',
            'comprehensive_improvement.py',
            'fix_data_issue.py',
            'quick_fix.py',
            'enhanced_train.py'
        ]
        
        for script in improvement_scripts:
            self.delete_file_safe(script, "改进脚本（已完成任务）")
    
    def clean_duplicate_corpus_files(self):
        """清理重复的语料库文件"""
        print("\n📚 清理重复语料库文件...")
        print("=" * 50)
        
        corpus_files_to_remove = [
            'corpus/expanded_training.utf8',  # 已合并到pku_training.utf8
            'corpus/pku_training.utf8.backup',  # 备份文件（有backup目录）
            'corpus/enhanced_dict.utf8',  # 增强词典（已应用到dict.utf8）
            'corpus/pku_training_dict.txt',  # 重复的词典文件
        ]
        
        for corpus_file in corpus_files_to_remove:
            self.delete_file_safe(corpus_file, "重复的语料库文件")
        
        # 删除expanded目录（临时生成的数据）
        self.delete_dir_safe('corpus/expanded', "临时扩展数据")
    
    def clean_output_files(self):
        """清理旧的输出文件"""
        print("\n📊 清理旧的输出文件...")
        print("=" * 50)
        
        # 保留最新的2个输出文件，删除其他的
        output_files = glob.glob('output/batch_seg_*.txt') + glob.glob('output/batch_seg_*.json')
        output_files.sort(key=os.path.getmtime, reverse=True)
        
        # 保留最新的4个文件（2个txt + 2个json）
        files_to_keep = output_files[:4]
        files_to_delete = output_files[4:]
        
        for file_path in files_to_delete:
            self.delete_file_safe(file_path, "旧的输出文件")
        
        # 删除旧的训练日志
        log_files = glob.glob('output/dnn_training_log_*.txt')
        if len(log_files) > 1:
            log_files.sort(key=os.path.getmtime, reverse=True)
            for old_log in log_files[1:]:
                self.delete_file_safe(old_log, "旧的训练日志")
    
    def clean_unused_scripts(self):
        """清理未使用的脚本"""
        print("\n🔧 清理未使用的脚本...")
        print("=" * 50)
        
        unused_scripts = [
            'transform_data_w2v.py',  # Word2Vec相关（项目使用DNN）
            'word2vec.py',  # Word2Vec相关
            'test_with_output.py',  # 重复功能（与custom_seg.py类似）
        ]
        
        for script in unused_scripts:
            self.delete_file_safe(script, "未使用的脚本")
    
    def clean_empty_directories(self):
        """清理空目录"""
        print("\n📁 清理空目录...")
        print("=" * 50)
        
        dirs_to_check = ['tmp', 'logs', 'data']
        
        for dir_name in dirs_to_check:
            if os.path.exists(dir_name):
                try:
                    if not os.listdir(dir_name):
                        self.delete_dir_safe(dir_name, "空目录")
                    else:
                        print(f"⚠️  保留非空目录: {dir_name}")
                except OSError:
                    print(f"⚠️  无法检查目录: {dir_name}")
    
    def organize_corpus_files(self):
        """整理语料库文件"""
        print("\n📋 整理语料库文件...")
        print("=" * 50)
        
        # 移动备份文件到backup目录
        corpus_patches = [
            'corpus/fixed_phrases.json',
            'corpus/labeling_rules.py', 
            'corpus/compound_processor_patch.py',
            'corpus/compound_words.json'
        ]
        
        # 创建patches子目录
        patches_dir = 'corpus/patches'
        os.makedirs(patches_dir, exist_ok=True)
        
        for patch_file in corpus_patches:
            if os.path.exists(patch_file):
                new_path = os.path.join(patches_dir, os.path.basename(patch_file))
                try:
                    shutil.move(patch_file, new_path)
                    print(f"✅ 移动: {patch_file} -> {new_path}")
                except Exception as e:
                    print(f"❌ 移动失败: {patch_file} - {e}")
    
    def create_project_structure_doc(self):
        """创建项目结构文档"""
        print("\n📄 创建项目结构文档...")
        
        structure_content = f"""# DNN_CWS 项目结构说明

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
*清理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open('PROJECT_STRUCTURE.md', 'w', encoding='utf-8') as f:
            f.write(structure_content)
        
        print("✅ 创建项目结构文档: PROJECT_STRUCTURE.md")
    
    def generate_summary(self):
        """生成清理总结"""
        print("\n" + "=" * 60)
        print("🎉 项目清理完成!")
        print("=" * 60)
        
        total_deleted_size = sum(size for _, size, _ in self.deleted_files)
        
        print(f"📊 清理统计:")
        print(f"  - 删除文件: {len(self.deleted_files)} 个")
        print(f"  - 删除目录: {len(self.deleted_dirs)} 个") 
        print(f"  - 释放空间: {total_deleted_size/1024/1024:.1f}MB")
        
        if self.deleted_files:
            print(f"\n📝 删除的文件:")
            for file_path, size, reason in self.deleted_files:
                print(f"  - {file_path} ({size/1024:.1f}KB) - {reason}")
        
        if self.deleted_dirs:
            print(f"\n📁 删除的目录:")
            for dir_path, reason in self.deleted_dirs:
                print(f"  - {dir_path} - {reason}")
        
        print(f"\n✨ 项目结构已优化，核心功能文件已保留")
    
    def run(self):
        """执行完整的清理流程"""
        print("🧹 DNN_CWS 项目清理工具")
        print("=" * 60)
        
        try:
            confirmation = input("确认要清理项目吗？这将删除临时文件和重复文件 (y/N): ").strip().lower()
        except EOFError:
            confirmation = 'y'  # 非交互模式
        
        if confirmation not in ['y', 'yes']:
            print("❌ 取消清理操作")
            return
        
        # 执行清理步骤
        self.clean_temporary_files()
        self.clean_improvement_scripts()
        self.clean_duplicate_corpus_files()
        self.clean_output_files()
        self.clean_unused_scripts()
        self.organize_corpus_files()
        self.clean_empty_directories()
        self.create_project_structure_doc()
        
        # 生成总结
        self.generate_summary()

def main():
    cleaner = ProjectCleaner()
    cleaner.run()

if __name__ == '__main__':
    main() 