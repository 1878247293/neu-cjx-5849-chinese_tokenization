#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
DNN_CWS 自定义分词工具
允许用户输入文本进行分词，并将结果保存到output目录
"""

import os
import json
from datetime import datetime
import tensorflow as tf
import traceback
from seg_dnn import SegDNN
import constant

def ensure_output_dir():
    """确保output目录存在"""
    if not os.path.exists('output'):
        os.makedirs('output')

def seg_text_to_file(text, output_filename=None):
    """对输入文本进行分词并保存到文件"""
    
    if not text.strip():
        print("错误：输入文本不能为空")
        return False
    
    try:
        # 初始化DNN分词器
        print("正在初始化DNN分词器...")
        tf.compat.v1.reset_default_graph()
        cws = SegDNN(constant.VOCAB_SIZE, 50, constant.DNN_SKIP_WINDOW)
        
        # 进行分词
        print(f"正在分词: {text}")
        result, tags = cws.seg(text) # 使用默认的最佳模型
        seg_result = result
        
        # 准备结果数据
        result_data = {
            "input": text,
            "output": seg_result,
            "tags": tags.tolist() if hasattr(tags, 'tolist') else list(tags),
            "word_count": len(seg_result),
            "char_count": len(text),
            "timestamp": datetime.now().isoformat(),
            "model": "DNN"
        }
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_filename:
            base_name = output_filename.replace('.txt', '').replace('.json', '')
        else:
            # 使用文本的前10个字符作为文件名
            safe_text = ''.join(c for c in text[:10] if c.isalnum() or c in '._-')
            if not safe_text:
                safe_text = "custom_seg"
            base_name = safe_text
        
        # 保存为JSON格式
        json_file = f'output/{base_name}_{timestamp}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        # 保存为文本格式
        txt_file = f'output/{base_name}_{timestamp}.txt'
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("DNN中文分词结果\n")
            f.write("=" * 50 + "\n")
            f.write(f"分词时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"模型类型: DNN (深度神经网络)\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"原文: {text}\n")
            f.write(f"分词结果: {' | '.join(seg_result)}\n")
            f.write(f"词数统计: {len(seg_result)}\n")
            f.write(f"字符数: {len(text)}\n")
            f.write("-" * 50 + "\n")
            
            f.write("\n详细词汇列表:\n")
            for i, word in enumerate(seg_result, 1):
                f.write(f"{i:2d}. {word}\n")
        
        # 显示结果
        print(f"\n✅ 分词完成！")
        print(f"原文: {text}")
        print(f"分词结果: {' | '.join(seg_result)}")
        print(f"词数: {len(seg_result)}")
        print(f"\n📁 结果已保存到:")
        print(f"  - JSON格式: {json_file}")
        print(f"  - 文本格式: {txt_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ 分词过程中出现错误: {e}")
        traceback.print_exc()
        return False

def batch_seg_from_file(input_file):
    """从文件中读取多行文本进行批量分词"""
    
    if not os.path.exists(input_file):
        print(f"错误：文件 {input_file} 不存在")
        return False
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        if not lines:
            print("错误：文件中没有有效内容")
            return False
        
        print(f"从文件 {input_file} 读取到 {len(lines)} 行文本")
        
        # 初始化分词器
        print("正在初始化DNN分词器...")
        tf.compat.v1.reset_default_graph()
        cws = SegDNN(constant.VOCAB_SIZE, 50, constant.DNN_SKIP_WINDOW)
        
        # 批量分词
        results = []
        for i, text in enumerate(lines, 1):
            try:
                print(f"正在处理第 {i}/{len(lines)} 行: {text[:20]}...")
                seg_result, tags = cws.seg(text) # 使用默认的最佳模型
                
                result_data = {
                    "id": i,
                    "input": text,
                    "output": seg_result,
                    "tags": tags.tolist() if hasattr(tags, 'tolist') else list(tags),
                    "word_count": len(seg_result),
                    "char_count": len(text)
                }
                results.append(result_data)
                
            except Exception as e:
                print(f"处理第 {i} 行时出错: {e}")
                results.append({
                    "id": i,
                    "input": text,
                    "error": str(e)
                })
        
        # 保存批量结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_file = f'output/batch_seg_{timestamp}.json'
        
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump({
                "source_file": input_file,
                "total_lines": len(lines),
                "successful": len([r for r in results if 'error' not in r]),
                "timestamp": timestamp,
                "results": results
            }, f, ensure_ascii=False, indent=2)
        
        # 保存为文本格式
        batch_txt_file = f'output/batch_seg_{timestamp}.txt'
        with open(batch_txt_file, 'w', encoding='utf-8') as f:
            f.write("DNN中文分词批量处理结果\n")
            f.write("=" * 60 + "\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"源文件: {input_file}\n")
            f.write(f"总行数: {len(lines)}\n")
            f.write(f"成功处理: {len([r for r in results if 'error' not in r])}\n")
            f.write("=" * 60 + "\n\n")
            
            for result in results:
                if 'error' not in result:
                    f.write(f"第 {result['id']:2d} 行: {result['input']}\n")
                    f.write(f"分词结果: {' | '.join(result['output'])}\n")
                    f.write(f"词数: {result['word_count']}\n")
                    f.write("-" * 40 + "\n")
                else:
                    f.write(f"第 {result['id']:2d} 行: {result['input']}\n")
                    f.write(f"错误: {result['error']}\n")
                    f.write("-" * 40 + "\n")
        
        print(f"\n✅ 批量分词完成！")
        print(f"📁 结果已保存到:")
        print(f"  - JSON格式: {batch_file}")
        print(f"  - 文本格式: {batch_txt_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ 批量分词过程中出现错误: {e}")
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🚀 DNN_CWS 自定义分词工具")
    print("=" * 50)
    
    # 确保输出目录存在
    ensure_output_dir()
    
    while True:
        print("\n请选择操作:")
        print("1. 输入文本进行分词")
        print("2. 从文件批量分词")
        print("3. 退出")
        
        choice = input("\n请输入选择 (1-3): ").strip()
        
        if choice == '1':
            text = input("\n请输入要分词的文本: ").strip()
            if text:
                filename = input("输入保存文件名 (可选，按回车使用默认名): ").strip()
                seg_text_to_file(text, filename if filename else None)
            else:
                print("输入文本不能为空！")
                
        elif choice == '2':
            filename = input("\n请输入文本文件路径: ").strip()
            if filename:
                batch_seg_from_file(filename)
            else:
                print("文件路径不能为空！")
                
        elif choice == '3':
            print("👋 谢谢使用！")
            break
            
        else:
            print("❌ 无效选择，请重新输入！")

if __name__ == '__main__':
    main() 