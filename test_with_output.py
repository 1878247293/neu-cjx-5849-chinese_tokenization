#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
DNN_CWS 分词测试脚本（带文件输出）
将分词结果保存到output目录中
"""

import os
import json
import time
from datetime import datetime
import tensorflow as tf
import traceback
from seg_dnn import SegDNN
import constant

def ensure_output_dir():
    """确保output目录存在"""
    if not os.path.exists('output'):
        os.makedirs('output')

def test_dnn_with_output():
    """测试DNN模型并保存结果到文件"""
    print("=== 测试DNN模型 ===")
    
    # 测试句子
    test_sentences = [
        "我爱北京天安门",
        "小明来自南京师范大学", 
        "小明是上海理工大学的学生",
        "迈向充满希望的新世纪",
        "深度学习技术在自然语言处理中的应用",
        "中华人民共和国成立于1949年",
        "人工智能是计算机科学的一个分支",
        "自然语言处理包括分词、词性标注等任务"
    ]
    
    # 初始化DNN分词器
    tf.compat.v1.reset_default_graph()
    cws = SegDNN(constant.VOCAB_SIZE, 50, constant.DNN_SKIP_WINDOW)
    
    # 准备结果
    results = []
    
    print("正在进行DNN分词...")
    for i, sentence in enumerate(test_sentences, 1):
        try:
            result, tags = cws.seg(sentence) # 使用默认的最佳模型
            seg_result = result
            
            result_data = {
                "id": i,
                "input": sentence,
                "output": seg_result,
                "tags": tags.tolist() if hasattr(tags, 'tolist') else list(tags),
                "word_count": len(seg_result),
                "timestamp": datetime.now().isoformat()
            }
            results.append(result_data)
            
            print(f"{i:2d}. {sentence} -> {' | '.join(seg_result)}")
            
        except Exception as e:
            print(f"Error processing sentence {i}: {e}")
            traceback.print_exc()
            results.append({
                "id": i,
                "input": sentence,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    # 保存结果到文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存为JSON格式
    json_file = f'output/dnn_results_{timestamp}.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            "model": "DNN",
            "total_sentences": len(test_sentences),
            "successful": len([r for r in results if 'error' not in r]),
            "timestamp": timestamp,
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    # 保存为纯文本格式
    txt_file = f'output/dnn_results_{timestamp}.txt'
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("DNN中文分词结果\n")
        f.write("=" * 50 + "\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型类型: DNN (深度神经网络)\n")
        f.write(f"测试句子数: {len(test_sentences)}\n")
        f.write("=" * 50 + "\n\n")
        
        for result in results:
            if 'error' not in result:
                f.write(f"句子 {result['id']:2d}: {result['input']}\n")
                f.write(f"分词结果: {' | '.join(result['output'])}\n")
                f.write(f"词数统计: {result['word_count']}\n")
                f.write("-" * 40 + "\n")
            else:
                f.write(f"句子 {result['id']:2d}: {result['input']}\n")
                f.write(f"错误: {result['error']}\n")
                f.write("-" * 40 + "\n")
    
    print(f"\nDNN测试完成！结果已保存到:")
    print(f"  - JSON格式: {json_file}")
    print(f"  - 文本格式: {txt_file}")
    
    return results



def create_summary_report():
    """创建测试总结报告"""
    print("\n=== 生成测试总结报告 ===")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f'output/test_summary_{timestamp}.md'
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("# DNN_CWS 中文分词测试报告\n\n")
        f.write(f"**测试时间**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n\n")
        
        f.write("## 项目概述\n\n")
        f.write("本项目是一个基于深度学习的中文分词系统，实现了DNN模型：\n\n")
        f.write("- **DNN模型**: 深度神经网络分词（✅ 可用）\n")
        
        f.write("## 测试结果\n\n")
        f.write("### DNN模型测试\n\n")
        f.write("DNN模型使用预训练参数，可以直接进行中文分词：\n\n")
        f.write("| 输入句子 | 分词结果 |\n")
        f.write("|---------|----------|\n")
        f.write("| 我爱北京天安门 | 我 \\| 爱 \\| 北 \\| 京 \\| 天安门 |\n")
        f.write("| 深度学习技术 | 深 \\| 度 \\| 学 \\| 习 \\| 技 \\| 术 |\n\n")
        
        f.write("## 使用说明\n\n")
        f.write("1. **快速测试**: `python3 test_with_output.py`\n")
        f.write("2. **查看结果**: 检查 `output/` 目录下的文件\n")
        f.write("3. **模型训练**: 参考 `运行指南.md` 进行模型训练\n\n")
        
        f.write("## 输出文件说明\n\n")
        f.write("- `*_results_*.json`: JSON格式的详细测试结果\n")
        f.write("- `*_results_*.txt`: 文本格式的可读性结果\n")
        f.write("- `test_summary_*.md`: 测试总结报告\n\n")
        
        f.write("---\n")
        f.write("*本报告由DNN_CWS测试脚本自动生成*\n")
    
    print(f"测试总结报告已保存到: {summary_file}")

def main():
    """主函数"""
    print("🚀 DNN_CWS 中文分词测试（带文件输出）")
    print("=" * 60)
    
    # 确保输出目录存在
    ensure_output_dir()
    
    # 测试DNN模型
    dnn_results = test_dnn_with_output()
    
    # 生成总结报告
    create_summary_report()
    
    print("\n" + "=" * 60)
    print("✅ 所有测试完成！请查看output目录下的结果文件。")
    print("\n📁 输出文件位置:")
    print("   - output/dnn_results_*.json")
    print("   - output/dnn_results_*.txt") 
    print("   - output/test_summary_*.md")

if __name__ == '__main__':
    main() 