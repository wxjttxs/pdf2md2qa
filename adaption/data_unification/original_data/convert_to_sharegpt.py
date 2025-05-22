#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import glob
from tqdm import tqdm

def convert_to_sharegpt_format(input_dir, output_file):
    """
    读取目录下所有JSON文件中的query和response字段，
    转换为OpenAI的ShareGPT格式，并合并到一个文件中
    
    Args:
        input_dir: 输入目录路径，包含JSON文件
        output_file: 输出文件路径
    """
    print(f"正在扫描目录: {input_dir}")
    
    # 获取目录下所有JSON文件
    json_files = glob.glob(os.path.join(input_dir, "**/*.json"), recursive=True)
    print(f"找到 {len(json_files)} 个JSON文件")
    
    # 用于存储转换后的数据
    sharegpt_data = []
    
    # 统计计数
    total_conversations = 0
    total_files = 0
    skipped_files = 0
    
    # 处理每个JSON文件
    for json_file in tqdm(json_files, desc="处理JSON文件"):
        try:
            # 读取JSON文件
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 处理单个文件或文件列表
            items = []
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = [data]
            
            # 统计有效对话数量
            valid_conversations = 0
            
            # 处理每个条目
            for item in items:
                # 检查是否包含query和response字段
                if 'query' in item and 'response' in item:
                    # 创建ShareGPT格式的对话
                    conversation = {
                        "conversations": [
                            {
                                "from": "human",
                                "value": item['query']
                            },
                            {
                                "from": "gpt",
                                "value": item['response']
                            }
                        ],
                        "system": ""
                    }
                    
                    # 添加到结果列表
                    sharegpt_data.append(conversation)
                    valid_conversations += 1
            
            # 更新统计信息
            if valid_conversations > 0:
                total_conversations += valid_conversations
                total_files += 1
            else:
                skipped_files += 1
                print(f"跳过文件 {json_file}，未找到有效的query和response对")
                
        except Exception as e:
            print(f"处理文件 {json_file} 时出错: {e}")
            skipped_files += 1
    
    # 保存转换后的数据
    if sharegpt_data:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)
        print(f"\n转换完成。共处理 {total_files} 个文件，跳过 {skipped_files} 个文件")
        print(f"提取了 {total_conversations} 个对话，已保存到 {output_file}")
    else:
        print("未找到有效的对话数据")

def main():
    # 设置输入和输出路径
    input_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(input_dir, "sharegpt_combined.json")
    
    # 执行转换
    convert_to_sharegpt_format(input_dir, output_file)

if __name__ == "__main__":
    main() 