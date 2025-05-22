#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import glob
import pandas as pd
from tqdm import tqdm

def convert_xlsx_to_sharegpt(input_dir, output_file):
    """
    读取目录下所有xlsx文件，提取第一列问题和第二列回答，
    转换为ShareGPT格式，并合并到一个文件中
    
    Args:
        input_dir: 输入目录路径，包含xlsx文件
        output_file: 输出文件路径
    """
    print(f"正在扫描目录: {input_dir}")
    
    # 获取目录下所有xlsx文件
    xlsx_files = glob.glob(os.path.join(input_dir, "*.xlsx"))
    print(f"找到 {len(xlsx_files)} 个xlsx文件")
    
    # 用于存储转换后的数据
    sharegpt_data = []
    
    # 统计计数
    total_conversations = 0
    total_files = 0
    skipped_files = 0
    
    # 处理每个xlsx文件
    for xlsx_file in tqdm(xlsx_files, desc="处理xlsx文件"):
        try:
            # 读取xlsx文件
            file_name = os.path.basename(xlsx_file)
            print(f"\n处理文件: {file_name}")
            
            # 使用pandas读取Excel文件
            df = pd.read_excel(xlsx_file)
            
            # 获取表格的列数
            num_cols = len(df.columns)
            
            # 确保至少有两列
            if num_cols < 2:
                print(f"跳过文件 {file_name}，列数不足 (仅有 {num_cols} 列)")
                skipped_files += 1
                continue
            
            # 获取有效行数（丢弃空行）
            valid_rows = df.dropna(subset=[df.columns[0], df.columns[1]]).shape[0]
            print(f"有效问答对数量: {valid_rows}")
            
            # 提取问答对
            conversations = 0
            for _, row in df.iterrows():
                # 获取第一列和第二列的值
                question = row.iloc[0]
                answer = row.iloc[1]
                
                # 跳过空值
                if pd.isna(question) or pd.isna(answer) or not question or not answer:
                    continue
                
                # 转换为字符串
                question = str(question).strip()
                answer = str(answer).strip()
                
                # 创建ShareGPT格式的对话
                conversation = {
                    "conversations": [
                        {
                            "from": "human",
                            "value": question
                        },
                        {
                            "from": "gpt",
                            "value": answer
                        }
                    ],
                    "system": ""
                }
                
                # 添加到结果列表
                sharegpt_data.append(conversation)
                conversations += 1
            
            # 更新统计信息
            if conversations > 0:
                total_conversations += conversations
                total_files += 1
                print(f"已提取 {conversations} 个问答对")
            else:
                skipped_files += 1
                print(f"未从 {file_name} 中提取到有效的问答对")
                
        except Exception as e:
            print(f"处理文件 {xlsx_file} 时出错: {e}")
            skipped_files += 1
    
    # 保存转换后的数据
    if sharegpt_data:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)
        print(f"\n转换完成。共处理 {total_files} 个文件，跳过 {skipped_files} 个文件")
        print(f"提取了 {total_conversations} 个问答对，已保存到 {output_file}")
    else:
        print("未找到有效的问答对数据")

def main():
    # 设置输入和输出路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "")
    output_file = os.path.join(base_dir, "sharegpt_qa_combined.json")
    
    # 确保输入目录存在
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录 {input_dir} 不存在!")
        exit(1)
    
    # 执行转换
    convert_xlsx_to_sharegpt(input_dir, output_file)

if __name__ == "__main__":
    main() 