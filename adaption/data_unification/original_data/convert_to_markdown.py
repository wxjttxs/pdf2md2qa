#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import requests
from tqdm import tqdm
import time

def convert_to_markdown(api_key, base_url, input_file, output_file):
    """
    读取ShareGPT格式的JSON文件，使用GPT-4o将gpt角色的回复转换为Markdown格式
    
    Args:
        api_key: OpenAI API密钥
        base_url: API基础URL
        input_file: 输入文件路径
        output_file: 输出文件路径
    """
    print(f"正在读取文件: {input_file}")
    
    # 读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"找到 {len(data)} 个对话")
    
    # 转换计数
    converted_count = 0
    error_count = 0
    
    # 遍历所有对话
    for idx, item in enumerate(tqdm(data, desc="转换为Markdown")):
        try:
            # 检查对话结构
            if "conversations" not in item or not isinstance(item["conversations"], list):
                continue
            
            # 查找gpt回复
            for conv_idx, conv in enumerate(item["conversations"]):
                if conv.get("from") == "gpt":
                    # 获取原始回复
                    original_text = conv.get("value", "")
                    
                    if not original_text.strip():
                        continue
                    
                    # 使用GPT-4o转换为Markdown
                    markdown_text = convert_text_to_markdown(original_text, api_key, base_url)
                    
                    # 更新回复内容
                    item["conversations"][conv_idx]["value"] = markdown_text
                    converted_count += 1
                    
                    # 每处理10个请求暂停一下，避免触发API限制
                    if converted_count % 10 == 0:
                        time.sleep(1)
        
        except Exception as e:
            print(f"处理第 {idx+1} 个对话时出错: {e}")
            error_count += 1
    
    # 保存修改后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n转换完成。共转换 {converted_count} 个回复，失败 {error_count} 个")
    print(f"结果已保存到 {output_file}")


def convert_text_to_markdown(text, api_key, base_url):
    """
    使用GPT-4o将文本转换为Markdown格式
    
    Args:
        text: 要转换的文本
        api_key: OpenAI API密钥
        base_url: API基础URL
    
    Returns:
        转换后的Markdown文本
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    prompt = f"""请将以下医学回答转换为Markdown格式，同时确保保留所有原始内容和医学信息。
使用适当的Markdown语法来增强可读性，例如：
- 对标题使用#、##等
- 对列表使用-或*
- 对重点内容使用**加粗**
- 对专业术语使用*斜体*
- 根据需要添加适当的分隔线、表格等

原始回答:
{text}

Markdown格式回答:"""
    
    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }
    
    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=data
        )
        
        response_json = response.json()
        
        if "choices" in response_json and len(response_json["choices"]) > 0:
            return response_json["choices"][0]["message"]["content"]
        else:
            print(f"API返回错误: {response_json}")
            return text
            
    except Exception as e:
        print(f"API调用出错: {e}")
        return text


def main():
    # 设置参数
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, "sharegpt_gongjingai.json")
    output_file = os.path.join(base_dir, "sharegpt_gongjingai_markdown.json")
    
    # API配置
    api_key = "sk-Dj5wmC7MBzrbdOlI7ulss2DPwR1p749S3SPHTKQavkoz7pGY"
    base_url = "https://api.huatuogpt.cn/v1"
    
    # 确保输入文件存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 {input_file} 不存在!")
        exit(1)
    
    # 执行转换
    convert_to_markdown(api_key, base_url, input_file, output_file)

if __name__ == "__main__":
    main() 