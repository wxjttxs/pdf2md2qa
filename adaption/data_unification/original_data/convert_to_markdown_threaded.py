#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import requests
import time
import threading
import queue
import concurrent.futures
from tqdm import tqdm
import copy
import random

# 线程安全的文件锁
file_lock = threading.Lock()

# 记录已处理的对话索引
processed_indices = set()

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
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=60  # 设置超时时间为60秒
            )
            
            response_json = response.json()
            
            if "choices" in response_json and len(response_json["choices"]) > 0:
                return response_json["choices"][0]["message"]["content"]
            else:
                print(f"API返回错误: {response_json}")
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))  # 指数退避
                    continue
                return text
                
        except Exception as e:
            print(f"API调用出错 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))  # 指数退避
            else:
                return text
    
    return text

def process_conversation(item_index, item, api_key, base_url, result_queue, stats_queue):
    """处理单个对话，将GPT回复转换为Markdown格式"""
    # 深拷贝对话，避免多线程修改同一对象
    conversation = copy.deepcopy(item)
    modified = False
    
    try:
        # 检查对话结构
        if "conversations" not in conversation or not isinstance(conversation["conversations"], list):
            result_queue.put((item_index, None, False))
            return
        
        # 查找gpt回复
        for conv_idx, conv in enumerate(conversation["conversations"]):
            if conv.get("from") == "gpt":
                # 获取原始回复
                original_text = conv.get("value", "")
                
                if not original_text.strip():
                    continue
                
                # 使用GPT-4o转换为Markdown
                markdown_text = convert_text_to_markdown(original_text, api_key, base_url)
                
                # 更新回复内容
                conversation["conversations"][conv_idx]["value"] = markdown_text
                modified = True
                stats_queue.put(("converted", 1))
    
    except Exception as e:
        print(f"处理第 {item_index+1} 个对话时出错: {e}")
        stats_queue.put(("error", 1))
        result_queue.put((item_index, None, False))
        return
    
    # 将处理结果放入队列
    result_queue.put((item_index, conversation, modified))

def save_results(output_file, data, processed_indices):
    """保存当前结果到文件"""
    with file_lock:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"已保存 {len(processed_indices)} 个处理结果到 {output_file}")

def convert_to_markdown_threaded(api_key, base_url, input_file, output_file, max_workers=5):
    """
    使用多线程将JSON文件中的GPT回复转换为Markdown格式，并即时保存结果
    
    Args:
        api_key: OpenAI API密钥
        base_url: API基础URL
        input_file: 输入文件路径
        output_file: 输出文件路径
        max_workers: 最大线程数
    """
    print(f"正在读取文件: {input_file}")
    
    # 读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_items = len(data)
    print(f"找到 {total_items} 个对话")
    
    # 检查是否存在部分处理结果
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                output_data = json.load(f)
                
            # 如果输出文件存在且有效，使用它作为基础
            if len(output_data) > 0:
                data = output_data
                # 标记已处理的对话
                for i in range(len(data)):
                    processed_indices.add(i)
                print(f"继续处理，已有 {len(processed_indices)} 个对话处理完成")
        except Exception as e:
            print(f"读取已有输出文件失败，将创建新文件: {e}")
    
    # 创建结果队列和统计队列
    result_queue = queue.Queue()
    stats_queue = queue.Queue()
    
    # 统计计数
    stats = {"converted": 0, "error": 0, "unchanged": 0}
    
    # 创建进度条
    pbar = tqdm(total=total_items - len(processed_indices), desc="转换为Markdown")
    
    # 定义更新进度条的函数
    def update_progress():
        while len(processed_indices) < total_items:
            try:
                stat_type, count = stats_queue.get(timeout=0.1)
                stats[stat_type] = stats.get(stat_type, 0) + count
                pbar.update(1)
                stats_queue.task_done()
            except queue.Empty:
                time.sleep(0.1)
            except Exception as e:
                print(f"更新进度出错: {e}")
    
    # 定义保存结果的函数
    def save_result_worker():
        save_interval = max(1, min(10, total_items // 20))  # 动态调整保存间隔
        last_save_count = 0
        
        while len(processed_indices) < total_items:
            try:
                item_index, conversation, modified = result_queue.get(timeout=0.5)
                
                if conversation is not None and modified:
                    data[item_index] = conversation
                
                processed_indices.add(item_index)
                result_queue.task_done()
                
                # 当处理的数量达到一定阈值时保存结果
                if len(processed_indices) - last_save_count >= save_interval:
                    save_results(output_file, data, processed_indices)
                    last_save_count = len(processed_indices)
            
            except queue.Empty:
                # 即使队列为空，也定期保存结果
                if len(processed_indices) > last_save_count:
                    save_results(output_file, data, processed_indices)
                    last_save_count = len(processed_indices)
                time.sleep(0.5)
            except Exception as e:
                print(f"保存结果出错: {e}")
    
    # 启动进度更新线程
    progress_thread = threading.Thread(target=update_progress, daemon=True)
    progress_thread.start()
    
    # 启动结果保存线程
    save_thread = threading.Thread(target=save_result_worker, daemon=True)
    save_thread.start()
    
    # 创建要处理的对话索引列表（排除已处理的）
    remaining_indices = [i for i in range(total_items) if i not in processed_indices]
    # 随机打乱顺序，避免同时处理相似内容
    random.shuffle(remaining_indices)
    
    # 使用线程池处理对话
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = []
        for idx in remaining_indices:
            future = executor.submit(
                process_conversation, 
                idx, data[idx], api_key, base_url, 
                result_queue, stats_queue
            )
            futures.append(future)
            # 小延迟，避免同时发送大量请求
            time.sleep(0.1)
        
        # 等待所有任务完成
        concurrent.futures.wait(futures)
    
    # 确保所有结果都被处理
    result_queue.join()
    stats_queue.join()
    
    # 最后保存一次结果
    save_results(output_file, data, processed_indices)
    
    # 关闭进度条
    pbar.close()
    
    print(f"\n转换完成。共转换 {stats['converted']} 个回复，失败 {stats['error']} 个，未修改 {stats['unchanged']} 个")
    print(f"结果已保存到 {output_file}")

def main():
    # 设置参数
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, "sharegpt_gongjingai.json")
    output_file = os.path.join(base_dir, "sharegpt_gongjingai_markdown.json")
    
    # API配置
    api_key = "sk-Dj5wmC7MBzrbdOlI7ulss2DPwR1p749S3SPHTKQavkoz7pGY"
    base_url = "https://api.huatuogpt.cn/v1"
    
    # 线程数量 - 可以根据需要调整
    max_workers = 5
    
    # 确保输入文件存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 {input_file} 不存在!")
        exit(1)
    
    # 执行转换
    convert_to_markdown_threaded(api_key, base_url, input_file, output_file, max_workers)

if __name__ == "__main__":
    main() 