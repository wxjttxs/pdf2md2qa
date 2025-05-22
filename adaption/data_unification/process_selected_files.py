#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jsonlines as jl
import os
import random
import json
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import random
import requests
from retrying import retry
import argparse
import re
import shutil

class GPT:
    def __init__(self,model_name = 'gpt-4o') -> None:
        self.key_ind = 0
        self.init_api_keys()
        self.max_wrong_time = 5
        self.model_name = model_name
        print(f'use model of {self.model_name}')

    def init_api_keys(self):
        self.keys = []
        with open('gpt_key.txt', encoding="utf-8", mode="r") as fr:
            for l in fr:
                cols = l.split('---')
                if len(cols[0]) < 45 or len(cols[0]) > 55:
                    continue
                if len(cols) == 1:
                    cols.append('None')
                self.keys.append((cols[0],cols[1]))
        assert len(self.keys) > 0, 'have no key'
        print(f'keys: {self.keys}')
        self.wrong_time = [0]*len(self.keys)
        random.shuffle(self.keys)
    
    def get_api_key(self):
        self.key_ind =  (self.key_ind + 1) % len(self.keys)
        return self.keys[self.key_ind]

    def call(self, content, args = {}, showkeys = False):
        api_key, organization = self.get_api_key()
        if showkeys:
            print(api_key, organization)
        if organization == 'None':
            organization = ''
        url = "https://api.huatuogpt.cn/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "OpenAI-Organization": organization,
        }
        if isinstance(content,str):
            parameters = {
                "model": self.model_name,
                "messages": [{'role': 'user', 'content': content}],
                **args,
            }
        else:
            parameters = {
                "model": self.model_name,
                "messages": content,
                **args,
            }
        response = requests.post(
            url,
            headers=headers,
            json=parameters
            # verify=False
        )
        response = json.loads(response.content.decode("utf-8"))
        if 'error' in response:
            self.wrong_time[self.key_ind] += 1
            if self.wrong_time[self.key_ind] > self.max_wrong_time:
                print(response)
                print(f'del {self.keys[self.key_ind]}')
                # del self.keys[self.key_ind]
                # del self.wrong_time[self.key_ind]
            assert False, str(response)
        return response['choices'][0]['message']['content']
    
    def test(self):
        for _ in range(len(self.keys)):
            try:
                print(self.call('你好',showkeys=True))
            except Exception as e:
                print(e)
    
    @retry(wait_fixed=200, stop_max_attempt_number=20)
    def retry_call(self, content, args = {}):
        return self.call(content, args)

# 命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, default="gongjingai_results")
parser.add_argument("--num_process", type=int, default=20)
args = parser.parse_args()

# 设置提示模板和参数
query_prompt = """Please create a <question> that closely aligns with the provided <text>. Ensure that the <question> is formulated in Chinese and does not explicitly reference the text. You may incorporate specific scenarios or contexts in the <question>, allowing the <text> to serve as a comprehensive and precise answer.

<text>: {}

<question>: 

**如果text中内容跟宫颈癌无关，请回复'SKIP'**
"""

ans_prompt = """You are HuatuoGPT-II, equipped with in-depth knowledge in medicine. Your task is to directly answer the user's <question> in Chinese. In formulating your response, you must thoughtfully reference the <reference text>, ensuring that your reply does not disclose your reliance on <reference text>. Aim to provide a comprehensive and informative response, incorporating relevant insights from <reference text> to best assist the user. Please be cautious to avoid including any content that might raise ethical concerns.

<question>: {}

<reference text>: {}

**如果text中内容跟宫颈癌无关，请回复'SKIP'**

<reply>: """

# 设置其他参数
query_try_num = 2
ans_try_num = 2
q_max_length = 400
a_max_length = 600
gpt = GPT(model_name='gpt-4o')

# 要处理的特定文件
TARGET_FILES = ['黄珊.json']

# 过滤函数
def filter_str(wds, input_text):
    for wd in wds:
        if wd in input_text:
            return False
    return True

# 处理生成的问题
def get_data_query(d):
    query = d['ChatGPT_response_0']
    
    # 检查是否为跳过标记
    if query.strip() == 'SKIP':
        return False, None
    
    # 检查长度限制
    if len(query) > 180:
        return False, None
    
    return True, query

# 处理生成的回答
def get_data_ans(d):
    da = d['ChatGPT_response_0']
    
    # 检查是否为跳过标记
    if da.strip() == 'SKIP':
        return False, None
    
    # 检查是否包含不需要的关键词
    ans_key_wds = ['参考']
    if not filter_str(ans_key_wds, da):
        return False, None
    
    return True, da

# 处理函数
wrongtime = 0
def write_piece_order_data(d):
    global wrongtime
    global save_dir
    
    try:
        save_path = os.path.join(save_dir, str(d['id']) + ".json")
        if os.path.exists(save_path):
            return -1
        
        # 生成问题
        if 'query' not in d:
            for ii in range(query_try_num):
                chatgpt_query = query_prompt.format(d['text'].replace('\n','\\n')[:q_max_length])
                d['ChatGPT_query'] = chatgpt_query
                query = gpt.retry_call(chatgpt_query)
                d['ChatGPT_response_0'] = query
                
                flag, query = get_data_query(d)
                if flag:
                    d['query'] = query
                    break
                elif query is None and 'SKIP' in d['ChatGPT_response_0']:
                    print(f"跳过与宫颈癌无关的内容，ID: {d['id']}")
                    return 0

        if 'query' not in d:
            print(f"无法生成有效问题，ID: {d['id']}")
            return 0

        # 生成回答
        if 'response' not in d:
            for ii in range(ans_try_num):
                chatgpt_query = ans_prompt.format(d['query'], d['text'].replace('\n','\\n')[:a_max_length])
                d['ChatGPT_query_a'] = chatgpt_query
                ans = gpt.retry_call(chatgpt_query)
                d['ChatGPT_response_0'] = ans
                
                flag, newqa = get_data_ans(d)
                if flag:
                    d['response'] = newqa
                    break
                elif newqa is None and 'SKIP' in d['ChatGPT_response_0']:
                    print(f"跳过与宫颈癌无关的内容，ID: {d['id']}")
                    return 0
        
        if 'response' not in d:
            print(f"无法生成有效回答，ID: {d['id']}")
            return 0
            
        # 保存处理结果
        with open(save_path, mode="w", encoding="utf-8") as fw:
            json.dump(d, fw, ensure_ascii=False, indent=2)
            wrongtime = 0
            return 1

    except Exception as e:
        print(str(e), flush=True)
        wrongtime += 1
        if wrongtime > 10:
            assert 1 == 0, 'wrong'
    
    return 0

def deduplicate(data, finished):
    idset = set()
    for da in finished:
        idset.add(da['id'])

    dedup_data = []
    for da in data:
        if da['id'] not in idset:
            dedup_data.append(da)

    return dedup_data

def merge_files(save_dir):
    _, _, filenames = [i for i in os.walk(save_dir)][0]
    json_files = [f for f in filenames if f.endswith('.json')]
    res = []
    for file_path in json_files:
        try:
            with open(os.path.join(save_dir, file_path), encoding="utf-8") as f:
                da = json.loads(f.read()) 
                res.append(da)
        except Exception as e:
            print(str(e))
    return res

def process_file(file_path, file_name):
    """处理单个JSON文件"""
    global save_dir
    
    print(f"\n处理文件: {file_name}")
    
    # 加载数据
    with open(file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # 准备处理数据
    data = []
    
    # 从JSON中提取文本内容
    if isinstance(json_data, list):
        for ii, item in enumerate(json_data):
            if 'text' in item:
                data.append({'id': ii, 'text': item['text']})
    else:
        print(f"警告: {file_name} 的格式不受支持")
        return
    
    print(f"从 {file_name} 中提取了 {len(data)} 条数据")
    
    # 创建输出文件名和目录
    base_name = os.path.splitext(file_name)[0]
    task_name = f'rewrite_{base_name}'
    
    # 检查输出文件是否已存在
    output_file = f'{task_name}.json'
    if os.path.exists(output_file):
        output_file = f'{task_name}_1.json'
        task_name = f'{task_name}_1'
    
    save_dir = f'tmp_data/{task_name}'
    
    # 创建输出目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"创建输出目录: {save_dir}")
    
    # 获取已处理的数据
    finished_data = merge_files(save_dir)
    print(f'已处理数据: {len(finished_data)}')
    
    # 排除已处理的数据
    data = deduplicate(data, finished_data)
    print(f"{len(data)} 条数据待处理")
    
    # 随机打乱数据顺序
    random.shuffle(data)
    
    # 使用多线程处理数据
    if data:
        successful = 0
        with ThreadPoolExecutor(max_workers=min(args.num_process, len(data))) as executor:
            results = list(tqdm(executor.map(write_piece_order_data, data), total=len(data), desc=f"处理 {file_name}", unit="sample"))
            successful = sum(1 for r in results if r == 1)
        
        print(f'{file_name} 处理完成，成功处理 {successful} 条数据')
        
        # 合并并保存结果
        finished_data = merge_files(save_dir)
        with open(output_file, 'w', encoding='utf-8') as fw:
            json.dump(finished_data, fw, ensure_ascii=False, indent=2)
        print(f'结果已保存至 {output_file}')
    else:
        print(f"所有数据已处理完成，无需重新处理")

def main():
    """主函数"""
    # 设置输入目录
    input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.input_dir)
    
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录 {input_dir} 不存在!")
        exit(1)
    
    # 获取目录中的所有文件
    all_files = os.listdir(input_dir)
    
    # 筛选目标文件
    target_files = [f for f in all_files if f in TARGET_FILES]
    
    if not target_files:
        print(f"错误: 在 {input_dir} 中没有找到目标文件!")
        exit(1)
    
    print(f"找到 {len(target_files)} 个目标文件: {target_files}")
    
    # 处理每个目标文件
    for file_name in target_files:
        file_path = os.path.join(input_dir, file_name)
        process_file(file_path, file_name)
    
    print("所有目标文件处理完成")

if __name__ == "__main__":
    main() 