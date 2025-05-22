#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import json
import glob
from tqdm import tqdm
import shutil
import argparse

# 引入MinerU的相关包
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod

# 定义文件和目录路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, '宫颈癌')
OUTPUT_DIR = os.path.join(BASE_DIR, 'gongjingai_results')
MARKDOWN_DIR = os.path.join(BASE_DIR, 'markdown_files')

# 定义分隔句子的正则表达式（中英文句号、问号、叹号）
SENTENCE_DELIMITER = r'(?<=[。！？.!?])'

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='处理PDF文件并提取内容为chunks')
    parser.add_argument('--subfolder', type=str, default=None, 
                      help='指定要处理的子文件夹名称，如"黄珊"，默认处理所有子文件夹')
    parser.add_argument('--force', action='store_true', 
                      help='强制重新处理所有文件，忽略已处理的文件')
    parser.add_argument('--clean', action='store_true', 
                      help='清空输出目录，重新开始处理')
    return parser.parse_args()

def ensure_dir(directory):
    """确保目录存在，不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_pdf_to_markdown(pdf_path, output_dir):
    """使用MinerU的Python API将PDF转换为Markdown"""
    try:
        # 提取文件名（不包含路径和扩展名）
        file_name = os.path.basename(pdf_path).rsplit('.', 1)[0]
        
        # 创建输出目录
        pdf_output_dir = os.path.join(output_dir, file_name)
        if os.path.exists(pdf_output_dir):
            shutil.rmtree(pdf_output_dir)
        
        os.makedirs(pdf_output_dir, exist_ok=True)
        
        # 创建图片目录
        image_dir = os.path.join(pdf_output_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        
        # 设置输出路径
        image_writer = FileBasedDataWriter(image_dir)
        md_writer = FileBasedDataWriter(pdf_output_dir)
        
        # 读取PDF文件
        reader = FileBasedDataReader("")
        pdf_bytes = reader.read(pdf_path)
        
        # 创建数据集实例
        ds = PymuDocDataset(pdf_bytes)
        
        # 根据PDF类型进行处理
        if ds.classify() == SupportedPdfParseMethod.OCR:
            # OCR模式处理
            print(f"使用OCR模式处理: {file_name}")
            infer_result = ds.apply(doc_analyze, ocr=True)
            pipe_result = infer_result.pipe_ocr_mode(image_writer)
        else:
            # 文本模式处理
            print(f"使用文本模式处理: {file_name}")
            infer_result = ds.apply(doc_analyze, ocr=False)
            pipe_result = infer_result.pipe_txt_mode(image_writer)
        
        # 导出Markdown文件
        md_path = os.path.join(pdf_output_dir, f"{file_name}.md")
        pipe_result.dump_md(md_writer, f"{file_name}.md", "images")
        
        if os.path.exists(md_path):
            return md_path
        else:
            # 尝试查找可能的Markdown文件
            md_files = glob.glob(os.path.join(pdf_output_dir, "*.md"))
            if md_files:
                return md_files[0]
            else:
                print(f"警告: 未在 {pdf_output_dir} 中找到Markdown文件")
                print(f"目录内容: {os.listdir(pdf_output_dir) if os.path.exists(pdf_output_dir) else '目录不存在'}")
                return None
    except Exception as e:
        print(f"Error extracting PDF {pdf_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def read_markdown_file(md_file):
    """读取Markdown文件内容"""
    try:
        with open(md_file, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading markdown file {md_file}: {e}")
        return ""

def split_text_into_chunks(text):
    """将文本按句子分隔符分割成chunk，并确保每个chunk长度大于200"""
    if not text:
        return []
    
    # 替换多余空格
    text = re.sub(r'\s+', ' ', text)
    
    # 按句子分隔符分割
    sentences = re.split(SENTENCE_DELIMITER, text)
    
    # 过滤空sentences并移除前后空格
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # 合并短句，确保每个chunk长度大于200
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # 如果当前chunk为空，直接添加这个句子
        if not current_chunk:
            current_chunk = sentence
        # 否则，将句子添加到当前chunk
        else:
            current_chunk += sentence
        
        # 如果当前chunk长度大于200，保存它并重置current_chunk
        if len(current_chunk) > 200:
            chunks.append(current_chunk)
            current_chunk = ""
    
    # 处理最后可能剩余的chunk
    if current_chunk and len(current_chunk) > 200:
        chunks.append(current_chunk)
    elif current_chunk and chunks:  # 如果最后一个chunk太短，将它添加到前一个chunk
        chunks[-1] += current_chunk
    
    return chunks

def process_pdf_files(target_subfolder=None):
    """处理PDF文件，先转为Markdown，再切割chunk
    
    Args:
        target_subfolder: 要处理的子文件夹名称，如果为None则处理所有子文件夹
    """
    # 创建输出目录
    ensure_dir(OUTPUT_DIR)
    ensure_dir(MARKDOWN_DIR)
    
    # 确定要处理的子目录
    if target_subfolder:
        subfolder_path = os.path.join(PDF_DIR, target_subfolder)
        if os.path.isdir(subfolder_path):
            subdirs = [target_subfolder]
            print(f"只处理指定子文件夹: {target_subfolder}")
        else:
            print(f"错误：指定的子文件夹 '{target_subfolder}' 不存在")
            return 0, 0
    else:
        # 获取宫颈癌目录下的所有子目录
        subdirs = [d for d in os.listdir(PDF_DIR) if os.path.isdir(os.path.join(PDF_DIR, d))]
        # 将根目录也加入处理
        subdirs.append("")
    
    total_chunks = 0
    total_files = 0
    
    # 检查已经处理过的文件
    processed_files = {}
    for root, _, files in os.walk(OUTPUT_DIR):
        for file in files:
            if file.endswith('.json'):
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for item in data:
                            if 'source' in item:
                                processed_files[item['source']] = file
                except Exception as e:
                    print(f"读取已处理文件 {file} 时出错: {e}")
    
    print(f"找到 {len(processed_files)} 个已处理的源文件")
    
    # 处理每个子目录
    for subdir in tqdm(subdirs, desc="处理子目录"):
        subdir_path = os.path.join(PDF_DIR, subdir)
        
        if not os.path.isdir(subdir_path):
            continue
            
        # 获取子目录中的所有PDF文件
        pdf_files = glob.glob(os.path.join(subdir_path, "*.pdf"))
        
        if not pdf_files:
            continue
            
        # 为子目录创建输出文件名
        if subdir:
            output_filename = f"{subdir.replace('/', '_')}.json"
            print(f"\n处理子目录: {subdir}，包含 {len(pdf_files)} 个PDF文件")
        else:
            output_filename = "根目录.json"
            print(f"\n处理根目录，包含 {len(pdf_files)} 个PDF文件")
            
        output_file = os.path.join(OUTPUT_DIR, output_filename)
        
        # 用于保存该子目录所有PDF的chunks
        subdir_chunks = []
        
        # 读取现有的JSON文件（如果存在）
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    subdir_chunks = json.load(f)
                print(f"从现有文件 {output_filename} 中加载了 {len(subdir_chunks)} 个chunks")
            except Exception as e:
                print(f"读取现有文件 {output_filename} 时出错: {e}")
                subdir_chunks = []
        
        # 记录已经处理过的文件
        processed_in_this_dir = {item['source'] for item in subdir_chunks if 'source' in item}
        
        # 处理子目录中的每个PDF文件
        for pdf_file in tqdm(pdf_files, desc=f"处理 {subdir if subdir else '根目录'} 中的PDF"):
            try:
                # 提取文件名（不包含路径和扩展名）
                file_name = os.path.basename(pdf_file).rsplit('.', 1)[0]
                
                # 检查是否已经处理过
                if file_name in processed_in_this_dir:
                    print(f"跳过已处理的文件: {file_name}")
                    continue
                
                # 使用MinerU提取PDF到Markdown
                print(f"使用MinerU提取: {file_name}")
                md_file = extract_pdf_to_markdown(pdf_file, MARKDOWN_DIR)
                
                if md_file and os.path.exists(md_file):
                    # 读取Markdown内容
                    markdown_text = read_markdown_file(md_file)
                    
                    # 分割成chunks
                    chunks = split_text_into_chunks(markdown_text)
                    
                    # 将每个chunk添加到该子目录的结果列表
                    valid_chunks = []
                    for chunk in chunks:
                        valid_chunks.append({
                            "text": chunk,
                            "source": file_name,
                            "markdown_source": md_file
                        })
                    
                    subdir_chunks.extend(valid_chunks)
                    total_chunks += len(valid_chunks)
                    total_files += 1
                    
                    print(f"从 {file_name} 提取了 {len(valid_chunks)} 个chunks")
                    
                    # 处理完一个文件就保存一次，避免中断导致数据丢失
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(subdir_chunks, f, ensure_ascii=False, indent=2)
                    print(f"更新保存至 {output_file}")
                else:
                    print(f"警告: 无法提取 {file_name} 的Markdown内容")
                
            except Exception as e:
                print(f"处理文件 {pdf_file} 时出错: {e}")
                import traceback
                traceback.print_exc()
        
        # 最终保存该子目录的结果到对应的JSON文件
        if subdir_chunks:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(subdir_chunks, f, ensure_ascii=False, indent=2)
            print(f"已将 {len(subdir_chunks)} 个chunks从 {subdir if subdir else '根目录'} 保存至 {output_file}")
    
    return total_chunks, total_files

def main():
    """主函数"""
    print("开始处理宫颈癌PDF文件...")
    print("使用MinerU Python API提取PDF到Markdown，然后切割为chunks")
    
    # 解析命令行参数
    args = parse_args()
    
    # 如果指定了clean参数，清空输出目录
    if args.clean:
        if args.subfolder:
            # 只清空特定子文件夹对应的输出
            base_name = args.subfolder.replace('/', '_')
            output_file = os.path.join(OUTPUT_DIR, f"{base_name}.json")
            if os.path.exists(output_file):
                print(f"删除文件: {output_file}")
                os.remove(output_file)
        else:
            # 清空整个输出目录
            if os.path.exists(OUTPUT_DIR):
                print(f"清空输出目录: {OUTPUT_DIR}")
                shutil.rmtree(OUTPUT_DIR)
                os.makedirs(OUTPUT_DIR)
    
    if args.force:
        print("强制模式: 将重新处理所有文件")
    
    # 处理所有PDF文件
    total_chunks, total_files = process_pdf_files(args.subfolder)
    
    print(f"\n处理完成。共处理 {total_files} 个PDF文件，提取 {total_chunks} 个chunks")
    print(f"Markdown文件保存在: {MARKDOWN_DIR}")
    print(f"JSON结果保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 