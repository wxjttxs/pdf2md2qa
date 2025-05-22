# 原始pdf 切割成chunk
## 只处理黄珊文件夹
python chunk_gongjingai.py --subfolder 黄珊

## 强制重新处理黄珊文件夹并清除之前的结果
python chunk_gongjingai.py --subfolder 黄珊 --clean --force

## 处理所有子文件夹（原来的行为）
python chunk_gongjingai.py

# chunk转换成问答对
python process_selected_files.py

python process_selected_files.py --input_dir=其他路径 --num_process=10