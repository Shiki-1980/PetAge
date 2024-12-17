import os
import shutil
import re

# 设置源目录和目标目录
source_dir = './CleanTrainduplicates'
target_dir = './doubledouble'

# 创建目标目录（如果不存在）
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 获取目录中的所有文件
files = os.listdir(source_dir)
print(len(files))
# 使用字典来存储重复的图片文件
file_dict = {}

# 遍历所有文件
for file in files:
    if file.endswith('.jpg'):  # 只处理jpg文件
        # 提取文件的基础名称（去掉数字后缀和扩展名）
        base_name = re.match(r'^(.*?)(\d+)?\.jpg$', file)
        if base_name:
            base_name = base_name.group(1)  # 获取基本文件名

            # 如果基本文件名已经存在于字典中，说明找到了重复
            if base_name in file_dict:
                file_dict[base_name].append(file)
            else:
                file_dict[base_name] = [file]

# 遍历字典，找出所有有重复的图片
for base_name, file_list in file_dict.items():
    if len(file_list) > 1:  # 如果有多个文件，则说明这些文件是重复的
        print(f'Found duplicates for {base_name}: {file_list}')
        
        # 移动重复的图片到目标目录
        for file in file_list:
            source_path = os.path.join(source_dir, file)
            target_path = os.path.join(target_dir, file)
            shutil.move(source_path, target_path)
            print(f'Moved {file} to {target_dir}')

print("Finished moving duplicate images.")
