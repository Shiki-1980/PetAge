import os
import re

# 设置目录路径
directory = './croppedTrainset'  # 替换成你需要处理的目录路径

# 获取目录中的所有文件
files = os.listdir(directory)

# 遍历文件
for file in files:
    # 确保只处理 .jpg 文件
    if file.endswith('.jpg'):
        # 使用正则表达式查找并去掉 Q 后面的数字后缀
        new_name = re.sub(r'Q\d+(\.jpg)$', r'Q\1', file)
        
        # 如果文件名发生了变化，进行重命名
        if new_name != file:
            old_path = os.path.join(directory, file)
            new_path = os.path.join(directory, new_name)
            os.rename(old_path, new_path)
            print(f'Renamed: {file} -> {new_name}')

print("Finished renaming files.")
