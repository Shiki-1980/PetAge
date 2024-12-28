import os

# 定义文件所在的目录
trainset_dir = './valset'
print("1")
# 遍历目录下的所有文件
for filename in os.listdir(trainset_dir):
    # 检查是否是以 "A" 开头的 jpg 文件
    if filename.startswith("A") and filename.endswith(".jpg"):
        # 找到第一个下划线的位置并替换为 "*"
        new_filename = filename.replace("_", "*", 1)
        
        # 构建完整路径
        old_path = os.path.join(trainset_dir, filename)
        new_path = os.path.join(trainset_dir, new_filename)
        
        # 重命名文件
        os.rename(old_path, new_path)
        print(f'Renamed: {filename} -> {new_filename}')
