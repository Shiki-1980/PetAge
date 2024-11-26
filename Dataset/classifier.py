import os
import shutil

# 定义路径
annotations_dir = './annotations'
train_txt_path = os.path.join(annotations_dir, 'val.txt')
trainset_dir = './valset'
classify_trainset_dir = './classify_valset'

# 创建classify_trainset目录（如果不存在的话）
if not os.path.exists(classify_trainset_dir):
    os.makedirs(classify_trainset_dir)

# 读取 train.txt 文件，获取图片名和标签
with open(train_txt_path, 'r') as file:
    lines = file.readlines()

# 遍历每一行，按标签组织图片
for line in lines:
    # 分割图片名和标签
    image_name, label = line.strip().split()
    
    # 构建源文件路径和目标目录路径
    src_image_path = os.path.join(trainset_dir, image_name)
    label_dir = os.path.join(classify_trainset_dir, label)
    
    # 如果标签目录不存在，创建该目录
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    
    # 复制图片到目标标签目录
    if os.path.exists(src_image_path):  # 确保源图片存在
        shutil.copy(src_image_path, os.path.join(label_dir, image_name))
    else:
        print(f"Warning: {src_image_path} does not exist.")
