import os
import shutil

import pandas as pd

# 定义CSV文件路径和图片文件夹路径
csv_file = './dog-breed-identification/labels.csv'  # 替换为实际的CSV文件路径
train_dir = './dog-breed-identification/train'  # 替换为实际的训练集图片文件夹路径
output_dir = 'breed'  # 替换为你希望保存分类结果的文件夹路径

# 创建分类文件夹
def create_breed_dirs(df):
    breeds = df['breed'].unique()
    for breed in breeds:
        breed_dir = os.path.join(output_dir, breed)
        if not os.path.exists(breed_dir):
            os.makedirs(breed_dir)

# 按照breed分类图片
def classify_images(df):
    for _, row in df.iterrows():
        breed = row['breed']
        image_id = row['id']
        image_name = image_id + '.jpg'  # 假设图片格式为 .jpg，若不同请修改
        image_path = os.path.join(train_dir, image_name)
        
        if os.path.exists(image_path):
            # 创建 breed 文件夹
            breed_dir = os.path.join(output_dir, breed)
            if not os.path.exists(breed_dir):
                os.makedirs(breed_dir)
            
            # 移动图片到对应的 breed 文件夹
            shutil.move(image_path, os.path.join(breed_dir, image_name))
        else:
            print(f"Image {image_name} does not exist in the train directory.")

# 读取CSV文件
df = pd.read_csv(csv_file)

# 创建 breed 目录
create_breed_dirs(df)

# 分类图片
classify_images(df)

print("Image classification completed.")
