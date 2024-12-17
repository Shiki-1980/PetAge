import os
import shutil

# 定义文件路径
annotations_path = "./annotations/val.txt"
source_folder = "./croppedValset"
middle_train_folder = "./MiddleTest"
other_train_folder = "./Othertest"

# 确保目标文件夹存在
os.makedirs(middle_train_folder, exist_ok=True)
os.makedirs(other_train_folder, exist_ok=True)

# 读取 annotations 文件并处理
with open(annotations_path, "r") as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    if not line:
        continue
    image_name, age_str = line.split("\t")
    age = int(age_str)

    # 根据分类规则将图片复制到对应的文件夹
    source_path = os.path.join(source_folder, image_name)
    if age >= 20 and age <= 90:
        target_folder = middle_train_folder
    else:
        target_folder = other_train_folder

    # 如果图片存在则复制
    #print(source_path)
    if os.path.exists(source_path):
        shutil.copy(source_path, os.path.join(target_folder, image_name))
    else:
        print(f"Warning: {source_path} does not exist.")
