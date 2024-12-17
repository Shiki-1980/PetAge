import os
from collections import Counter

# 定义文件夹路径
cropped_valset_path = "./croppedValset"
valset_path = "./valset"

# 检查路径是否存在
if not os.path.exists(cropped_valset_path) or not os.path.exists(valset_path):
    print("请检查文件夹路径是否正确！")
    exit()

# 获取文件夹中的文件列表
valset_files = set(os.listdir(valset_path))
cropped_valset_files = list(os.listdir(cropped_valset_path))  # 使用列表以检测重复

# 找出valset中缺失或重复的文件
missing_files = valset_files - set(cropped_valset_files)
extra_files = set(cropped_valset_files) - valset_files
duplicates = [file for file, count in Counter(cropped_valset_files).items() if count > 1]

# 输出结果
if missing_files:
    print(f"缺少的文件 ({len(missing_files)} 个):")
    print("\n".join(missing_files))
else:
    print("没有缺少的文件。")

if duplicates:
    print(f"\n重复的文件 ({len(duplicates)} 个):")
    print("\n".join(duplicates))
else:
    print("没有重复的文件。")

if extra_files:
    print(f"\n多余的文件 ({len(extra_files)} 个):")
    print("\n".join(extra_files))
else:
    print("没有多余的文件。")
