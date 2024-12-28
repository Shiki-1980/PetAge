import os
import shutil

# 定义文件夹路径
cropped_valset_path = "./croppedValset"
valset_path = "./valset"

# 检查路径是否存在
if not os.path.exists(cropped_valset_path) or not os.path.exists(valset_path):
    print("请检查文件夹路径是否正确！")
    exit()

# 获取文件夹中的文件列表
cropped_valset_files = set(os.listdir(cropped_valset_path))
valset_files = set(os.listdir(valset_path))

# 找出valset中有但croppedValset中没有的文件
missing_files = valset_files - cropped_valset_files

if not missing_files:
    print("croppedValset 中没有缺失文件。")
else:
    print(f"发现 {len(missing_files)} 个缺失文件，正在复制...")

    # 将缺失的文件从 valset 复制到 croppedValset
    for file_name in missing_files:
        src = os.path.join(valset_path, file_name)
        dest = os.path.join(cropped_valset_path, file_name)
        shutil.copy2(src, dest)
        print(f"已复制: {file_name}")

    print("所有缺失文件已复制完成。")
