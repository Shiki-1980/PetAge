import numpy as np

def calculate_mean_std(file_path):
    labels = []
    
    # 读取文件中的每一行
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()  # 以空格分割每一行
            label = float(parts[1])  # 假设第二列是标签（年龄，月份）
            labels.append(label)
    
    labels = np.array(labels)
    mean = np.mean(labels)  # 计算均值
    std = np.std(labels)    # 计算标准差
    
    return mean, std

# 读取 train.txt 和 val.txt 计算均值和标准差
train_file = './DataSet/annotations/train.txt'
val_file = './DataSet/annotations/val.txt'

train_mean, train_std = calculate_mean_std(train_file)
val_mean, val_std = calculate_mean_std(val_file)
import numpy as np

def calculate_mean_std(file_path):
    labels = []
    
    # 读取文件中的每一行
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()  # 以空格分割每一行
            label = float(parts[1])  # 假设第二列是标签（年龄，月份）
            labels.append(label)
    
    labels = np.array(labels)
    mean = np.mean(labels)  # 计算均值
    std = np.std(labels)    # 计算标准差
    
    return mean, std

train_mean, train_std = calculate_mean_std(train_file)
val_mean, val_std = calculate_mean_std(val_file)

# 输出结果
print(f"Train set - Mean: {train_mean}, Std: {train_std}")
print(f"Validation set - Mean: {val_mean}, Std: {val_std}")

# 计算所有数据的均值和标准差
all_labels = []

# 合并训练集和验证集的标签
with open(train_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        all_labels.append(float(parts[1]))

with open(val_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        all_labels.append(float(parts[1]))

all_labels = np.array(all_labels)
all_mean = np.mean(all_labels)
all_std = np.std(all_labels)

print(f"Overall - Mean: {all_mean}, Std: {all_std}")


