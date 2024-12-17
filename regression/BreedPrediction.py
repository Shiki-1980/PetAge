import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import time
from datetime import timedelta
from model import Breed50
from datasets import BatchDataset
from torch.utils.data import DataLoader,dataset
from tqdm import tqdm
import torch.nn.functional as F

# 设置 DataSet 目录路径
dataset_dir = "../DataSet/breed"
img_size = 256
transform2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([img_size, img_size]),
])

# 获取所有子目录（类别名）
class_names = [d.name for d in os.scandir(dataset_dir) if d.is_dir()]
device = torch.device("cuda:7")
eval_dataset = BatchDataset('eval', transform2)
eval_dataset_loader = DataLoader(eval_dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)
model = Breed50().to(device)
model.eval()
output_file = './prediction/Breed/evalPredictions.txt'
with open(output_file, 'w') as f:
    with torch.no_grad():
        for data in tqdm(eval_dataset_loader):
            image, age, filename = data
            image = image.to(device)
            age = age.to(device)
            out = model(image)
            probabilities = F.softmax(out, dim=1)
            #print(probabilities)
            _, predicted_idx = torch.max(out, 1)
            #print(predicted_idx)
            pred = [class_names[idx.item()] for idx in predicted_idx]  # 将每个索引映射为类别名称
            #print(pred)
            # 打印预测值和真实标签
            # print(out)
            # print(age)

            # 写入每个文件的预测结果（文件名与输出中间用空格隔开）
            for fn, pred in zip(filename, pred):
                print(pred)
                f.write(f"{fn}\t{pred}\n")  # 假设pred是一个大小为[batch_size, 1]的张量


# 打印类别名称列表
print(class_names)
print(len(class_names))