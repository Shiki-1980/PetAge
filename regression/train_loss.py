import glob
import os.path
import cv2
import numpy as np
import rawpy
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader,dataset
from torchvision import transforms
from tqdm import tqdm
from datasets import BatchDataset
from model import ModelReg320,Model50,BreedDual,Model,Model34,ModelCLIP
import torchvision
import matplotlib.pyplot as plt


# 设置加载路径
checkpoint_path = os.path.join(os.getcwd(), './results/CLIP/best_model.pth')

# 加载模型权重
def load_model(checkpoint_path, model):
    # 检查权重文件是否存在
    if os.path.exists(checkpoint_path):
        print(f"加载模型权重: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 如果权重是存储在 'model_state_dict' 中
        model.load_state_dict(checkpoint)
        
        print("模型权重加载成功!")
    else:
        print(f"权重文件 {checkpoint_path} 未找到!")

if __name__ == "__main__":
    device = torch.device('cuda:7')
    model = ModelCLIP()

    img_size = 224
    model = model.to(device)
    transform2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([img_size, img_size]),
    ])
    
    train_dataset = BatchDataset('train', transform2)
    train_dataset_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)
    load_model(checkpoint_path, model)
    model.eval()

    loss_fn = nn.L1Loss().to(device)
    output_file = './prediction/CLIP/TrainMAEloss.txt'
    
    # 存储每个图片的loss
    loss_dict = {}

    with open(output_file, 'w') as f:
        with torch.no_grad():
            for data in tqdm(train_dataset_loader):
                image, age, filename = data
                image = image.to(device)
                age = age.to(device)
                out = model(image)

#                写入每个文件的损失值（文件名与loss之间用空格隔开）
                for fn, pred, true_age in zip(filename, out, age):
                    # 计算每个图片的loss
                    loss = loss_fn(pred, true_age).item()  # 计算损失值，并转为python原生类型
                    loss_dict[fn] = loss  # 图片名为key，loss为value
                    
                    # 将文件名和对应的损失值写入文件
                    f.write(f"{fn}\t{loss}\n")  # 假设fn是文件名，loss是对应的损失值
                    #f.write(f"{fn}\t{pred}\n") #写入预测值


    # 绘制loss的分布图
    losses = list(loss_dict.values())
    
    plt.figure(figsize=(10, 6))
    plt.hist(losses, bins=50, color='blue', alpha=0.7)
    plt.title('Loss Distribution')
    plt.xlabel('Loss')
    plt.ylabel('Frequency')
    plt.grid(True)
# 保存图像到文件
    loss_plot_path = './prediction/CLIP/Trainloss_distribution.png'
    plt.savefig(loss_plot_path)
    print(f"Loss distribution plot saved to {loss_plot_path}")

    # 输出平均MAE
    mae_mean = np.array(losses).mean()
    print('eval dataset mae: ', mae_mean)