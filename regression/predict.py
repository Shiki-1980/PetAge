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
from model import Model, Model2
import torchvision


# 设置加载路径
checkpoint_path = os.path.join(os.getcwd(), 'saved_model_age/first/checkpoint_0008.pth')

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
    device = torch.device('cuda:6')
    model = Model()

    #gpus = [6,7]
    #model = nn.DataParallel(model, device_ids=gpus)
    img_size = 256
    model = model.to(device)
    transform2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([img_size, img_size]),
    ])
    eval_dataset = BatchDataset('eval', transform2)
    eval_dataset_loader = DataLoader(eval_dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)
    load_model(checkpoint_path, model)
    model.eval()
    maes = []
    loss_fn = nn.L1Loss().to(device)
    output_file = 'predictions.txt'
    with open(output_file, 'w') as f:
        with torch.no_grad():
            for data in tqdm(eval_dataset_loader):
                image, age, filename = data
                image = image.to(device)
                age = age.to(device)
                out = model(image)

                # 打印预测值和真实标签
                # print(out)
                # print(age)

                # 写入每个文件的预测结果（文件名与输出中间用空格隔开）
                for fn, pred in zip(filename, out.cpu().numpy()):
                    f.write(f"{fn}\t{pred}\n")  # 假设pred是一个大小为[batch_size, 1]的张量

                # 计算MAE
                mae = loss_fn(out, age)
                maes.append(mae.item())

        mae_mean = np.array(maes).mean()
        print('eval dataset mae: ', mae_mean)