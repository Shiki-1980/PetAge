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
from model import ModelReg320,Model50,BreedDual,OptimModel,BreedOptim
import torchvision


# 设置加载路径
checkpoint_path = os.path.join(os.getcwd(), './results/Breed/Breed360/best/best_model.pth')
#checkpoint_path = os.path.join(os.getcwd(), './results/regnetx_320/SmoothL1/best_model.pth')
# 加载模型权重
def load_model(checkpoint_path, model):
    # 检查权重文件是否存在
    if os.path.exists(checkpoint_path):
        print(f"加载模型权重: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device1)
        
        # 如果权重是存储在 'model_state_dict' 中
        model.load_state_dict(checkpoint)
        
        print("模型权重加载成功!")
    else:
        print(f"权重文件 {checkpoint_path} 未找到!")

if __name__ == "__main__":
    # device = torch.device('cuda:6')
    # model = BreedDual()
    device1 = torch.device('cuda:3')
    device2 = torch.device('cuda:6')
    device3 = torch.device('cuda:5')
    #model = DualModel('resnet34','resnet34',0.6,0.1)
    model=BreedOptim(device1=device1,device2=device2,device3=device3)
    #model=ModelReg320()
    #gpus = [6,7]
    #model = nn.DataParallel(model, device_ids=gpus)
    img_size = 256
    #model = model.to(device)
    transform2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([img_size, img_size]),
    ])
    eval_dataset = BatchDataset('eval', transform2)
    eval_dataset_loader = DataLoader(eval_dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)
    #load_model(checkpoint_path, model)
    model.eval()
    maes = []
    loss_fn = nn.L1Loss().to(device1)
    output_file = './prediction/Optim2/Predictions2.txt'
    with open(output_file, 'w') as f:
        with torch.no_grad():
            for data in tqdm(eval_dataset_loader):
                image, age, filename = data
                image = image.to(device1)
                age = age.to(device1)
                out,check= model(image,age)
                out=out.to(device1)
                check =check.to(device1)
                # 打印预测值和真实标签
                # print(out)
                # print(age)

                # 写入每个文件的预测结果（文件名与输出中间用空格隔开）
                for fn, pred,check in zip(filename, out.cpu().numpy(),check.cpu().numpy()):
                    f.write(f"{fn}\t{pred}\t{check}\n")  # 假设pred是一个大小为[batch_size, 1]的张量

                # 计算MAE
                mae = loss_fn(out, age)
                maes.append(mae.item())

        mae_mean = np.array(maes).mean()
        print('eval dataset mae: ', mae_mean)