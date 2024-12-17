import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

import os
from PIL import Image

class BatchDataset(Dataset):
    def __init__(self, mode, transform=None, data_dir='../DataSet'):
        """
        初始化数据集
        :param mode: 'train' 或 'eval'，指定数据集类型
        :param transform: 数据预处理的转换函数
        :param data_dir: 数据集所在的目录
        """
        self.mode = mode
        self.transform = transform
        self.data_dir = data_dir

        # 加载图片路径和标签
        self.image_paths = self.load_image_paths(mode)  # 加载图片路径
        self.labels = self.load_labels(mode)  # 加载标签

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 获取图片路径和标签
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 读取图片
        img = Image.open(img_path)

        # 如果有预处理，应用它
        if self.transform:
            img = self.transform(img)

        # 获取文件名
        file_name = os.path.basename(img_path)
        return img, label, file_name  # 返回图片、标签和文件名

    def load_image_paths(self, mode):
        """
        加载数据路径
        :param mode: 'train' 或 'eval'
        :return: 图片路径列表
        """
        image_paths = []
        if mode == 'train' or mode == 'eval':
            # 假设图片存储在data_dir/trainset/ 或 data_dir/valset/ 文件夹中
            #croppedTrainset  croppedValset
            img_dir = os.path.join(self.data_dir, 'croppedTrainset' if mode == 'train' else 'croppedValset')
            image_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.jpeg')]
        print(len(image_paths))
        return image_paths

    def load_labels(self, mode):
        """
        加载标签，图片名与标签通过train.txt或valset.txt关联
        :param mode: 'train' 或 'eval'
        :return: 标签列表
        """
        labels = []
        label_file = ''
        if mode == 'train':
            label_file = os.path.join(self.data_dir, 'annotations', 'SmoothTrain.txt')
        elif mode == 'eval':
            label_file = os.path.join(self.data_dir, 'annotations', 'val.txt')
        else:
            raise ValueError("Mode must be 'train' or 'eval'")

        # 创建图片名到标签的映射
        img_to_label = {}
        
        with open(label_file, 'r') as f:
            for line in f:
                img_name, label = line.strip().split()
                img_to_label[img_name] = int(label)  # 假设标签是整数类型

        # 使用图片名列表获取标签
        for img_path in self.image_paths:
            img_name = os.path.basename(img_path)
            labels.append(img_to_label.get(img_name, -1))  # 默认标签为-1，如果找不到对应标签

        return labels

