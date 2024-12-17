import os
import time
import copy
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import BatchDataset
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
from PIL import Image

# Model
class SSRNet(nn.Module):
    def __init__(self, stage_num=[3, 3, 3], image_size=64,
                 class_range=192, lambda_index=1., lambda_delta=1.):
        super(SSRNet, self).__init__()
        self.image_size = image_size
        self.stage_num = stage_num
        self.lambda_index = lambda_index
        self.lambda_delta = lambda_delta
        self.class_range = class_range
        
        # Stream 1: Increased channel sizes and more convolutional layers
        self.stream1_stage3 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(4, 4)
        )
        self.stream1_stage2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )
        self.stream1_stage1 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # Stream 2: Increased channel sizes and more convolutional layers
        self.stream2_stage3 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.MaxPool2d(4, 4)
        )
        self.stream2_stage2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.MaxPool2d(2, 2)
        )
        self.stream2_stage1 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.Tanh(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.Tanh(),
        )
        
        # Fusion Blocks (no major changes here, as they are part of the high-level fusion structure)
        self.funsion_block_stream1_stage_3_before_PB = nn.Sequential(
            nn.Conv2d(64, 10, 1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(16, 16)
        )
        self.funsion_block_stream1_stage_3_prediction_block = nn.Sequential(
            nn.Dropout(0.2, ),
            nn.Linear(10 * 4 * 4, self.stage_num[2]),
            nn.ReLU()
        )
        
        self.funsion_block_stream1_stage_2_before_PB = nn.Sequential(
            nn.Conv2d(128, 10, 1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(4, 4)
        )
        self.funsion_block_stream1_stage_2_prediction_block = nn.Sequential(
            nn.Dropout(0.2, ),
            nn.Linear(10 * 8 * 8, self.stage_num[1]),
            nn.ReLU()
        )
        
        self.funsion_block_stream1_stage_1_before_PB = nn.Sequential(
            nn.Conv2d(256, 10, 1, padding=0),
            nn.ReLU(),
        )
        self.funsion_block_stream1_stage_1_prediction_block = nn.Sequential(
            nn.Dropout(0.2, ),
            nn.Linear(10 * 16 * 16, self.stage_num[0]),
            nn.ReLU()
        )
        
        # Similar changes for Stream 2
        self.funsion_block_stream2_stage_3_before_PB = nn.Sequential(
            nn.Conv2d(64, 10, 1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(16, 16)
        )
        self.funsion_block_stream2_stage_3_prediction_block = nn.Sequential(
            nn.Dropout(0.2, ),
            nn.Linear(10 * 4 * 4, self.stage_num[2]),
            nn.ReLU()
        )
        
        self.funsion_block_stream2_stage_2_before_PB = nn.Sequential(
            nn.Conv2d(128, 10, 1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(4, 4)
        )
        self.funsion_block_stream2_stage_2_prediction_block = nn.Sequential(
            nn.Dropout(0.2, ),
            nn.Linear(10 * 8 * 8, self.stage_num[1]),
            nn.ReLU()
        )
        
        self.funsion_block_stream2_stage_1_before_PB = nn.Sequential(
            nn.Conv2d(256, 10, 1, padding=0),
            nn.ReLU(),
        )
        self.funsion_block_stream2_stage_1_prediction_block = nn.Sequential(
            nn.Dropout(0.2, ),
            nn.Linear(10 * 16 * 16, self.stage_num[0]),
            nn.ReLU()
        )
        
        # Final stages FC layers
        self.stage3_FC_after_PB = nn.Sequential(
            nn.Linear(self.stage_num[0], 2 * self.stage_num[0]),
            nn.ReLU()
        )
        self.stage3_prob = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.ReLU()
        )
        self.stage3_index_offsets = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.Tanh()
        )
        self.stage3_delta_k = nn.Sequential(
            nn.Linear(10 * 4 * 4, 1),
            nn.Tanh()
        )
        
        self.stage2_FC_after_PB = nn.Sequential(
            nn.Linear(self.stage_num[0], 2 * self.stage_num[0]),
            nn.ReLU()
        )
        self.stage2_prob = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.ReLU()
        )
        self.stage2_index_offsets = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.Tanh()
        )
        self.stage2_delta_k = nn.Sequential(
            nn.Linear(10 * 8 * 8, 1),
            nn.Tanh()
        )
        
        self.stage1_FC_after_PB = nn.Sequential(
            nn.Linear(self.stage_num[0], 2 * self.stage_num[0]),
            nn.ReLU()
        )
        self.stage1_prob = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.ReLU()
        )
        self.stage1_index_offsets = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.Tanh()
        )
        self.stage1_delta_k = nn.Sequential(
            nn.Linear(10 * 16 * 16, 1),
            nn.Tanh()
        )
        self.init_params()
    
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def forward(self, image_):
        feature_stream1_stage3 = self.stream1_stage3(image_)
        feature_stream1_stage2 = self.stream1_stage2(feature_stream1_stage3)
        feature_stream1_stage1 = self.stream1_stage1(feature_stream1_stage2)
        feature_stream2_stage3 = self.stream2_stage3(image_)
        
        feature_stream2_stage2 = self.stream2_stage2(feature_stream2_stage3)
        
        feature_stream2_stage1 = self.stream2_stage1(feature_stream2_stage2)
        
        feature_stream1_stage3_before_PB = self.funsion_block_stream1_stage_3_before_PB(feature_stream1_stage3)
        feature_stream1_stage2_before_PB = self.funsion_block_stream1_stage_2_before_PB(feature_stream1_stage2)
        feature_stream1_stage1_before_PB = self.funsion_block_stream1_stage_1_before_PB(feature_stream1_stage1)
        feature_stream2_stage3_before_PB = self.funsion_block_stream2_stage_3_before_PB(feature_stream2_stage3)
        feature_stream2_stage2_before_PB = self.funsion_block_stream2_stage_2_before_PB(feature_stream2_stage2)
        feature_stream2_stage1_before_PB = self.funsion_block_stream2_stage_1_before_PB(feature_stream2_stage1)
        
        embedding_stream1_stage3_before_PB = feature_stream1_stage3_before_PB.view(feature_stream1_stage3_before_PB.size(0), -1)
        embedding_stream1_stage2_before_PB = feature_stream1_stage2_before_PB.view(feature_stream1_stage2_before_PB.size(0), -1)
        embedding_stream1_stage1_before_PB = feature_stream1_stage1_before_PB.view(feature_stream1_stage1_before_PB.size(0), -1)
        
        embedding_stream2_stage3_before_PB = feature_stream2_stage3_before_PB.view(feature_stream2_stage3_before_PB.size(0), -1)
        embedding_stream2_stage2_before_PB = feature_stream2_stage2_before_PB.view(feature_stream2_stage2_before_PB.size(0), -1)
        embedding_stream2_stage1_before_PB = feature_stream2_stage1_before_PB.view(feature_stream2_stage1_before_PB.size(0), -1)
        stage1_delta_k = self.stage1_delta_k(torch.mul(embedding_stream1_stage1_before_PB, embedding_stream2_stage1_before_PB))
        stage2_delta_k = self.stage2_delta_k(torch.mul(embedding_stream1_stage2_before_PB, embedding_stream2_stage2_before_PB))
        stage3_delta_k = self.stage3_delta_k(torch.mul(embedding_stream1_stage3_before_PB, embedding_stream2_stage3_before_PB))
        embedding_stage1_after_PB = torch.mul(self.funsion_block_stream1_stage_1_prediction_block(embedding_stream1_stage1_before_PB),
                                              self.funsion_block_stream2_stage_1_prediction_block(embedding_stream2_stage1_before_PB))
        embedding_stage2_after_PB = torch.mul(self.funsion_block_stream1_stage_2_prediction_block(embedding_stream1_stage2_before_PB),
                                              self.funsion_block_stream2_stage_2_prediction_block(embedding_stream2_stage2_before_PB))
        embedding_stage3_after_PB = torch.mul(self.funsion_block_stream1_stage_3_prediction_block(embedding_stream1_stage3_before_PB),
                                              self.funsion_block_stream2_stage_3_prediction_block(embedding_stream2_stage3_before_PB))
        
        embedding_stage1_after_PB = self.stage1_FC_after_PB(embedding_stage1_after_PB)
        embedding_stage2_after_PB = self.stage2_FC_after_PB(embedding_stage2_after_PB)
        embedding_stage3_after_PB = self.stage3_FC_after_PB(embedding_stage3_after_PB)
        
        prob_stage_1 = self.stage1_prob(embedding_stage1_after_PB)
        index_offset_stage1 = self.stage1_index_offsets(embedding_stage1_after_PB)
        
        prob_stage_2 = self.stage2_prob(embedding_stage2_after_PB)
        index_offset_stage2 = self.stage2_index_offsets(embedding_stage2_after_PB)
        
        prob_stage_3 = self.stage3_prob(embedding_stage3_after_PB)
        index_offset_stage3 = self.stage3_index_offsets(embedding_stage3_after_PB)
        stage1_regress = prob_stage_1[:, 0] * 0
        stage2_regress = prob_stage_2[:, 0] * 0
        stage3_regress = prob_stage_3[:, 0] * 0
        for index in range(self.stage_num[0]):
            stage1_regress = stage1_regress + (index + self.lambda_index * index_offset_stage1[:, index]) * prob_stage_1[:, index]
        stage1_regress = torch.unsqueeze(stage1_regress, 1)
        stage1_regress = stage1_regress / (self.stage_num[0] * (1 + self.lambda_delta * stage1_delta_k))
        
        for index in range(self.stage_num[1]):
            stage2_regress = stage2_regress + (index + self.lambda_index * index_offset_stage2[:, index]) * prob_stage_2[:, index]
        stage2_regress = torch.unsqueeze(stage2_regress, 1)
        stage2_regress = stage2_regress / (self.stage_num[0] * (1 + self.lambda_delta * stage1_delta_k) *
                                           (self.stage_num[1] * (1 + self.lambda_delta * stage2_delta_k)))
        
        for index in range(self.stage_num[2]):
            stage3_regress = stage3_regress + (index + self.lambda_index * index_offset_stage3[:, index]) * prob_stage_3[:, index]
        stage3_regress = torch.unsqueeze(stage3_regress, 1)
        stage3_regress = stage3_regress / (self.stage_num[0] * (1 + self.lambda_delta * stage1_delta_k) *
                                           (self.stage_num[1] * (1 + self.lambda_delta * stage2_delta_k)) *
                                           (self.stage_num[2] * (1 + self.lambda_delta * stage3_delta_k))
                                           )
        regress_class = (stage1_regress + stage2_regress + stage3_regress) * self.class_range
        regress_class = torch.squeeze(regress_class, 1)
        return regress_class
    
# Define training function
save_model_dir="./results/SSR-Net"
def train_model(model_, dataloaders_, criterion_, optimizer_,device, num_epochs_=25):
    global lr_scheduler

    since = time.time()
    val_acc_history = []

    best_model_wts = copy.deepcopy(model_.state_dict())
    best_acc = float("Inf")
    # tensorboard_writer.add_graph(model_, dataloaders_['train'])
    for epoch in range(1, num_epochs_ + 1):
        print(f"Epoch [{epoch}/{num_epochs_}]")
        print("=====" * 10)

        # for phase in ['train', 'val']:
        for phase in sorted(dataloaders_.keys()):
            if phase == "train":
                model_.train()  # Set model to training mode
            else:
                model_.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_error = 0.0
            running_corrects_3 = 0
            running_corrects_5 = 0
            g_loss = []
            g_mae = []
            for data in tqdm(dataloaders_[phase]):
                image, age, filename = data
            #data_iter = tqdm(enumerate(dataloaders_[phase]), total=len(dataloaders_[phase]))
            #for i, (inputs, labels) in data_iter:
                inputs = image.to(device)
                labels = age.to(device).float()

                # zero the parameter gradients
                optimizer_.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model_(inputs)
                    loss = criterion_(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer_.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_error += torch.sum(torch.abs(outputs - labels))  # MAE
                running_corrects_3 += torch.sum(torch.abs(outputs - labels) < 3)  # CA 3
                running_corrects_5 += torch.sum(torch.abs(outputs - labels) < 5)  # CA 5
                # data_iter.set_description(
                #     f"{phase} Loss: {running_loss / ((i+1)*inputs.size(0)):.4f} MAE: {running_error / ((i+1)*inputs.size(0)):.4f} CA_3: {running_corrects_3.double() / ((i+1)*inputs.size(0)):.4f} CA_5: {running_corrects_5.double() / ((i+1)*inputs.size(0)):.4f}"
                # )

            epoch_loss = running_loss / len(dataloaders_[phase].dataset)
            MAE = running_error / len(dataloaders_[phase].dataset)
            CA_3 = running_corrects_3.double() / len(dataloaders_[phase].dataset)
            CA_5 = running_corrects_5.double() / len(dataloaders_[phase].dataset)


            print(
                f"{phase} Loss: {epoch_loss:.4f} MAE: {MAE:.4f} CA_3: {CA_3:.4f} CA_5: {CA_5:.4f}"
            )
            time_elapsed = time.time() - since
            print(
                "Complete in {:.0f}m {:.0f}s".format(
                    time_elapsed // 60, time_elapsed % 60
                )
            )

            save_best_model_path = os.path.join(save_model_dir,f'best_model.pth')
            # deep copy the model
            if phase == "val" and MAE < best_acc:
                best_acc = MAE
                torch.save(model_.state_dict(), save_best_model_path)
            if phase == "val":
                val_acc_history.append(MAE)
                print('eval dataset  mae: ', MAE)

        lr_scheduler.step(epoch)

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s\n\n".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val MAE: {:4f}".format(best_acc))

    # load best model weights
    model_.load_state_dict(best_model_wts)
    return model_, val_acc_history
print("Finished")

# Training and validation
num_epochs = 20
learning_rate = 0.0001  # originally 0.001
weight_decay = 1e-4  # originally 1e-4
load_pretrained = False

# 3. dataset 和 data loader, num_workers设置线程数目，pin_memory设置固定内存
img_size = 256
transform1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([img_size, img_size]),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.2, hue=0.1),
    #transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
    transforms.RandomRotation(degrees=(-90, 90)),
])
transform2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([img_size, img_size]),
])

dev = torch.device("cuda:2")
train_dataset = BatchDataset('train', transform1)
train_loader = DataLoader(train_dataset, batch_size=32*4, shuffle=True, num_workers=8, pin_memory=True)

eval_dataset = BatchDataset('eval', transform2)
eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)



model_to_train = SSRNet(image_size=img_size)

if load_pretrained:
    loaded_model = torch.load("")
    model_to_train.load_state_dict(loaded_model["state_dict"])


total_dataloader = {
    "train": train_loader,
    "val": eval_loader,
}

model_to_train = model_to_train.to(dev)
params_to_update = model_to_train.parameters()

optimizer = optim.Adam(params_to_update, lr=learning_rate, weight_decay=weight_decay)
criterion = nn.L1Loss()
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

model_to_train, hist = train_model(
    model_to_train,
    total_dataloader,
    criterion,
    optimizer,
    device = dev,
    num_epochs_=num_epochs,
)

torch.save(
    {
        "epoch": num_epochs,
        "state_dict": model_to_train.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
    f"SSRNet_Adam_L1Loss_LRDecay_weightDecay{weight_decay}_batch{batch_size}_lr{learning_rate}_epoch{num_epochs}+90_64x64.pth",
)