import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time 
import clip
import re
import os
import numpy as np

RED = "\033[31m"
RESET = "\033[0m"
BLUE = "\033[94m"

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "ViT-L/14@336px"
clip_model, preprocess = clip.load(model_name, device=device)
clip_model = clip_model.float()
txtprocess = clip.tokenize


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3):
        super(ResidualBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.shortcut = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return self.layer(x) + self.shortcut(x)

class EnhancedRegressionModelResidual(nn.Module):
    def __init__(self, input_dim=768):
        super(EnhancedRegressionModelResidual, self).__init__()
        self.network = nn.Sequential(
            ResidualBlock(input_dim, 512, 0.5),
            ResidualBlock(512, 256, 0.4),
            ResidualBlock(256, 128, 0.3),
            ResidualBlock(128, 64, 0.2),
            nn.Linear(64, 1)  # Output layer for regression
        )
    
    def forward(self, x):
        return self.network(x)

class RegressionDataset(Dataset):
    def __init__(self, age_info, image_dir, breed_info,  preprocess, txtpro, extrem=None, age_class=None):
        self.preprocess = preprocess
        self.image_paths = []
        self.ages= []
        self.txtprocess = txtpro
        self.breedtxt= {}
        self.age_classes = {}
        self.extrem = extrem

        a = 0
        if age_class!=None:
            with open(age_class, 'r') as f:
                for line in f:
                    line=line.strip()
                    parts=re.split(r'\s+',line)
                    if(len(parts)==2):
                        image_name, ageclass=parts 
                        path = os.path.join(image_dir, image_name)
                        if not os.path.exists(path):
                            a+=1
                            continue# You can also return a placeholder 
                        self.age_classes[path]= ageclass
                    else:
                        print("hello")
                print(f"{a} images not found")

        with open(age_info, 'r') as f:
            for line in f:
                line=line.strip()
                parts=re.split(r'\s+',line)
                if(len(parts)==2):
                    image_name, age =parts 
                    path = os.path.join(image_dir, image_name)
                    if not os.path.exists(path):
                        a+=1
                        continue# You can also return a placeholder 
                    if age_class == None:
                        if self.extrem == None:
                            self.image_paths.append(path)  # Add the full image path
                            self.ages.append(int(age))
                        elif self.extrem and (int(age)>100 or int(age)<20):
                            self.image_paths.append(path)  # Add the full image path
                            self.ages.append(int(age))
                        elif not self.extrem and  (int(age)<100 and int(age)>20):
                            self.image_paths.append(path)  # Add the full image path
                            self.ages.append(int(age))
                    else:            
                        if self.extrem == None:
                            self.image_paths.append(path)  # Add the full image path
                            self.ages.append(int(age))
                        elif self.extrem and int(self.age_classes[path])==1:
                            self.image_paths.append(path)  # Add the full image path
                            self.ages.append(int(age))
                        elif not self.extrem and int(self.age_classes[path])==0:
                            self.image_paths.append(path)  # Add the full image path
                            self.ages.append(int(age))
                else:                
                    print("hello")
            print(f"{a} images not found")
                                     
        with open(breed_info, 'r') as f:
            for line in f            :
                line=line            .strip()
                parts=re.split(r'\s+',line)
                if(len(parts)==2):
                    image_name, breed=parts 
                    path = os.path.join(image_dir, image_name)
                    if not os.path.exists(path):
                        a+=1
                        continue# You can also return a placeholder 
                    breed = self.txtprocess(breed)
                    self.breedtxt[path]=breed
                else:
                    print("hello")
            print(f"{a} images not found")

        


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        txt = self.breedtxt[img_path]
        # ageclass = self.age_classes[img_path]
        image = self.preprocess(image)
        target = self.ages[idx]
        return image, txt, target

metadata_file = 'annotations/smooth.txt'  # Path to your metadata file
image_dire = 'trainset'  # Directory where the images are stored
test_file = 'annotations/val.txt'
test_dir = 'valset'
ttxt = 'yolov5/atrain.txt'
tbtxt = 'yolov5/eval.txt'

# train_common_set = RegressionDataset(metadata_file, image_dire, ttxt, 'pre.txt', preprocess, txtprocess, False)
# train_extreme_set = RegressionDataset(metadata_file, image_dire, ttxt, 'pre.txt', preprocess, txtprocess, True)
train_set= RegressionDataset(metadata_file, image_dire, ttxt, preprocess, txtprocess)
test_set= RegressionDataset(test_file, test_dir, tbtxt, preprocess, txtprocess)
# print(len(train_set))
train_loader = DataLoader(train_set, batch_size=128, num_workers= 8,pin_memory=True, shuffle=True)
test_loader = DataLoader(test_set, batch_size=128, num_workers= 8,pin_memory=True, shuffle=True)

class CLIPMODEL(nn.Module):
    def __init__(self, input_dim=768):
        super(CLIPMODEL, self).__init__()
        self.clip_model = clip_model
        self.regression = EnhancedRegressionModelResidual(input_dim)
        for param in self.clip_model.parameters():
            param.requires_grad = False
    def forward(self, img, txt):
        image_features = self.clip_model.encode_image(img)
        text_features = self.clip_model.encode_text(txt)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # print(image_features)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        combined_features = torch.cat((image_features, text_features), dim=1)
        combined_features = combined_features.float()
        return self.regression(combined_features)

model = CLIPMODEL(1536).to(device)
# state = torch.load(f'weight/epoch_6.pth')
# model.load_state_dict(state)

for param in model.clip_model.visual.transformer.resblocks[-1].parameters():
    param.requires_grad = True

# Similarly, unfreeze the last transformer block of the text encoder if needed
for param in model.clip_model.transformer.resblocks[-1].parameters():
    param.requires_grad = True


optimizer = optim.Adam([
    {'params': model.clip_model.visual.transformer.resblocks[-1].parameters(), 'lr': 1e-5},
    {'params': model.clip_model.transformer.resblocks[-1].parameters(), 'lr': 1e-5},
    {'params': model.regression.parameters(), 'lr': 1e-3}
    # {'params': model.regression.parameters(), 'lr': 1e-4}
])

num_epochs = 50 
min = 1000
train_l = []
test_l = []
best_epoch = 0
for epoch in range(0, num_epochs):
    model.train()
    total_loss = 0
    total = 0
    print(len(train_loader))
    for images, txt, y in tqdm(train_loader,desc="processing"):
        images = images.to(device)
        txt = txt.to(device)
        txt = txt.squeeze(dim=1)
        y = y.float().to(device).unsqueeze(1)  # shape: [batch_size, 1]
        
        # Extract CLIP features without computing gradients for CLIP

        # Forward through regression head
        preds = model(images, txt)
        loss = torch.mean(torch.abs(preds-y))

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += (loss.item()*y.size(0))
        total += y.size(0)


    avg_loss = total_loss /total
    train_l.append(avg_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(),f'weight/epoch_{epoch+1}.pth')

    model.eval()
    with torch.no_grad():
        test_total = 0
        test_loss = 0.0
        for images, txt, y in tqdm(test_loader,desc="processing"):
            images = images.to(device)
            txt = txt.to(device)
            txt = txt.squeeze(dim=1)
            y = y.float().unsqueeze(1).to(device)  # shape: [batch_size, 1]
            
            # Extract CLIP features without computing gradients for CLIP
            # Forward through regression head
            preds = model(images, txt)

            loss = torch.mean(torch.abs(preds-y))
            test_loss+= (loss.item()*y.size(0))
            test_total+= y.size(0)

        avg_loss = test_loss/test_total
        test_l.append(avg_loss)
        if avg_loss<min:
            min = avg_loss
            best_epoch = epoch +1
        print(f"{RED}Epoch {epoch+1}/{num_epochs}, test Loss: {avg_loss:.4f}{RESET}")



                                   