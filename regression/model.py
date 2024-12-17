import torch
import torch.nn as nn
from torchinfo import summary
from torchvision import models
from ptflops import get_model_complexity_info
import timm
from densenet import DenseNet
from safetensors.torch import load_file
import torch.nn.functional as F
import random


def print_shapes(model, input_tensor):
    for name, layer in model.named_children():
        input_tensor = layer(input_tensor)
        print(f"{name}: {input_tensor.shape}")
    return input_tensor


class base_net(nn.Module):
    def __init__(self, input_features, num_features=64):
        super().__init__()
        self.num_features = num_features
        self.conv = nn.Sequential(
            nn.Conv2d(input_features, num_features, kernel_size=3, padding=3//2),
            #nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features*2, kernel_size=3, padding=3//2),
            #nn.BatchNorm2d(num_features*2),
            nn.ReLU(inplace=True),

            nn.Conv2d(num_features*2, num_features, kernel_size=3, padding=3 // 2),
            #nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2),
            #nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),

            nn.Conv2d(num_features, num_features, kernel_size=3, padding=3//2),
        )

    def forward(self, x):
        x = self.conv(x)

        return x
class Predictor(nn.Module):
    """ The header to predict age (regression branch) """

    def __init__(self, num_features, num_classes=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_features, num_features // 4, kernel_size=3, padding=3 // 2),
            nn.BatchNorm2d(num_features // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(num_features // 4, num_features // 8, kernel_size=3, padding=3 // 2),
            nn.BatchNorm2d(num_features // 8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(num_features // 8, num_features // 16, kernel_size=3, padding=3 // 2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(num_features//16, num_classes, kernel_size=1, bias=True)
        #self.dp = nn.Dropout(0.5)
        self.channel_attention = ChannelAttention(num_features // 16)
        self.spatial_attention = SpatialAttention(kernel_size=7)
    def forward(self, x):
        #print(x.shape)  # 打印查看张量形状
        x = self.conv(x)
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        x = self.gap(x)
        #x = self.dp(x)
        x = self.fc(x)
        x = x.squeeze(-1).squeeze(-1).squeeze(-1)
        return x



class LPredictor(nn.Module):
    """ The header to predict age (regression branch) """

    def __init__(self, num_features, num_classes=1):
        super().__init__()
        # 用一个卷积层处理特征
        self.conv = nn.Conv2d(num_features, num_features // 4, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm2d(num_features // 4)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        
        # 自适应平均池化，输出形状为 [batch_size, channels, 1, 1]
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # 输出层（将卷积后的特征映射到预测值）
        self.fc = nn.Conv2d(num_features // 4, num_classes, kernel_size=1, bias=True)
    
    def forward(self, x):
        # 卷积 + BatchNorm + ReLU + Dropout
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # 全局平均池化
        x = self.gap(x)
        
        # 通过全连接层得到最终输出
        x = self.fc(x)
        
        # 去除多余的维度，得到最终的预测值
        x = x.squeeze(-1).squeeze(-1).squeeze(-1)
        
        return x

class Classifier(nn.Module):
    """ The header to predict gender (classification branch) """

    def __init__(self, num_features, num_classes=100):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_features, num_features // 4, kernel_size=3, padding=3 // 2),
            nn.BatchNorm2d(num_features // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(num_features // 4, num_features // 8, kernel_size=3, padding=3 // 2),
            nn.BatchNorm2d(num_features // 8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(num_features // 8, num_features // 16, kernel_size=3, padding=3 // 2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(num_features//16, num_classes, kernel_size=1, bias=True)
        self.dp = nn.Dropout(0.4)


    def forward(self, x):
        x = self.conv(x)
        x = self.gap(x)

        x = self.dp(x)
        x = self.fc(x)
        x = x.squeeze(-1).squeeze(-1)
        # x = nn.functional.softmax(x, dim=1)
        return x

#https://github.com/NICE-FUTURE/predict-gender-and-age-from-camera/tree/master
class Model(nn.Module):
    """ A model to predict age and gender """

    def __init__(self, timm_pretrained=True):
        super().__init__()
        safetensors_path = "./weights/resnet18/model.safetensors"
        self.backbone = timm.create_model("resnet18", pretrained=False)
        #self.backbone = models.resnet18(pretrained=True)  # 不加载预训练权重
        if safetensors_path:
            self.load_safetensors(safetensors_path)
        self.predictor = Predictor(self.backbone.num_features)
        # self.classifier = Classifier(self.backbone.num_features)


    def forward(self, x):

        x = self.backbone.forward_features(x)  # shape: B, D, H, W
        age = self.predictor(x)
        #gender = self.classifier(x)

        return age
    
    def load_safetensors(self, safetensors_path):
        # 加载 safetensors 文件
        state_dict = load_file(safetensors_path)

        # 将 safetensors 文件加载到模型中
        self.backbone.load_state_dict(state_dict, strict=False)

        print(f"Successfully loaded weights from {safetensors_path}")


class Model34(nn.Module):
    """ A model to predict age and gender """

    def __init__(self, timm_pretrained=True):
        super().__init__()
        safetensors_path = "./weights/resnet34/model.safetensors"
        self.backbone = timm.create_model("resnet34", pretrained=False)
        #self.backbone = models.resnet18(pretrained=True)  # 不加载预训练权重
        if safetensors_path:
            self.load_safetensors(safetensors_path)
        self.predictor = Predictor(self.backbone.num_features)
        # self.classifier = Classifier(self.backbone.num_features)


    def forward(self, x):

        x = self.backbone.forward_features(x)  # shape: B, D, H, W
        age = self.predictor(x)
        #gender = self.classifier(x)

        return age
    
    def load_safetensors(self, safetensors_path):
        # 加载 safetensors 文件
        state_dict = load_file(safetensors_path)

        # 将 safetensors 文件加载到模型中
        self.backbone.load_state_dict(state_dict, strict=False)

        print(f"Successfully loaded weights from {safetensors_path}")

class Model50(nn.Module):
    """ A model to predict age and gender """

    def __init__(self, timm_pretrained=True):
        super().__init__()
        safetensors_path = "./weights/resnet50/model.safetensors"
        self.backbone = timm.create_model("resnext50_32x4d.a1h_in1k", pretrained=False)
        #self.backbone = models.resnet18(pretrained=True)  # 不加载预训练权重
        if safetensors_path:
            self.load_safetensors(safetensors_path)
        self.predictor = Predictor(self.backbone.num_features)
        # self.classifier = Classifier(self.backbone.num_features)


    def forward(self, x):

        x = self.backbone.forward_features(x)  # shape: B, D, H, W
        age = self.predictor(x)
        #gender = self.classifier(x)

        return age
    
    def load_safetensors(self, safetensors_path):
        # 加载 safetensors 文件
        state_dict = load_file(safetensors_path)

        # 将 safetensors 文件加载到模型中
        self.backbone.load_state_dict(state_dict, strict=False)

        print(f"Successfully loaded weights from {safetensors_path}")


class ModelReg320(nn.Module):
    """ A model to predict age and gender """

    def __init__(self, timm_pretrained=True):
        super().__init__()
        safetensors_path = "./weights/regnetx_320/model.safetensors"
        self.backbone = timm.create_model("regnetx_320.tv2_in1k", pretrained=False)
        #self.backbone = models.resnet18(pretrained=True)  # 不加载预训练权重
        if safetensors_path:
            self.load_safetensors(safetensors_path)
        self.predictor = Predictor(self.backbone.num_features)
        # self.classifier = Classifier(self.backbone.num_features)


    def forward(self, x):
        #print(x.shape)
        x = self.backbone.forward_features(x)  # shape: B, D, H, W
        #print(x.shape)
        age = self.predictor(x)
        #gender = self.classifier(x)
        #print(age.shape)
        
        return age
    
    def load_safetensors(self, safetensors_path):
        # 加载 safetensors 文件
        state_dict = load_file(safetensors_path)

        # 将 safetensors 文件加载到模型中
        self.backbone.load_state_dict(state_dict, strict=False)

        print(f"Successfully loaded weights from {safetensors_path}")

    
class Dino(nn.Module):
    """ A model to predict age and gender """

    
    def __init__(self, timm_pretrained=True,freeze_layers=False,num_frozen_layers=3):
        super().__init__()
        safetensors_path = "./weights/vit_small_dinov2/model.safetensors"
        self.backbone = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False)
        #self.backbone = models.resnet18(pretrained=True)  # 不加载预训练权重
        if safetensors_path:
            self.load_safetensors(safetensors_path)
        
        if freeze_layers:
            self.freeze_backbone_layers(num_frozen_layers)
        self.predictor = Predictor(self.backbone.num_features)
        # self.classifier = Classifier(self.backbone.num_features)


    def forward(self, x):

        x = self.backbone.forward_features(x)  # shape: B, D, H, W
        y = x.shape[0]
        x = x.permute(0, 2, 1)  # 维度顺序调整为 [128, 384, 1370]
        x = x[:, :, :1369]
        #print(x.shape)
        x = x.reshape(y, 384, 37, 37)
        age = self.predictor(x)
        #gender = self.classifier(x)

        return age
    
    def load_safetensors(self, safetensors_path):
        # 加载 safetensors 文件
        state_dict = load_file(safetensors_path)

        # 将 safetensors 文件加载到模型中
        self.backbone.load_state_dict(state_dict, strict=False)

        print(f"Successfully loaded weights from {safetensors_path}")

    def freeze_backbone_layers(self, num_frozen_layers):
        """
        冻结前 num_frozen_layers 个模块的参数。
        """
        frozen_count = 0
        for name, module in self.backbone.named_children():
            if frozen_count >= num_frozen_layers:
                break
            for param in module.parameters():
                param.requires_grad = False
            frozen_count += 1
        print(f"Froze {frozen_count} layers in the backbone.")


class ModelCLIP(nn.Module):

    def __init__(self, timm_pretrained=True,freeze_layers=True,num_frozen_layers=8):
        super().__init__()
        safetensors_path = "./weights/clip/model.safetensors"
        self.backbone = timm.create_model("vit_large_patch14_clip_224.openai_ft_in12k_in1k", pretrained=False)
        #self.backbone = models.resnet18(pretrained=True)  # 不加载预训练权重
        if safetensors_path:
            self.load_safetensors(safetensors_path)
        
        if freeze_layers:
            self.freeze_backbone_layers(num_frozen_layers)
        self.predictor = Predictor(257)
        # self.classifier = Classifier(self.backbone.num_features)


    def forward(self, x):
        x = self.backbone.forward_features(x)  # shape: B, D, H, W
        y = x.shape[0]
        x = x.reshape(y, 257, 32, 32)
        # print(self.backbone.num_features)
        # print(x.shape)
        age = self.predictor(x)
        #gender = self.classifier(x)

        return age
    
    def load_safetensors(self, safetensors_path):
        # 加载 safetensors 文件
        state_dict = load_file(safetensors_path)

        # 将 safetensors 文件加载到模型中
        self.backbone.load_state_dict(state_dict, strict=False)

        print(f"Successfully loaded weights from {safetensors_path}")

    def freeze_backbone_layers(self, num_frozen_layers):
        """
        冻结前 num_frozen_layers 个模块的参数。
        """
        frozen_count = 0
        for name, module in self.backbone.named_children():
            if frozen_count >= num_frozen_layers:
                break
            for param in module.parameters():
                param.requires_grad = False
            module.eval()  # 设置为 eval 模式
            frozen_count += 1
        print(f"Froze {frozen_count} layers in the backbone.")
    
models_dict = {
    'CLIP': lambda: ModelCLIP(),
    'reg320': lambda: ModelReg320(),
    'resnet18': lambda: Model(),
    'resnet34': lambda: Model34(),
    'resnet50': lambda: Model50(),
    'dino': lambda: Dino(),
}

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)

    def forward(self, x):
        # Global average pooling
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)

        # Channel-wise attention
        avg_out = self.fc2(F.relu(self.fc1(avg_pool)))
        max_out = self.fc2(F.relu(self.fc1(max_pool)))

        # Combine the results
        out = avg_out + max_out
        out = torch.sigmoid(out)
        return out * x  # Apply attention to the input

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_pool, max_pool], dim=1)
        spatial_attention_map = self.conv(concat)
        spatial_attention_map = torch.sigmoid(spatial_attention_map)
        return spatial_attention_map * x  # A


class DualModel(nn.Module):
    def __init__(self, MainModel, DualModel, MainWeight, DualWeight):
        super().__init__()
        print("1")
        # 假设models字典已经定义，里面有预训练好的模型
        self.Main = models_dict[MainModel]().backbone  # Main model
        self.LeftTop = models_dict[DualModel]().backbone  # 左上分支模型
        self.LeftBot = models_dict[DualModel]().backbone  # 左下分支模型
        self.RightTop = models_dict[DualModel]().backbone  # 右上分支模型
        self.RightBot = models_dict[DualModel]().backbone  # 右下分支模型
        self.fc = LPredictor(5)  # 输入5个通道的特征
        # 权重初始化
        self.MainW = MainWeight
        self.DualW = DualWeight
        self.conv = nn.Sequential(
            # 第1层卷积：输入 [128, 1000] -> [128, 32, 1000]
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),  # 输出: [128, 32, 1000]
            nn.BatchNorm1d(32),  # 添加 BatchNorm
            nn.ReLU(inplace=True),
            
            # 第2层卷积：进一步提取特征，输出 [128, 32, 1000]
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1),  # 输出: [128, 32, 1000]
            nn.BatchNorm1d(32),  # 添加 BatchNorm
            nn.ReLU(inplace=True),
            
            # 第3层卷积：逐渐减小空间维度
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1),  # 输出: [128, 32, 1000]
            nn.BatchNorm1d(32),  # 添加 BatchNorm
            nn.ReLU(inplace=True),
            
            # 第4层卷积：压缩空间尺寸，输出 [128, 32, 32]
            nn.Conv1d(32, 32, kernel_size=31, stride=31, padding=0),  # 输出: [128, 32, 32]
            nn.BatchNorm1d(32),  # 添加 BatchNorm
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        # 主模型的预测
        main_output = self.Main(x)
        
        # 获取每个区域的切割，分割后调整大小为256x256
        left_top = x[:, :, :x.shape[2]//2, :x.shape[3]//2]
        left_bot = x[:, :, x.shape[2]//2:, :x.shape[3]//2]
        right_top = x[:, :, :x.shape[2]//2, x.shape[3]//2:]
        right_bot = x[:, :, x.shape[2]//2:, x.shape[3]//2:]
        # Resize所有区域到256x256
        left_top_resized = F.interpolate(left_top, size=(256, 256), mode='bilinear', align_corners=False)
        left_bot_resized = F.interpolate(left_bot, size=(256, 256), mode='bilinear', align_corners=False)
        right_top_resized = F.interpolate(right_top, size=(256, 256), mode='bilinear', align_corners=False)
        right_bot_resized = F.interpolate(right_bot, size=(256, 256), mode='bilinear', align_corners=False)

        # 分支模型的预测
        left_top_output = self.conv(self.LeftTop(left_top_resized).unsqueeze(1))
        left_bot_output = self.conv(self.LeftBot(left_bot_resized).unsqueeze(1))
        right_top_output = self.conv(self.RightTop(right_top_resized).unsqueeze(1))
        right_bot_output = self.conv(self.RightBot(right_bot_resized).unsqueeze(1))
        main_output =self.conv(main_output.unsqueeze(1))
        # 将五个分支的输出合并成一个5通道的特征图
        combined_output = torch.cat([main_output.unsqueeze(1), left_top_output.unsqueeze(1),
                                    left_bot_output.unsqueeze(1), right_top_output.unsqueeze(1),
                                    right_bot_output.unsqueeze(1)], dim=1)
        # 将合并后的特征传递给 Predictor
        final_output = self.fc(combined_output)
        
        return final_output

class ModelMix(nn.Module):
    def __init__(self, device1,device2):
        super().__init__()
        safetensors_path = "./weights/regnetx_320/model.safetensors"
        path2 = "./weights/resnet50/model.safetensors"
        self.branch2 = timm.create_model("regnetx_320.tv2_in1k", pretrained=False)
        self.branch1 = timm.create_model("resnext50_32x4d.a1h_in1k", pretrained=False)
        #self.backbone = models.resnet18(pretrained=True)  # 不加载预训练权重
        if safetensors_path:
            self.load_safetensors(safetensors_path,path2)
        self.predictor = Predictor(self.branch1.num_features)
        # self.classifier = Classifier(self.backbone.num_features)
        self.classifier = Classifier(self.branch2.num_features, 4)  # 4 age ranges

    def forward(self, x):
        #print(x.shape)
        x1 = self.branch1.forward_features(x)  # shape: B, D, H, W
        x2 = self.branch2.forward_features(x)
        #print(x.shape)
        age = self.predictor(x1)
        #gender = self.classifier(x)

        age_range_logits = self.classifier(x2)  # shape: B, 4
        age_range_probs = F.softmax(age_range_logits, dim=1)  # shape: B, 4
        #print(age.shape)
        
        return age,age_range_probs
    
    def load_safetensors(self, safetensors_path,path2):
        # 加载 safetensors 文件
        state_dict = load_file(safetensors_path)
        state_dict2=load_file(path2)

        # 将 safetensors 文件加载到模型中
        self.branch1.load_state_dict(state_dict, strict=False)
        self.branch2.load_state_dict(state_dict2,strict=False)
        print(f"Successfully loaded weights from {safetensors_path}")

def Breed50(pretrained=True,device="cuda",weights='IMAGENET1K_V1',weights_path='./results/Breed50/best_model.pth'):
    model = models.resnet50(weights=weights)  # 现在使用weights参数
    def set_untrainable(layer):
        for p in layer.parameters():
            p.requires_grad = False

    # for layer in model.children():
    #     layer.apply(set_untrainable)
    model.fc = nn.Sequential(model.fc, nn.Dropout(p=.5), nn.ReLU(inplace=True), nn.Linear(1000, 120))
    if weights_path:
        # Load custom weights if the path is provided
        print(f"Loading weights from {weights_path}")
        model.load_state_dict(torch.load(weights_path))
        print("Custom weights loaded.")
    #model.to(device)
    model.eval()
    return model

class BreedDual(nn.Module):
    def __init__(self, timm_pretrained=True):
        super().__init__()
        safetensors_path = "./weights/regnetx_320/model.safetensors"
        self.main = timm.create_model("regnetx_320.tv2_in1k", pretrained=False)
        self.breed = Breed50()
        #self.backbone = models.resnet18(pretrained=True)  # 不加载预训练权重
        if safetensors_path:
            self.load_safetensors(safetensors_path)
        self.predictor = Predictor(self.main.num_features+360)
        # self.classifier = Classifier(self.backbone.num_features)


    def forward(self, x):
        #print(x.shape)
        x_resized = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        breed = self.breed(x_resized)
        #print(breed.shape)
        #print(x.shape)
        x1 = self.main.forward_features(x)  # shape: B, D, H, W
        #print(x.shape)
        breed = breed.repeat(1,3)
        breed_expanded = breed.unsqueeze(-1).unsqueeze(-1)  # 扩展维度以匹配空间大小, shape: [B, 120, 1, 1]
        breed_expanded = breed_expanded.repeat(1, 1, x1.size(2), x1.size(3))  # 重复以匹配主模型的空间大小
        
        # 合并特征
        combined = torch.cat((x1, breed_expanded), dim=1)  # 在通道维度上拼接, shape: [B, D+120, H, W]
        
        #gender = self.classifier(x)
        #print(age.shape)
        age = self.predictor(combined)  # 假设通过结合后的特征进行年龄预测
        return age

    # def forward(self, x):
    #     #print(x.shape)
    #     x_resized = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
    #     breed = self.breed(x_resized)
    #     #print(breed.shape)
    #     #print(x.shape)
    #     breed_expanded = breed.unsqueeze(2).unsqueeze(3)  # [B, 120] -> [B, 120, 1,1]
    #     breed_expanded = breed_expanded.expand(-1, -1, 256, 256)   # 扩展为 [B, 1, 256, 256]
    #     breed_expanded = breed_expanded.mean(dim=1, keepdim=True)  # 将 120 维的维度聚合到一起，保持最后维度
    #     breed_expanded = breed_expanded.expand(-1, 3, -1, -1)  # 将 breed_expanded 扩展为 [B, 3, 256, 256]
    #     #print(breed_expanded.shape)
    #     #print(breed_expanded)
    #     combined = x + 5*breed_expanded  # shape: [B, 3, 256, 256]
    #     #print(combined.shape)
    #     x1 = self.main.forward_features(combined)  # shape: B, D, H, W
    #     #print(x.shape)

        
    #     # 合并特征
        
    #     #gender = self.classifier(x)
    #     #print(age.shape)
    #     age = self.predictor(x1)  # 假设通过结合后的特征进行年龄预测
    #     return age
    
    def load_safetensors(self, safetensors_path):
        # 加载 safetensors 文件
        state_dict = load_file(safetensors_path)

        # 将 safetensors 文件加载到模型中
        self.main.load_state_dict(state_dict, strict=False)

        print(f"Successfully loaded weights from {safetensors_path}")

class AgeClassifer(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(weights='IMAGENET1K_V1')
        self.breed = Breed50()
        safetensors_path=None
        if safetensors_path:
            self.load_safetensors(safetensors_path)
        self.classifer = nn.Linear(1360, 2)
    def forward(self,x):
        x_resized = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        breed = self.breed(x_resized)
        x1 = self.backbone(x)
        breed = breed.repeat(1,3)
        combined = torch.cat((x1, breed), dim=1)  # 在通道维度上拼接, shape: [B, D+120, H, W]
        inv = self.classifer(combined)

        return inv


    def load_safetensors(self, safetensors_path):
        # 加载 safetensors 文件
        state_dict = load_file(safetensors_path)

        # 将 safetensors 文件加载到模型中
        self.backbone.load_state_dict(state_dict, strict=False)

        print(f"Successfully loaded weights from {safetensors_path}")

class CRegnet(nn.Module):
    def __init__(self, timm_pretrained=True):
        super().__init__()
        safetensors_path = "./weights/regnetx_320/model.safetensors"
        self.backbone = timm.create_model("regnetx_320.tv2_in1k", pretrained=False)
        #self.backbone = models.resnet18(pretrained=True)  # 不加载预训练权重
        if safetensors_path: 
            self.load_safetensors(safetensors_path)
        self.classifer = Classifier(self.backbone.num_features,num_classes=4)
        # self.classifier = Classifier(self.backbone.num_features)


    def forward(self, x):
        #print(x.shape)
        x = self.backbone.forward_features(x)  # shape: B, D, H, W
        #print(x.shape)
        age = self.classifer(x)
        #gender = self.classifier(x)
        #print(age.shape)
        
        return age
    
    def load_safetensors(self, safetensors_path):
        # 加载 safetensors 文件
        state_dict = load_file(safetensors_path)

        # 将 safetensors 文件加载到模型中
        self.backbone.load_state_dict(state_dict, strict=False)

        print(f"Successfully loaded weights from {safetensors_path}")


class RegAgeClassifer(nn.Module):
    def __init__(self):
        super().__init__()
        safetensors_path = "./weights/regnetx_320/model.safetensors"
        self.backbone = timm.create_model("regnetx_320.tv2_in1k", pretrained=False)
        #self.backbone = models.resnet18(pretrained=True)  # 不加载预训练权重
        if safetensors_path:
            self.load_safetensors(safetensors_path)
        self.breed = Breed50()
        safetensors_path=None
        self.classifer = Classifier(self.backbone.num_features+360,2)
    def forward(self,x):
        x_resized = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        breed = self.breed(x_resized)
        x1 = self.backbone.forward_features(x)
        
        breed = breed.repeat(1,3)
        breed_expanded = breed.unsqueeze(-1).unsqueeze(-1)  # 扩展维度以匹配空间大小, shape: [B, 120, 1, 1]
        breed_expanded = breed_expanded.repeat(1, 1, x1.size(2), x1.size(3))  # 重复以匹配主模型的空间大小
        combined = torch.cat((x1, breed_expanded), dim=1)  # 在通道维度上拼接, shape: [B, D+120, H, W]
        inv = self.classifer(combined)

        #print(inv.shape)

        return inv


    def load_safetensors(self, safetensors_path):
        # 加载 safetensors 文件
        state_dict = load_file(safetensors_path)

        # 将 safetensors 文件加载到模型中
        self.backbone.load_state_dict(state_dict, strict=False)

        print(f"Successfully loaded weights from {safetensors_path}")

class OptimModel(nn.Module):
     def __init__(self,device1,device2,device3):
        super().__init__()
        self.classifier = AgeClassifer().to(device1)
        self.main = BreedDual().to(device2)
        self.sup = BreedDual().to(device3)
        self.load_safetensors()
        self.device=[device1,device2,device3]

     def load_safetensors(self):
        # 加载 safetensors 文件
        Age_classifier_state_dict = torch.load("./results/AgeClassify/best_model.pth")
        Middel_state_dict=torch.load("./results/Breed/MiddleBreed/best_model.pth")
        Other_state_dict=torch.load("./results/Breed/OtherBreed/best_model.pth")
        # 将 safetensors 文件加载到模型中
        self.classifier.load_state_dict(Age_classifier_state_dict, strict=False)
        self.main.load_state_dict(Middel_state_dict,strict=False)
        self.sup.load_state_dict(Other_state_dict,strict=False)
        print(f"Successfully loaded weights ")
     def forward(self, x,age):
        #print(x.shape)
        pred_logits = self.classifier(x.to(self.device[0]))  # 获取模型的logits输出
        _, predicted = torch.max(pred_logits, 1)  # 获取每个样本的最大值索引作为预测类别
        true_labels = ((age > 20) & (age < 90)).long()  # 转换为 long 类型

        # 统计预测正确的样本数量
        correct_predictions = (predicted == true_labels).sum().item()
        total_samples = len(age)
        accuracy = correct_predictions / total_samples * 100

        print(f"Classification Accuracy: {accuracy:.2f}%")
        # error_prob = 0.1
        # random_value = random.random()<error_prob
        # flip = (age>20 and age <90) ^ random_value
        pred_main = self.main(x.to(self.device[1])).to(self.device[0])
        pred_main = torch.clamp(pred_main, min=20, max=90)
        pred_sup = self.sup(x.to(self.device[2])).to(self.device[0])
        pred_sup = torch.where((pred_sup > 20) & (pred_sup <= 55), 
                           torch.tensor(20, device=self.device[0]), 
                           pred_sup)
        pred_sup = torch.where( (pred_sup > 55) & (pred_sup < 90), 
                           torch.tensor(92, device=self.device[0]), 
                           pred_sup)
        pred = torch.where(predicted.bool(), pred_main, pred_sup)
        # if(predicted.item()):
        #     age =age.to(self.device[1])
        #     mask_main = (age > 20) & (age < 90)
        #     pred = self.main(x.to(self.device[1]))
        #     pred = torch.where(mask_main, 
        #                    torch.clamp(pred, min=20, max=90), 
        #                    pred)

        # else:
        #     age = age.to(self.device[2])
        #     mask_main = (age > 20) & (age < 90)
        #     pred = self.sup(x.to(self.device[2]))
        #     pred = torch.where((~mask_main) & (pred > 20) & (pred <= 55), 
        #                    torch.tensor(20, device=self.device[2]), 
        #                    pred)
        #     pred = torch.where((~mask_main) & (pred > 55) & (pred < 90), 
        #                    torch.tensor(90, device=self.device[2]), 
        #                    pred)
        return pred
     
class densnet(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = DenseNet()
        self.predictor = Predictor(188)

    def forward(self, x):
        x = self.backbone.forward(x)  # shape: B, D, H, W
        age = self.predictor(x)
        return age


class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        return torch.mean(diff.abs() + torch.log1p(torch.exp(-2 * diff.abs())) - torch.log(torch.tensor(2.0)))
    
class HuberLoss:
    def __init__(self, delta=50):
        self.delta = delta  # 设置阈值为75

    def __call__(self, y_pred, y_true):
        # 计算Huber损失
        return F.smooth_l1_loss(y_pred, y_true, beta=self.delta, reduction='mean')

def Mix_loss(predicted_age, true_age, age_range_probs,R_loss, alpha=1.0, beta=1.0):
    """
    Compute combined loss: Classification loss + Regression loss
    - Classification loss: Cross-Entropy Loss for age range prediction
    - Regression loss: MSE Loss for actual age prediction
    """
        # 根据 true_age 的批量数据确定真实的年龄区间标签，并转换为 one-hot 编码
    true_age_range = torch.zeros(true_age.size(0), 4, device=age_range_probs.device)  # 初始化四个类别的one-hot编码
    
    # 条件判断生成 one-hot 编码
    true_age_range[true_age < 30] = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=age_range_probs.device)  # 0类
    true_age_range[(30 <= true_age) & (true_age <= 80)] = torch.tensor([0, 1, 0, 0], dtype=torch.float32, device=age_range_probs.device)  # 1类
    true_age_range[(81 <= true_age) & (true_age <= 120)] = torch.tensor([0, 0, 1, 0], dtype=torch.float32, device=age_range_probs.device)  # 2类
    true_age_range[true_age > 120] = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=age_range_probs.device)  # 3类

    # Classification loss (cross-entropy loss for age range)
    # 使用真实的 one-hot 编码标签和 age_range_probs 来计算分类损失
    classification_loss = 50* F.binary_cross_entropy_with_logits(age_range_probs, true_age_range)
    print(classification_loss)
    # Regression loss (MSE loss for actual age prediction)
    regression_loss = R_loss(predicted_age, true_age)
    
    # Combined loss
    total_loss = alpha * classification_loss + beta * regression_loss
    return total_loss
# if __name__ == "__main__":
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # device = 'cpu'
#     print(device)
#     modelviz = Model2().to(device)
#     # 打印模型结构
#     print(modelviz)
#     summary(modelviz, input_size=(2, 3, 256, 256), col_names=["kernel_size", "output_size", "num_params", "mult_adds"])
#     # for p in modelviz.parameters():
#     #     if p.requires_grad:
#     #         print(p.shape)

#     input = torch.rand(2, 3, 256, 256).to(device)
#     out = modelviz(input)




#     macs, params = get_model_complexity_info(modelviz, (3, 256, 256), verbose=True, print_per_layer_stat=True)
#     print(macs, params)
#     params = float(params[:-3])
#     macs = float(macs[:-4])

#     print(macs * 2, params)  # 8个图像的 FLOPs, 这里的结果 和 其他方法应该一致
#     print('out:', out.shape, out)


