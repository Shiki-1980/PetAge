
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
from model import Model,densnet,ModelReg320,ModelCLIP,Dino,HuberLoss,LogCoshLoss,DualModel,Mix_loss,ModelMix,BreedDual,AgeClassifer,RegAgeClassifer
import torchvision
import torch.nn.functional as F

def load_safetensors(self, safetensors_path):
        # 加载 safetensors 文件
        state_dict = load_file(safetensors_path)

        # 将 safetensors 文件加载到模型中
        self.main.load_state_dict(state_dict, strict=False)

        print(f"Successfully loaded weights from {safetensors_path}")

if __name__ == "__main__":
    # 1.当前版本信息
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.backends.cudnn.version())
    print(torch.cuda.get_device_name(0))

    checkpoint_path="./results/RegAge/first/best_model.pth"
    np.random.seed(0)
    torch.cuda.manual_seed_all(0)
    torch.manual_seed(3407)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 2. 设置device信息 和 创建model
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    # device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    device = torch.device('cuda:5')
    #model = DualModel('resnet34','resnet34',0.6,0.1)
    model=AgeClassifer()
    # for layer in model.modules():
    #     print(layer)
    #gpus = [6,7]
    #model = nn.DataParallel(model, device_ids=gpus)
    #checkpoint = torch.load(checkpoint_path)
    #model.load_state_dict(checkpoint)
    model = model.to(device)

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
    train_dataset = BatchDataset('train', transform1)
    train_dataset_loader = DataLoader(train_dataset, batch_size=32*4, shuffle=True, num_workers=8, pin_memory=True)

    eval_dataset = BatchDataset('eval', transform2)
    eval_dataset_loader = DataLoader(eval_dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)
    print('load dataset !', len(train_dataset), len(eval_dataset))

    # 4. 损失函数 和  优化器
    age_criterion = nn.MSELoss()
    gender_criterion = nn.CrossEntropyLoss().to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    #loss_fn = nn.L1Loss().to(device)
    #loss_fn = nn.SmoothL1Loss().to(device)
    #loss_fn = HuberLoss()

    learning_rate = 1 * 1e-4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # lr_step = 50
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_step, gamma=0.5)

    # 5. hyper para 设置
    epochs = 200

    save_epoch = 10
    save_model_dir = './results/RegAge/second'

    eval_epoch = 100
    save_sample_dir = './results/RegAge/second'
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    # 6. 是否恢复模型
    resume = 0
    last_epoch = 0
    if resume and last_epoch > 1:
        model.load_state_dict(torch.load(
            save_model_dir + '/checkpoint_%04d.pth' % (last_epoch),
            map_location=device))
        print('resume ' , save_model_dir + '/checkpoint_%04d.pth' % (last_epoch))

    # 7. 训练epoch

    f1 = open('./results/RegAge/second/traininfo.txt', 'a')
    f2 = open('./results/RegAge/second/evalinfo.txt', 'a')
    max_acc = 0
    
    for epoch in range(last_epoch + 1, epochs + 1):
        if epoch %50==0:
            learning_rate = learning_rate *0.5
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        
        print('current epoch:', epoch, 'current lr:', optimizer.state_dict()['param_groups'][0]['lr'])
        if epoch < last_epoch + 101:
            save_epoch = 10
            eval_epoch = 1
        else:
            save_epoch = 10
            eval_epoch = 1
        # 8. train loop
        model.train()
        g_loss = []
        g_mae = []
        correct_predictions = 0
        total_samples = 0
        for data in tqdm(train_dataset_loader):
            image, age, filename = data
            # print(image.shape, age, filename)
            image = image.to(device)
            age = age.to(device)
            llabels = torch.where((age > 20) & (age < 90), torch.tensor(1, device=device), torch.tensor(0, device=device)).long()
            labels = F.one_hot(llabels, num_classes=2).float()  # 转换为 one-hot 向量 [B, 2]

            pred_logits = model(image)  # 获取模型的logits输出
            loss = loss_fn(pred_logits, labels)  # 使用交叉熵损失

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # training result
            g_loss.append(loss.item())
            _, predicted = torch.max(pred_logits, 1)  # 获取每个样本的最大值索引作为预测类别
            correct_predictions += (predicted == llabels).sum().item()
            total_samples += labels.size(0)
            #print( loss.item(), mae)
        #print(len(g_loss), len(g_mae))
        mean_loss = np.mean(g_loss)
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        print(f'Epoch {epoch:04d}, Train Loss: {mean_loss:.4f}, Train Accuracy: {accuracy:.4f}')
        f1.write("%d, %.6f, %.6f\n" % (epoch, mean_loss, accuracy))

        # 9. save model
        if epoch % save_epoch == 0:
            save_model_path = os.path.join(save_model_dir, f'checkpoint_{epoch:04d}.pth')
            torch.save(model.state_dict(), save_model_path)
        # 10. eval test and save some samples if needed
        save_best_model_path = os.path.join(save_model_dir,f'best_model.pth')
        if epoch % eval_epoch == 0:
            model.eval()
            maes = []
            with torch.no_grad():
                test_correct_predictions = 0
                test_total_samples = 0
                for data in  tqdm(eval_dataset_loader):
                    image, age, filename = data
                    image = image.to(device)
                    age = age.to(device)
                    labels = torch.where((age > 20) & (age < 90), torch.tensor(1, device=device), torch.tensor(0, device=device)).long()
                    pred_logits = model(image)  # 获取模型的logits输出
                    _, predicted = torch.max(pred_logits, 1)  # 获取每个样本的最大值索引作为预测类别
                    test_correct_predictions += (predicted == labels).sum().item()
                    #print("predicted:",predicted,"labels:",labels,"correct:",test_correct_predictions)
                    test_total_samples += labels.size(0)
                accuracy=  test_correct_predictions/test_total_samples
                if(accuracy >max_acc):
                    max_acc = accuracy
                    torch.save(model.state_dict(), save_best_model_path)
                print('Test accuracy: ',accuracy)
                f2.write("%d, %.6f\n" % (epoch,  accuracy))
        #scheduler.step()  # 更新学习率

