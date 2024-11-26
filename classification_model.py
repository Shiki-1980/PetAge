import glob
import os

import cv2
import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms

from tqdm import tqdm


# def VGG16_predict(img_path):
#     '''
#     Use pre-trained VGG-16 model to obtain index corresponding to
#     predicted ImageNet class for image at specified path

#     Args:
#         img_path: path to an image

#     Returns:
#         Index corresponding to VGG-16 model's prediction
#     '''
#     # define VGG16 model
#     VGG16 = models.vgg16(pretrained=True)
#     # check if CUDA is available
#     use_cuda = torch.cuda.is_available()
#     # move model to GPU if CUDA is available
#     if use_cuda:
#         VGG16 = VGG16.cuda()
#     ## Load and pre-process an image from the given img_path
#     ## Return the *index* of the predicted class for that image
#     # Image Resize to 256
#     image = Image.open(img_path)
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]
#     image_transforms = transforms.Compose([transforms.Resize(256),
#                                            transforms.CenterCrop(224),
#                                            transforms.ToTensor(),
#                                            transforms.Normalize(mean, std)])
#     image_tensor = image_transforms(image)
#     image_tensor.unsqueeze_(0)
#     if use_cuda:
#         image_tensor = image_tensor.cuda()
#     output = VGG16(image_tensor)
#     _, classes = torch.max(output, dim=1)
#     return classes.item()  # predicted class index

# ### returns "True" if a dog is detected in the image stored at img_path
# def dog_detector(img_path):
#     ## TODO: Complete the function.
#     class_dog=VGG16_predict(img_path)
#     return class_dog >= 151 and class_dog <=268 # true/false


# def resnet50_predict(img_path):
#     resnet50 = models.resnet50(pretrained=True)
#     use_cuda = torch.cuda.is_available()
#     if use_cuda:
#         resnet50.cuda()
#     image = Image.open(img_path)
#     mean=[0.485, 0.456, 0.406]
#     std=[0.229, 0.224, 0.225]
#     image_transforms = transforms.Compose([transforms.Resize(256),
#                                            transforms.CenterCrop(224),
#                                            transforms.ToTensor(),
#                                            transforms.Normalize(mean,std)])
#     image_tensor = image_transforms(image)
#     image_tensor.unsqueeze_(0)
#     if use_cuda:
#         image_tensor=image_tensor.cuda()
#     resnet50.eval()
#     output = resnet50(image_tensor)
#     _,classes = torch.max(output,dim=1)
#     return classes.item()
# def resnet50_dog_detector(image_path):
#     class_idx = resnet50_predict(image_path)
#     return class_idx >= 151 and class_idx <=268

device = torch.device('cuda:5')

def get_train_set_info(dir):
    dog_files_train = glob.glob(dir + '\\*.jpg')
    mean = np.array([0.,0.,0.])
    std = np.array([0.,0.,0.])
    for i in tqdm(range(len(dog_files_train))):
        image=cv2.imread(dog_files_train[i])
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = image/255.0
        mean[0] += np.mean(image[:,:,0])
        mean[1] += np.mean(image[:,:,1])
        mean[2] += np.mean(image[:,:,2])
        std[0] += np.std(image[:,:,0])
        std[1] += np.std(image[:,:,1])
        std[2] += np.std(image[:,:,2])
    mean = mean/len(dog_files_train)
    std = std/len(dog_files_train)
    return mean,std

from PIL import ImageFile


def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf

    for epoch in range(1, n_epochs + 1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(tqdm(loaders['train'])):
            # move to GPU
            if use_cuda:
                data, target = data.to(device), target.to(device)
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
        ######################
        # validate the model #
        ######################
        correct = 0.
        correct2 = 0
        correct3 = 0
        correct4 = 0
        total = 0.
        model.eval()
        for batch_idx, (data, target) in enumerate(tqdm(loaders['valid'])):
            # move to GPU
            if use_cuda:
                data, target = data.to(device), target.to(device)
            ## update the average validation loss
            output = model(data)
            loss = criterion(output, target)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

            pred = output.data.max(1, keepdim=True)[1]
            # compare predictions to true label
            correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            correct2 += np.sum(np.squeeze(np.abs(pred.cpu().numpy() - (target.data.view_as(pred).cpu().numpy())) < 5))
            correct3 += np.sum(np.squeeze(np.abs(pred.cpu().numpy() - (target.data.view_as(pred).cpu().numpy())) < 10))
            correct4 += np.sum(np.squeeze(np.abs(pred.cpu().numpy() - (target.data.view_as(pred).cpu().numpy()))))
            total += data.size(0)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
        ))
        print('Test Accuracy: %2d%% (%2d/%2d)' % (
            100. * correct / total, correct, total))
        print('Test Accuracy: %2d%% (%2d/%2d)' % (
            100. * correct2 / total, correct2, total))
        print('Test Accuracy: %2d%% (%2d/%2d)' % (
            100. * correct3 / total, correct3, total))
        print('Test Accuracy: %2d' % (
                correct4 / total))

        ## TODO: save the model if validation loss has decreased
        if valid_loss_min > valid_loss:
            print('Saving Model...')
            valid_loss_min = valid_loss
            torch.save(model.state_dict(), save_path)
    # return trained model
    return model


if __name__ == "__main__":

    # 1. vgg16 和 resnet50 的识别能力

    dir = "./Dataset"

    # 3. 训练
    mean_train_set = [0.595504,  0.54956806, 0.51172713]
    std_train_set = [0.2101685, 0.21753638, 0.22078435]
    train_dir = './DataSet/classify_trainset'
    valid_dir = './DataSet/classify_valset'
    #test_dir = r'D:\commit\valset\valset2'
    train_transforms = transforms.Compose([transforms.Resize([256, 256]),
                                           transforms.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0.1),
                                           transforms.RandomHorizontalFlip(p=0.5),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean_train_set, std_train_set)])
    valid_test_transforms = transforms.Compose([transforms.Resize([256, 256]),
                                                #transforms.CenterCrop(256),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean_train_set, std_train_set)])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_test_transforms)
    #test_dataset = datasets.ImageFolder(test_dir, transform=valid_test_transforms)

    # num_workers=8, pin_memory=True 很重要，训练速度明显
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=16, pin_memory=True)
    validloader = DataLoader(valid_dataset, batch_size=32, shuffle=False,num_workers=8, pin_memory=True) 
    #testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    loaders_scratch = {}
    loaders_scratch['train'] = trainloader
    loaders_scratch['valid'] = validloader
    #loaders_scratch['test'] = testloader
    use_cuda = torch.cuda.is_available()

    # instantiate the CNN
    num_class = 191
    # model_scratch = Net(num_class)
    model_scratch = models.resnet50(pretrained=True)
    for param in model_scratch.parameters():
        param.requires_grad = True
    # model_scratch.classifier = nn.Sequential(nn.Linear(1024, 512),
    #                                           nn.ReLU(),
    #                                           nn.Dropout(0.2),
    #                                           nn.Linear(512, 133))
    #
    # model_scratch.load_state_dict(torch.load('model_transfer.pt', map_location='cuda:0'))
    model_scratch.classifier = nn.Sequential(nn.Linear(1024, 512),
                                             nn.ReLU(),
                                             nn.Dropout(0.2),
                                             nn.Linear(512, num_class))
    # move tensors to GPU if CUDA is available
    if use_cuda:
        model_scratch.to(device)
    criterion_scratch = nn.CrossEntropyLoss()
    optimizer_scratch = optim.Adam(model_scratch.parameters(), lr=0.0005)
    print('training !')
    # epoch
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    model_scratch = train(100, loaders_scratch, model_scratch, optimizer_scratch,
                          criterion_scratch, use_cuda, ' classification.pt')

    # load the model that got the best validation accuracy
    # model_scratch.load_state_dict(torch.load('model_scratch.pt'))
