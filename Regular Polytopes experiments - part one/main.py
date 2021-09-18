
"""
##############################################################################
##############################################################################
########               Artificial Intelligence                ###############
########                      Thesy                            ###############
########         Prof. Capobianco - Prof. Lo Monaco            ###############
########                                                       ###############
########                                                       ###############
########             Students: FRANCESCO CASSINI               ###############
########             Sapienza IDs:       785771                ###############
########     Master in Roboics and Artificial Intelligence     ###############
##############################################################################
##############################################################################




conda deactivate
source /home/francesco/Desktop/PYTORCH/bin/activate
cd /home/francesco/Desktop/TESI
python3 main.py 




##############################################################################
##############                                                ################
##############              BASE CODE TAKEN FROM              ################
##############                                                ################
#  https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html    #
##############                                                ################
##############                                                ################
##############################################################################
##############################################################################
"""

# Standard library for python
from __future__ import print_function, division
import time
import os
import copy
import random
from collections import OrderedDict
import inspect


# Torch library with import of Resnet18 and Resnet for build my model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torchvision.models import resnet18, resnet50


# Math and visualization library
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import itertools
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


# My utils for create model
import models_utils


##############################################################################
##############                                                ################
##############              MAKE TEST REPRODUCIBLE            ################
##############                                                ################
##############################################################################
##############################################################################
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
##############################################################################



##############################################################################
##############                                                ################
##############                    PART 1                      ################
##############                                                ################
##############           CONFIGURATION PARAMETERS             ################
##############                                                ################
##############################################################################
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

batch_size = 256  # Batch size for training (change depending on how much memory you have)
num_epochs = 250  # Number of epochs to train for
feature_extract = True   # Flag for feature extracting. When False, we finetune the whole model, when True we only update the reshaped layer params
data_dir = 'data'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
phase = 'TRAIN'  # 'TRAIN'   'PREDICT'
checkpoint_path = data_dir + '/checkpoint.pt'
dataset = 'cifar10'  # 'mnist, 'cifar10', 'cifar100'
filename = 'log/'+dataset+'_'


##############################################################################
##############                                                ################
##############                    PART 2                      ################
##############                                                ################
##############               DATASET LOADING                  ################
##############                                                ################
##############################################################################

# Data augmentation and normalization for training
# Just normalization for validation

def train_transform(stats):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32,padding=4,padding_mode="reflect"),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

def test_transform(stats):
    return transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

def test_transform_mnist(stats):
    return transforms.Compose([
    transforms.RandomCrop(32,padding=4,padding_mode="reflect"),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

# Mean and STD data taken from : https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
mean = {
'mnist': (0.1307),
'cifar10': (0.4914, 0.4822, 0.4465),
'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
'mnist': (0.3081),
'cifar10': (0.2023, 0.1994, 0.2010),
'cifar100': (0.2675, 0.2565, 0.2761),
}




#stats = (mean[dataset], std[dataset])
if dataset == 'mnist':
    stats = ((0.5), (0.5))
    train_transform = train_transform(stats)
    test_transform = test_transform_mnist(stats)
else:
    stats = ((0.5,0.5,0.5), (0.5,0.5,0.5))
    train_transform = train_transform(stats)
    test_transform = test_transform(stats)

# Data loading code
if  dataset == 'mnist':
    train_dataset = torchvision.datasets.MNIST('/home/francesco/Datasets/mnist', train = True, transform = train_transform, download = True) 
    val_dataset = torchvision.datasets.MNIST('/home/francesco/Datasets/mnist', train = False, transform = test_transform, download = True) 

elif dataset == 'cifar10':
    train_dataset = torchvision.datasets.CIFAR10('/home/francesco/Datasets/cifar10', train = True, transform = train_transform, download = True) 
    val_dataset = torchvision.datasets.CIFAR10('/home/francesco/Datasets/cifar10', train = False, transform = test_transform, download = True) 

elif dataset == 'cifar100':
    train_dataset = torchvision.datasets.CIFAR100('/home/francesco/Datasets/cifar100', train = True, transform = train_transform, download = True) 
    val_dataset = torchvision.datasets.CIFAR100('/home/francesco/Datasets/cifar100', train = False, transform = test_transform, download = True) 



train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


dataloaders = {'train': train_loader, 'val':val_loader}
dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}
class_names = dataloaders['train'].dataset.classes
num_classes = len(class_names)



##############################################################################
##############                                                ################
##############                    PART 3                      ################
##############                                                ################
##############              CNN CUSTOM MODEL                  ################
##############                                                ################
##############################################################################

class LeNet5(nn.Module):

    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        
        self.feature_extractor = nn.Sequential(
        OrderedDict([
        ('conv1', nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)),
        ('Tanh1', nn.Tanh()),
        ('AvgPool2d1', nn.AvgPool2d(kernel_size=2)),
        ('layer1', nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)),
        ('Tanh2', nn.Tanh()),
        ('AvgPool2d2', nn.AvgPool2d(kernel_size=2)),
        ('layer3', nn.Conv2d(16, 120, kernel_size=(5, 5), stride=(1, 1))),
        ('layer4', nn.Tanh())
        ]))

        # for:
        # dsimplex = 5
        # hadamard, dorthoplex, simple, identity = 10
        # dcube = 4
        self.classifier = nn.Sequential(
        OrderedDict([
        ('flatten', nn.Flatten()),
        ('linear1', nn.Linear(120,  84, bias=False)),
        ('Tanh', nn.Tanh()),
        ('linear', nn.Linear(in_features=84, out_features=num_classes)),
        ]))


    def forward(self, x):
        x = self.feature_extractor(x)
        y = self.classifier(x)
        #y = F.softmax(y, dim=1)
        return x, y





class MLP(nn.Module):
    """Multi-Layer Perceptron
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, output_size)
    def forward(self, x):
        #print(x.shape)
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        return output



class VGG16(nn.Module):
    def __init__(self, pretrained: bool, layer_reduction_for_polytype : bool, last_size:int):
        super().__init__()
        orig_vgg: VGG = vgg16(pretrained=pretrained)

        self.feature_extractor = nn.Sequential(
        OrderedDict([
            ('0', nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            ('1', nn.ReLU(inplace=True)),
            ('2', nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            ('3', nn.ReLU(inplace=True)),
            ('4', nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),
            ('5', nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            ('6', nn.ReLU(inplace=True)),
            ('7', nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            ('8', nn.ReLU(inplace=True)),
            ('9', nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),
            ('10', nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            ('11', nn.ReLU(inplace=True)),
            ('12', nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            ('13', nn.ReLU(inplace=True)),
            ('14', nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            ('15', nn.ReLU(inplace=True)),
            ('16', nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),
            ('17', nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            ('18', nn.ReLU(inplace=True)),
            ('19', nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            ('20', nn.ReLU(inplace=True)),
            ('21', nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            ('22', nn.ReLU(inplace=True)),
            ('23', nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),
            ('24', nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            ('25', nn.ReLU(inplace=True)),
            ('26', nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            ('27', nn.ReLU(inplace=True)),
            ('28', nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            ('29', nn.ReLU(inplace=True)),
            ('30', nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),
        ]))


        # for:
        # dsimplex = 5
        # hadamard, dorthoplex, simple, identity = 10
        # dcube = 4
        self.classifier = nn.Sequential(
        OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(output_size=(7, 7))),
            ('flatten', nn.Flatten()),
            ('linear', nn.Linear(in_features=25088, out_features=last_size, bias=False)),
        ]))


    def forward(self, x):
        x = self.feature_extractor(x)
        y = self.classifier(x) 
        #y = F.softmax(y, dim=1)
        return x, y




class ResNet18(nn.Module):
    def __init__(self, pretrained: bool, layer_reduction_for_polytype : bool, last_size:int):
        super().__init__()
        orig_resnet: ResNet = resnet18(pretrained=pretrained)

        downsample = nn.Sequential(
        OrderedDict([
          ('0', nn.Conv2d(64, 32, kernel_size=(1, 1), stride=(2, 2), bias=False)),
          ('1', nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
        ]))

        layer4_reduct = nn.Sequential(
        OrderedDict([
        ('conv1', nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)),
        ('bn1', nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
        ('relu', nn.ReLU(inplace=True)),
        ('conv2', nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
        ('bn2', nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
        ('downsample', downsample),
        ('bconv1', nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
        ('bbn1', nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
        ('brelu', nn.ReLU(inplace=True)),
        ('bconv2', nn.Conv2d(16, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
        ('bbn2', nn.BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
        ]))
        
        if layer_reduction_for_polytype:
            last_layer = layer4_reduct
        else:
            last_layer = orig_resnet.layer4

        self.feature_extractor = nn.Sequential(
        OrderedDict([
        ('conv1', orig_resnet.conv1),
        ('bn1', orig_resnet.bn1),
        ('relu1', orig_resnet.relu),
        ('maxpool1', orig_resnet.maxpool),
        ('layer1', orig_resnet.layer1),
        ('layer2', orig_resnet.layer2),
        ('layer3', orig_resnet.layer3),
        ('layer4', last_layer)
        ]))

        # for:
        # dsimplex = 5
        # hadamard, dorthoplex, simple, identity = 10
        # dcube = 4
        self.classifier = nn.Sequential(
        OrderedDict([
        ('avgpool', nn.AdaptiveAvgPool2d((1, 1))),
        ('flatten', nn.Flatten()),
        ('linear', nn.Linear(512,  last_size, bias=False)),
        ]))


    def forward(self, x):
        x = self.feature_extractor(x)
        y = self.classifier(x) 
        #y = F.softmax(y, dim=1)
        return x, y



class ResNet50(nn.Module):
    def __init__(self, pretrained: bool, layer_reduction_for_polytype : bool, last_size:int):
        super().__init__()
        orig_resnet: ResNet = resnet50(pretrained=pretrained)

        downsample = nn.Sequential(
        OrderedDict([
          ('0', nn.Conv2d(64, 32, kernel_size=(1, 1), stride=(2, 2), bias=False)),
          ('1', nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
        ]))


        self.feature_extractor = nn.Sequential(
        OrderedDict([
        ('conv1', orig_resnet.conv1),
        ('bn1', orig_resnet.bn1),
        ('relu1', orig_resnet.relu),
        ('maxpool1', orig_resnet.maxpool),
        ('layer1', orig_resnet.layer1),
        ('layer2', orig_resnet.layer2),
        ('layer3', orig_resnet.layer3),
        ('layer4', orig_resnet.layer4),
        ]))


        self.classifier = nn.Sequential(
        OrderedDict([
        ('avgpool', nn.AdaptiveAvgPool2d((1, 1))),
        ('flatten', nn.Flatten()),
        #('linear', nn.Linear(2048, last_size, bias=False)),
        ]))


    def forward(self, x):
        x = self.feature_extractor(x)
        y = self.classifier(x)
        #y = F.softmax(y, dim=1)
        return x, y



class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out



class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


##############################################################################
##############                                                ################
##############                    PART 4                      ################
##############                                                ################
##############               TRAIN FUNCTION                   ################
##############                                                ################
##############################################################################

def train(model, num_classes, criterion, optimizer, scheduler, num_epochs, filename):
    from matplotlib import cm
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    perplexity = 80

    eval_loss_history = []
    train_loss_history = []       
    train_acc_history = []
    eval_acc_history = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                preds_history = torch.tensor([], dtype=torch.long).to(device)
                labels_history = torch.tensor([], dtype=torch.long).to(device)
                # For TSNE visualizations we have to zero the tensor
                try:
                    test_embeddings = torch.zeros((0, model.classifier[2].in_features), dtype=torch.float32)
                except:
                    try:
                        test_embeddings = torch.zeros((0, model.feature_extractor.layer4[1].conv2.out_channels), dtype=torch.float32)
                    except:
                        test_embeddings = torch.zeros((0, model.feature_extractor.layer3.out_channels), dtype=torch.float32)
                test_predictions = []

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = Variable(inputs.to(device))
                labels = Variable(labels.to(device))
                # print('input shape: ', inputs.shape)
                # print('label shape: ', labels.shape)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    embeddings, outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    else:
                        preds_history = torch.cat((preds_history, preds), 0) 
                        labels_history = torch.cat((labels_history, labels.data), 0)
                        test_embeddings = torch.cat((test_embeddings, embeddings.squeeze().detach().cpu()), 0)
                        test_predictions.extend(preds.detach().cpu().tolist())    # reduce dimensionality with t-sne



                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.cpu().numpy())
                    
            else:
                eval_loss_history.append(epoch_loss)
                eval_acc_history.append(epoch_acc.cpu().numpy())

                #Feature visualizations by TSNE
                # pca = PCA(n_components=50)
                # X_pca_results_2d = pca.fit_transform(test_embeddings)
                if (epoch > 10 and epoch <30) or (epoch > 150 and epoch <160) or (epoch > 200 and epoch <210) or epoch > 290:
                #if epoch > 298:
                    classification = metrics.classification_report(labels_history.to('cpu'),torch.clip(preds_history, min=0, max=99).to('cpu'),  target_names=class_names)
                    print(classification)
                    tsne_2d = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=500, learning_rate=1000)
                    tsne_results_2d = tsne_2d.fit_transform(test_embeddings)

                    # pca = PCA(n_components=2, whiten=True, tol = 10)
                    # tsne_results_2d = pca.fit_transform(test_embeddings)
                    df_tsne_2d = pd.DataFrame(tsne_results_2d, columns=['comp1', 'comp2'])
                    df_tsne_2d['label'] = test_predictions
                    sns.lmplot(x='comp1', y='comp2', data=df_tsne_2d, hue='label', fit_reg=False)
                    plt.savefig(filename+'_epoch'+str(epoch)+'_per'+str(perplexity)+'.png', bbox_inches='tight')
                    if epoch > 300:
                        tsne_3d = TSNE(n_components=3, verbose=1, perplexity=perplexity, n_iter=500, learning_rate=1000)
                        tsne_results_3d = tsne_3d.fit_transform(test_embeddings)

                        # visualize
                        df_tsne_3d = pd.DataFrame(tsne_results_3d, columns=['comp1', 'comp2', 'comp3'])
                        sns.set_style("whitegrid", {'axes.grid' : False})

                        fig = plt.figure(figsize=(8,8))
                        ax = Axes3D(fig) # Method 1

                        x = df_tsne_3d['comp1']
                        y = df_tsne_3d['comp2']
                        z = df_tsne_3d['comp3']
                        targets = test_predictions


                        ax.scatter(x, y, z, c=targets, marker='o')
                        ax.set_xlabel('X Label')
                        ax.set_ylabel('Y Label')
                        ax.set_zlabel('Z Label')

                        plt.savefig(filename+'_epoch'+str(epoch)+'_per'+str(perplexity)+'_3D.png', bbox_inches='tight')




                #Feature visualizations by PCA
                # pca = PCA(n_components=2)
                # X_pca_results_2d = pca.fit_transform(test_embeddings)
                # df_pca_2d = pd.DataFrame(X_pca_results_2d, columns=['comp1', 'comp2'])
                # df_pca_2d['label'] = test_predictions
                # sns.lmplot(x='comp1', y='comp2', data=df_pca_2d, hue='label', fit_reg=False)
                # plt.savefig('log/dorthoplex-n_pca'+str(epoch)+'.png', bbox_inches='tight')

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        print()



    # pca = PCA(n_components=50)
    # X_pca = pca.fit_transform(test_embeddings)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print()

    acc = train_acc_history
    val_acc = eval_acc_history
    loss = train_loss_history
    val_loss = eval_loss_history
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.ylabel('accuracy') 
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(filename+'_epoch'+str(epoch)+'_per'+str(perplexity)+'_ACCURACY.png', bbox_inches='tight')
    plt.clf() #plot clear


    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.ylabel('loss') 
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(filename+'_epoch'+str(epoch)+'_per'+str(perplexity)+'_LOSS.png', bbox_inches='tight')
    plt.clf() #plot clear

    cm = metrics.confusion_matrix(labels_history.short().to('cpu'), preds_history.short().to('cpu'))
    # Absolute
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt='.0f', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(filename+'_epoch'+str(epoch)+'_per'+str(perplexity)+'_CONFUSION_ABSOLUTE.png', bbox_inches='tight')
    # Normalise
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(filename+'_epoch'+str(epoch)+'_per'+str(perplexity)+'_CONFUSION_PERCENTAGE.png', bbox_inches='tight')

    print()
    classification = metrics.classification_report(labels_history.to('cpu'),torch.clip(preds_history, min=0, max=99).to('cpu'),  target_names=class_names)
    print(classification)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model






########## PREDICT FUNCTION works without graphs (for Jetson Nano execution)!! 
##########  Strange... but It's correct!
def predict(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            print('image: ', labels, '  -  ', preds)

        #     for j in range(inputs.size()[0]):
        #         images_so_far += 1
        #         ax = plt.subplot(num_images//2, 2, images_so_far)
        #         ax.axis('off')
        #         ax.set_title('predicted: {}'.format(class_names[preds[j]]))
        #         plt.show(inputs.cpu().data[j])

        #         if images_so_far == num_images:
        #             model.train(mode=was_training)
        #             return
        # model.train(mode=was_training)




##############################################################################
##############                                                ################
##############                    PART 5                      ################
##############                                                ################
##############                MODEL LOADING                   ################
##############                                                ################
##############################################################################


############################################################################
###  MODEL CHOICES OPTIONS
############################################################################

model_name =  "dorthoplex"
layer_reduction_for_polytype = False
model_type = 'RESNET18'
out_dim = 10

if phase == 'TRAIN':
    if model_name == "simple": 
        filename += model_name +'_' + model_type
        fc = None
        last_embedding_layer = out_dim

    elif model_name == "fixed_identity":  
        filename += model_name +'_' + model_type        
        fc = nn.Identity()
        last_embedding_layer = out_dim

    elif model_name == "dorthoplex": 
        filename += model_name + str(out_dim)  + '_' + model_type
        fixed_classifier_feat_dim = int(np.ceil(out_dim / 2).astype(int))
        fixed_weights = models_utils.dorthoplex_matrix(num_classes=out_dim)
        fc = nn.Linear(fixed_classifier_feat_dim, out_dim, bias=False)
        last_embedding_layer = out_dim // 2

    elif model_name == "dsimplex":  
        filename += model_name + str(out_dim) +'_' + model_type
        fixed_classifier_feat_dim = out_dim - 1
        fixed_weights = torch.transpose(models_utils.dsimplex_matrix(num_classes=out_dim),0,1)
        fc = nn.Linear(fixed_classifier_feat_dim, out_dim, bias=False)
        last_embedding_layer = out_dim - 1

    elif model_name == "dcube": 
        filename += model_name + str(out_dim) +'_' + model_type
        fixed_classifier_feat_dim = 4
        out_dim = fixed_classifier_feat_dim ** 2
        fixed_weights = models_utils.dcube_matrix(num_classes=out_dim)
        fc = nn.Linear(fixed_classifier_feat_dim, out_dim, bias=False)
        last_embedding_layer = fixed_classifier_feat_dim

    elif model_name == "hadamard": 
        out_dim = 16
        filename += model_name + str(out_dim)  + '_' + model_type
        fixed_weights = models_utils.hadamard_matrix(out_dim)
        print( fixed_weights)
        fc = nn.Linear(out_dim, out_dim, bias=False)
        last_embedding_layer = out_dim



    if model_type == 'RESNET50':
        model = ResNet50(pretrained=True, layer_reduction_for_polytype = layer_reduction_for_polytype, last_size=100)  #50 for orthplex, 100 for other #7 for dcube
    elif model_type == 'RESNET18':
        model = ResNet18(pretrained=True, layer_reduction_for_polytype = layer_reduction_for_polytype, last_size=last_embedding_layer)
    elif model_type == 'LeNet5':
        model = LeNet5(num_classes=out_dim)

    if fc != None:
        if model_name != "fixed_identity":
            fc.weight.requires_grad_(False)
            fc.weight.copy_(fixed_weights)
        model.classifier.add_module("fc", fc)
    if dataset == 'mnist' and model_type != 'LeNet5':
        model.feature_extractor.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)



    print(model)
    print(model_name)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Decay LR by a factor of 0.1 every 150 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.1)

    model = train(model, num_classes, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs, filename = filename)
    torch.save(model, checkpoint_path)


else:
    model = torch.load(checkpoint_path)
    predict(model)
