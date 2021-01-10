import torch                                                        
from torch.utils.data import DataLoader                             
from torchvision import transforms                                  
import torch.optim as optim                                         
import torch.nn as nn                                               
import torch.backends.cudnn as cudnn                                
import torchvision.datasets as datasets  

import os
import argparse
import numpy as np
import random

import matplotlib.pyplot as plt
import cv2


def load_imagenet_class100_small(batch_size, img_s_load=512, img_s_return=448, divs=1,
        server_type='libigpu5', isRandomResize=False):
    if server_type == 'libigpu4':
        path = '/home/choi574/datasets/ImageNet2012_class100_ss/'
    elif server_type == 'libigpu5':
        path = '/home/choi574/datasets/ImageNet2012_class100_ss/'
    elif server_type == 'libigpu6':
        path = '/home/choi574/datasets/ImageNet2012_class100_ss/'
    elif server_type == 'libigpu7':
        path = '/home/choi574/datasets/ImageNet2012_class100_ss/'
    else:
        print("undefined server type")
    path = os.path.join(path, str(divs))
    path_val = '/home/choi574/datasets/ImageNet2012_class100/'

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if isRandomResize:
        print('load imagenet with RandomResize')
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(img_s_return),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ]) 
    else:
        print('load imagenet without RandomResize')
        train_transforms = transforms.Compose([
            transforms.Resize(img_s_load),
            transforms.CenterCrop(img_s_return),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ]) 


    train_data = datasets.ImageFolder(root=os.path.expanduser(path + '/train/'),
                                        transform=train_transforms)
    '''train_data = datasets.ImageFolder(root=os.path.expanduser(path + 'train/'),
    transform=transforms.Compose([
    transforms.RandomResizedCrop(img_s_return),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
    ]))'''
    test_data =  datasets.ImageFolder(root=os.path.expanduser(path_val + 'val/'),
        transform=transforms.Compose([
            transforms.Resize(img_s_load),
            transforms.CenterCrop(img_s_return),
            transforms.ToTensor(),
            normalize
        ]))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)
    return train_loader, test_loader, 100

