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
from PIL import Image
import sys

plt.switch_backend('agg') 
#from mpl_toolkits.axes_grid1 import ImageGrid
#from sklearn.utils import linear_assignment_
#from scipy.stats import itemfreq
#from sklearn.cluster import KMeans
#from itertools import chain

# This is required for salicon dataset
import datasetSALICON as ds


import numpy as np
from scipy.stats import pearsonr

def pearson_r(x, y):
    """
    Calculate Pearson's correlation coefficient (r) between two numpy arrays,
    for each feature dimension.
    Written by ChatGPT

    Parameters
    ----------
    x : numpy array, size n_samples by feature_dimensions
        First input array.
    y : numpy array, size n_samples by feature_dimensions
        Second input array.

    Returns
    -------
    r : numpy array, size feature_dimensions
        Pearson's correlation coefficient (r) between x and y, for each feature dimension.
    """
    r = np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        r[i], _ = pearsonr(x[:,i], y[:,i])
        
    return r


def check_nan(np_array):
    '''
    2023.02.07. Minkyu
    Return True if any Nan is included'''
    return np.isnan(np_array).any()

class Conv2d(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, stride=1, 
                 padding=0, padding_mode='zeros',
                 dilation=1, bias=False, activation=True):
        super().__init__()
        self.activation = activation
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, 
                              padding=padding, padding_mode=padding_mode,
                              dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = x
        out = self.relu(self.bn(self.conv(out))) if self.activation else self.conv(out)
        return out


from pathlib import Path
def make_dir(path, parents=False, exist_ok=True):
    Path(os.path.expanduser(path)).mkdir(parents=parents, exist_ok=exist_ok)

def remove_nan(fms_norm):
    '''2023.01.25. 
    Remove nan from given tensor. 
    Repalce nan value to 0.0. 
    '''
    
    fms_norm[fms_norm != fms_norm] = 0
    return fms_norm


class Load_PIL_to_tensor():
    '''
    Modified 2022.11.02, Minkyu
        1) mean and std can be given as inputs. 
    '''
    def __init__(self, img_s_load=227, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], isResize=True):
        if isResize:
            self.transform = transforms.Compose([
                transforms.Resize(img_s_load),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=mean,
                    std=std,
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=mean,
                    std=std,
                ),
            ])

    def pil_loader(self, path: str) -> Image.Image:
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        # https://github.com/pytorch/vision/blob/main/torchvision/datasets/folder.py#L244
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def load_image_to_tensor(self, path: str) -> torch.Tensor:
        image_pil = self.pil_loader(path)
        #image_tensor = transforms.ToTensor()(image_pil).unsqueeze_(0)
        image_tensor = self.transform(image_pil)
        #print(image_tensor.size(), torch.max(image_tensor), torch.min(image_tensor))
        
        return image_tensor.unsqueeze(0)

def isNan(tensor):
    '''get a tensor and return True if it includes Nan'''
    return (tensor!=tensor).any()

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

def initialize_weight(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def load_place2(batch_size, img_s_load=256, img_s_return=224, server_type='libigpu0', num_workers=4):
    if server_type == 'libigpu0':
        path = '~/DATASET/Places2/places365_standard/'
    elif server_type == 'libigpu1':
        path = '~/DATASET/Places2/places365_standard/'
    elif server_type == 'home':
        path = '~/DATASETS/Places2/places365_standard/'
    elif server_type == 'libigpu3':
        path = '~/DATASET/Places2/places365_standard/'
    elif server_type == 'libigpu4' or server_type == 'libigpu5':
        path = '~/datasets/places365_standard/'
    else:
        print("undefined server type")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    train_data = datasets.ImageFolder(root=os.path.expanduser(path + 'train'),
            transform=transforms.Compose([
                transforms.RandomResizedCrop(img_s_return),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
                ]))
    test_data =  datasets.ImageFolder(root=os.path.expanduser(path + 'val'),
            transform=transforms.Compose([
                transforms.Resize(img_s_load),
                transforms.CenterCrop(img_s_return),
                transforms.ToTensor(),
                normalize
                ]))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True,
            num_workers=int(num_workers/2), pin_memory=True, drop_last=True)
    return train_loader, test_loader, 365




def load_places_class100_large(batch_size, img_s_load=512, img_s_return=448, server_type='libigpu5', num_workers=8, isRandomResize=True):
    print(server_type)
    if server_type == 'libigpu2':
        path = '/home/choi574/datasets/places100_sampled_large/'
    elif server_type=='libigpu2' or server_type=='libigpu5' or server_type=='libigpu6':
        path = '/home/choi574/datasets/places100_sampled_large/'
    elif server_type=='libigpu0' or server_type=='libigpu1' or server_type=='libigpu3':
        path = '/home/min/datasets/places100_sampled_large/'
    else:
        print("undefined server type")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if isRandomResize:
        print('load places100 with RandomResize')
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(img_s_return),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ]) 
    else:
        print('load places100 without RandomResize')
        train_transforms = transforms.Compose([
            transforms.Resize(img_s_load),
            transforms.CenterCrop(img_s_return),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ]) 

    train_data = datasets.ImageFolder(root=os.path.expanduser(path + 'train/'),
                                        transform=train_transforms)
    test_data =  datasets.ImageFolder(root=os.path.expanduser(path + 'val/'),
        transform=transforms.Compose([
            transforms.Resize(img_s_load),
            transforms.CenterCrop(img_s_return),
            transforms.ToTensor(),
            normalize
        ]))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True,
        num_workers=int(num_workers/2), pin_memory=True, drop_last=True)
    return train_loader, test_loader, 100


def load_places_class100(batch_size, img_s_load=256, img_s_return=224, server_type='libigpu5', num_workers=4, isRandomResize=True):
    print(server_type)
    if server_type == 'libigpu2':
        path = '/home/choi574/datasets/places100_sampled/'
    elif server_type=='libigpu2' or server_type=='libigpu5' or server_type=='libigpu6':
        path = '/home/choi574/datasets/places100_sampled/'
    elif server_type=='libigpu0' or server_type=='libigpu1' or server_type=='libigpu3':
        path = '/home/min/datasets/places100_sampled/'
    else:
        print("undefined server type")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if isRandomResize:
        print('load places100 with RandomResize')
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(img_s_return),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ]) 
    else:
        print('load places100 without RandomResize')
        train_transforms = transforms.Compose([
            transforms.Resize(img_s_load),
            transforms.CenterCrop(img_s_return),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ]) 

    train_data = datasets.ImageFolder(root=os.path.expanduser(path + 'train/'),
                                        transform=train_transforms)
    test_data =  datasets.ImageFolder(root=os.path.expanduser(path + 'val/'),
        transform=transforms.Compose([
            transforms.Resize(img_s_load),
            transforms.CenterCrop(img_s_return),
            transforms.ToTensor(),
            normalize
        ]))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True,
        num_workers=int(num_workers/2), pin_memory=True, drop_last=True)
    return train_loader, test_loader, 365


def load_places_class365(batch_size, img_s_load=512, img_s_return=448, server_type='libigpu5', isRandomResize=True):
    if server_type == 'libigpu4':
        path = '/home/choi574/datasets/places20/'
    elif server_type == 'libigpu5':
        path = '/home/choi574/datasets/places20/'
    elif server_type == 'libigpu6':
        path = '/home/choi574/datasets/places20/'
    elif server_type == 'libigpu7':
        path = '/home/choi574/datasets/places20/'
    elif server_type == 'libigpu0':
        path = '/home/min/DATASET/Places2/places365_standard/'
    else:
        print("undefined server type")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if isRandomResize:
        print('load places365 with RandomResize')
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(img_s_return),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ]) 
    else:
        print('load places365 without RandomResize')
        train_transforms = transforms.Compose([
            transforms.Resize(img_s_load),
            transforms.CenterCrop(img_s_return),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ]) 


    train_data = datasets.ImageFolder(root=os.path.expanduser(path + 'train/'),
                                        transform=train_transforms)
    test_data =  datasets.ImageFolder(root=os.path.expanduser(path + 'val/'),
        transform=transforms.Compose([
            transforms.Resize(img_s_load),
            transforms.CenterCrop(img_s_return),
            transforms.ToTensor(),
            normalize
        ]))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True)
    return train_loader, test_loader, 365

def load_places_class20(batch_size, img_s_load=512, img_s_return=448, server_type='libigpu5', isRandomResize=True):
    if server_type == 'libigpu4':
        path = '/home/choi574/datasets/places20/'
    elif server_type == 'libigpu5':
        path = '/home/choi574/datasets/places20/'
    elif server_type == 'libigpu6':
        path = '/home/choi574/datasets/places20/'
    elif server_type == 'libigpu7':
        path = '/home/choi574/datasets/places20/'
    else:
        print("undefined server type")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if isRandomResize:
        print('load places20 with RandomResize')
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(img_s_return),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ]) 
    else:
        print('load places20 without RandomResize')
        train_transforms = transforms.Compose([
            transforms.Resize(img_s_load),
            transforms.CenterCrop(img_s_return),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ]) 


    train_data = datasets.ImageFolder(root=os.path.expanduser(path + 'train/'),
                                        transform=train_transforms)
    test_data =  datasets.ImageFolder(root=os.path.expanduser(path + 'val/'),
        transform=transforms.Compose([
            transforms.Resize(img_s_load),
            transforms.CenterCrop(img_s_return),
            transforms.ToTensor(),
            normalize
        ]))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True)
    return train_loader, test_loader, 20

def notused_load_imagenet_class100(batch_size, img_s_load=512, img_s_return=448, server_type='libigpu5', isRandomResize=True, num_workers=4, num_workers_t=None):
    if server_type == 'libigpu4':
        path = '/home/choi574/datasets/ImageNet2012_class100/'
    elif server_type == 'libigpu5':
        path = '/home/choi574/datasets/ImageNet2012_class100/'
    elif server_type == 'libigpu6':
        path = '/home/choi574/datasets/ImageNet2012_class100/'
    elif server_type == 'libigpu7':
        path = '/home/choi574/datasets/ImageNet2012_class100/'
    elif server_type == 'libigpu0':
        path = '/home/min/datasets/ImageNet2012_class100/'
    elif server_type == 'libigpu1':
        path = '/home/min/datasets/ImageNet2012_class100/'
    elif server_type == 'libigpu3':
        path = '/home/min/datasets/ImageNet2012_class100/'
    elif server_type == 'libigpu2':
        path = '/home/choi574/datasets/ImageNet2012_class100/'
    else:
        print("undefined server type")

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


    train_data = datasets.ImageFolder(root=os.path.expanduser(path + 'train/'),
                                        transform=train_transforms)
    '''train_data = datasets.ImageFolder(root=os.path.expanduser(path + 'train/'),
    transform=transforms.Compose([
    transforms.RandomResizedCrop(img_s_return),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
    ]))'''
    test_data =  datasets.ImageFolder(root=os.path.expanduser(path + 'val/'),
        transform=transforms.Compose([
            transforms.Resize(img_s_load),
            transforms.CenterCrop(img_s_return),
            transforms.ToTensor(),
            normalize
        ]))

    if num_workers_t == None:
        num_workers_t = num_workers
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers_t, pin_memory=True, drop_last=True)
    return train_loader, test_loader, 100

def notused_load_imagenet_class100_DDP(batch_size, img_s_load=512, img_s_return=448, server_type='libigpu5', isRandomResize=True, num_workers=4):
    if server_type == 'libigpu4':
        path = '/home/choi574/datasets/ImageNet2012_class100/'
    elif server_type == 'libigpu2':
        path = '/home/choi574/datasets/ImageNet2012_class100/'
    elif server_type == 'libigpu5':
        path = '/home/choi574/datasets/ImageNet2012_class100/'
    elif server_type == 'libigpu6':
        path = '/home/choi574/datasets/ImageNet2012_class100/'
    elif server_type == 'libigpu7':
        path = '/home/choi574/datasets/ImageNet2012_class100/'
    elif server_type == 'libigpu0':
        path = '/home/min/datasets/ImageNet2012_class100/'
    elif server_type == 'libigpu1':
        path = '/home/min/datasets/ImageNet2012_class100/'
    elif server_type == 'libigpu3':
        path = '/home/min/datasets/ImageNet2012_class100/'
    else:
        print("undefined server type")

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


    train_data = datasets.ImageFolder(root=os.path.expanduser(path + 'train/'),
                                        transform=train_transforms)
    test_data =  datasets.ImageFolder(root=os.path.expanduser(path + 'val/'),
        transform=transforms.Compose([
            transforms.Resize(img_s_load),
            transforms.CenterCrop(img_s_return),
            transforms.ToTensor(),
            normalize
        ]))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    return train_loader, test_loader, 100, train_sampler


def load_imagenet(batch_size, img_s_load=512, img_s_return=448, server_type='libigpu5', isRandomResize=True, num_workers=4, num_workers_t=None):
    if server_type == 'libigpu0':
        path = '/home/libi/datasets/ImageNet2012/'
    elif server_type == 'libigpu1':
        path = '/home/libiadm/data/ImageNet2012/'
    elif server_type == 'home':
        path = '~/datasets/ImageNet2012/'
    elif server_type == 'libigpu2':
        path = '/home/libiadm/datasets/ImageNet2012/'
    elif server_type == 'libigpu3':
        path = '/home/libiadm/data/ImageNet2012/'
    elif server_type == 'libigpu4':
        path = '/home/libiadm/datasets/ImageNet2012/'
    elif server_type == 'libigpu5':
        path = '/home/libiadm/datasets/ImageNet2012/'
    elif server_type == 'libigpu6':
        path = '/home/libiadm/datasets/ImageNet2012/'
    elif server_type == 'libigpu7':
        path = '/home/libiadm/datasets/ImageNet2012/'
    elif server_typt == 'arc':
        path = '/tmp/minkyu/ImageNet2012/'
    else:
        print("undefined server type")

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


    train_data = datasets.ImageFolder(root=os.path.expanduser(path + 'train/'),
                                        transform=train_transforms)
    '''train_data = datasets.ImageFolder(root=os.path.expanduser(path + 'train/'),
    transform=transforms.Compose([
    transforms.RandomResizedCrop(img_s_return),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
    ]))'''
    test_data =  datasets.ImageFolder(root=os.path.expanduser(path + 'val/'),
        transform=transforms.Compose([
            transforms.Resize(img_s_load),
            transforms.CenterCrop(img_s_return),
            transforms.ToTensor(),
            normalize
        ]))

    if num_workers_t == None:
        num_workers_t = num_workers

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    return train_loader, test_loader, 1000

def load_imagenet_DDP(batch_size, img_s_load=512, img_s_return=448, server_type='libigpu5', isRandomResize=True, num_workers=4):
    if server_type == 'libigpu0':
        path = '/home/libi/datasets/ImageNet2012/'
    elif server_type == 'libigpu1':
        path = '/home/libiadm/data/ImageNet2012/'
    elif server_type == 'home':
        path = '~/datasets/ImageNet2012/'
    elif server_type == 'libigpu2':
        path = '/home/libiadm/datasets/ImageNet2012/'
    elif server_type == 'libigpu3':
        path = '/home/libiadm/data/ImageNet2012/'
    elif server_type == 'libigpu4':
        path = '/home/libiadm/datasets/ImageNet2012/'
    elif server_type == 'libigpu5':
        #path = '/home/libiadm/datasets/ImageNet2012/'
        path = '/home/choi574/datasets/ImageNet2012/'
    elif server_type == 'libigpu6':
        path = '/home/libiadm/datasets/ImageNet2012/'
    elif server_type == 'libigpu7':
        path = '/home/libiadm/datasets/ImageNet2012/'
    elif server_typt == 'arc':
        path = '/tmp/minkyu/ImageNet2012/'
    else:
        print("undefined server type")

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


    train_data = datasets.ImageFolder(root=os.path.expanduser(path + 'train/'),
                                        transform=train_transforms)
    '''train_data = datasets.ImageFolder(root=os.path.expanduser(path + 'train/'),
    transform=transforms.Compose([
    transforms.RandomResizedCrop(img_s_return),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
    ]))'''
    test_data =  datasets.ImageFolder(root=os.path.expanduser(path + 'val/'),
        transform=transforms.Compose([
            transforms.Resize(img_s_load),
            transforms.CenterCrop(img_s_return),
            transforms.ToTensor(),
            normalize
        ]))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True,
        num_workers=int(num_workers/2), pin_memory=True, drop_last=True)
    return train_loader, test_loader, 1000, train_sampler


def load_salicon(batch_size, server_type, img_size=(480, 640)):
    '''In order to use this function, you need to move all the images in the ./test/ into ./test/1/. 
    This is because the pytorch's imageFolder and Dataloader works in this way. 
    '''
    if server_type == 'libigpu1':
        path_dataset = os.path.expanduser('~/datasets/salicon_original')
    elif server_type == 'libigpu5' or server_type == 'libigpu7' or server_type == 'libigpu6' or server_type == 'libigpu4':
        path_dataset = os.path.expanduser('~/datasets/salicon_original')
    else:
        print('[ERROR]: Server type not implemented: ', server_type)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    data_train = ds.SALICON(path_dataset, mode='train', size_img=img_size, size_sal=img_size)
    data_val = ds.SALICON(path_dataset, mode='val', size_img=img_size, size_sal=img_size)
    data_test = datasets.ImageFolder(root=os.path.expanduser(os.path.join(path_dataset,'image', 'images', 'test')), 
            transform=transforms.Compose([transforms.ToTensor(), normalize]))
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, 
            num_workers=4, pin_memory=False, drop_last=True)
    val_loader = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=False, 
            num_workers=1, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=False, 
            num_workers=1, pin_memory=True, drop_last=False)
    return train_loader, val_loader, test_loader

def load_mit300(batch_size, server_type):
    if server_type == 'libigpu1':
        path_dataset = os.path.expanduser('/home/libiadm/HDD1/libigpu1/minkyu/datasets/mit300/BenchmarkIMAGES')
    else:
        print('[ERROR]: Server type not implemented')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    test_data =  datasets.ImageFolder(root=os.path.expanduser(path_dataset),
        transform=transforms.Compose([
            transforms.Resize((360, 480)),
            transforms.ToTensor(),
            normalize
            ]))

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, drop_last=False)
    return test_loader, test_loader, test_loader

def load_MIE(batch_size, server_type, data_type):
    '''
    When I downloaded the corresponding dataset from https://www-percept.irisa.fr/asperger_to_kanner/, 
    I unzip the file and changed file names from one digit to two digits by hand. >> 1.png --> 01.png 
    And I also put a ./0 dir inside the MIE_Fo and MIE_No and moved all images into it. 

    data_type: 'fo' or 'no'
    '''
    if server_type == 'libigpu1':
        if data_type == 'fo':
            path_dataset = os.path.expanduser('/home/libiadm/HDD1/libigpu1/minkyu/datasets/ASD/MIE_Fo/stimuli/')
        elif data_type == 'no':
            path_dataset = os.path.expanduser('/home/libiadm/HDD1/libigpu1/minkyu/datasets/ASD/MIE_No/stimuli/')
        else:
            print('[ERROR]: something is wrong')
    else:
        print('[ERROR]: Server type not implemented')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    test_data =  datasets.ImageFolder(root=os.path.expanduser(path_dataset),
        transform=transforms.Compose([
            transforms.Resize((360, 480)),
            transforms.ToTensor(),
            normalize
            ]))

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, drop_last=False)
    return test_loader, test_loader, test_loader

def load_sequence_mnist100(batch_size):
    data = np.load('/home/libilab/a/users/choi574/DATASETS/IMAGE/mnist/cluttered_sequence/mnist_sequence3_sample_8dsistortions9x9.npz')
    x_train, y_train = data['X_train'], data['y_train']
    x_test, y_test = data['X_test'], data['y_test']
    
    print(np.shape(x_train), np.size(x_train))
    x_train = (np.reshape(x_train, (x_train.shape[0], 1, 100, 100))-0.5)*2.0
    x_test = (np.reshape(x_test, (x_test.shape[0], 1, 100, 100))-0.5)*2.0
    #x_train = (x_test.reshape((x_train.shape[0], 1, 100, 100))-0.5)*2.0
    #x_test = (x_test.reshape((x_test.shape[0], 1, 100, 100))-0.5)*2.0

    train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)), 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)), 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True)
    return train_loader, test_loader, 10



def load_cluttered_mnist60(batch_size):
    data = np.load('/home/libilab/a/users/choi574/DATASETS/IMAGE/mnist/cluttered_60/mnist_cluttered_60x60_6distortions.npz')
    x_train, y_train = data['x_train'], np.argmax(data['y_train'], axis=-1)
    x_valid, y_valid = data['x_valid'], np.argmax(data['y_valid'], axis=-1)
    x_test, y_test = data['x_test'], np.argmax(data['y_test'], axis=-1)
    x_tv = np.concatenate((x_train, x_valid), 0)
    y_tv = np.concatenate((y_train, y_valid), 0)
    x_tv = (x_tv.reshape((x_tv.shape[0], 1, 60, 60))-0.5)*2.0
    x_test = (x_test.reshape((x_test.shape[0], 1, 60, 60))-0.5)*2.0
    train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(x_tv), torch.from_numpy(y_tv)), 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)), 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True)
    return train_loader, test_loader, 10



def load_lsun(batch_size, img_size=256):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    train_loader = torch.utils.data.DataLoader(
            datasets.LSUN(root=os.path.expanduser('/home/libi/HDD1/minkyu/DATASETS/IMAGE/LSUN'), classes='train', transform=transforms.Compose([
                transforms.RandomHorizontalFlip(), 
                transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0), ratio=(1,1.3)),
                transforms.ToTensor(),
                normalize]), target_transform=None), 
            batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(
            datasets.LSUN(root=os.path.expanduser('/home/libi/HDD1/minkyu/DATASETS/IMAGE/LSUN'), classes='val', transform=transforms.Compose([
                #transforms.RandomHorizontalFlip(), 
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(1,1.3)),
                transforms.ToTensor(),
                normalize]), target_transform=None), 
            batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
            datasets.LSUN(root=os.path.expanduser('/home/libi/HDD1/minkyu/DATASETS/IMAGE/LSUN'), classes='test', transform=transforms.Compose([
                #transforms.RandomHorizontalFlip(), 
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(1,1.3)),
                transforms.ToTensor(),
                normalize]), target_transform=None), 
            batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True, drop_last=True)
    return train_loader, valid_loader, 10

def load_mnist_old(batch_size, img_size=32, server_type='libigpu4567'):
    if server_type == 'libigpu4567':
        save_dir = '/home/choi574/datasets/mnist'
    else:
        print('dir not defined for other gpu servers')

    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],
                    std=[0.5, 0.5, 0.5])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=save_dir, 
            train=True, transform=transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=batch_size, shuffle=True,
        num_workers=1, pin_memory=True, drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=save_dir, 
            train=False, transform=transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=batch_size, shuffle=True,
        num_workers=1, pin_memory=True, drop_last=True)
    return train_loader, test_loader, 10

def load_mnist(batch_size, img_size=32, server_type='libigpu4567', isNormalize=True):
    if server_type == 'libigpu4567':
        save_dir = './'
    else:
        print('dir not defined for other gpu servers')

    if isNormalize:
        normalize = transforms.Normalize(mean=[0.5],
                                        std=[0.5])

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root=save_dir,
                train=True, transform=transforms.Compose([
                #transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=batch_size, shuffle=True,
            num_workers=1, pin_memory=True, drop_last=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root=save_dir,
                train=False, transform=transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=batch_size, shuffle=True,
            num_workers=1, pin_memory=True, drop_last=True)

    else:
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root=save_dir,
                train=True, transform=transforms.Compose([
                #transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ]), download=True),
            batch_size=batch_size, shuffle=True,
            num_workers=1, pin_memory=True, drop_last=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root=save_dir,
                train=False, transform=transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ]), download=True),
            batch_size=batch_size, shuffle=True,
            num_workers=1, pin_memory=True, drop_last=True)

    return train_loader, test_loader, 10
   


def load_cifar100(batch_size, img_size=32):
    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],
                                         std=[0.5, 0.5, 0.5])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='../../../../../../../DATASETS/cifar100/', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='../../../../../../../DATASETS/cifar100/', train=False, transform=transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True, drop_last=True)
    return train_loader, test_loader, 100

def load_cifar10(batch_size):
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2029, 0.2024, 0.2025])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='../../../../../../DATASETS/cifar10/', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            #transforms.Resize((64, 64)),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='../../../../../../DATASETS/cifar10/', train=False, transform=transforms.Compose([
            #transforms.Resize((64, 64)),
            transforms.ToTensor(),
            normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True, drop_last=True)
    return train_loader, test_loader, 10


def load_stanford_dogs(batch_size, img_size):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5])
    train_data = datasets.ImageFolder(root='../../../../../../DATASETS/IMAGE/StanfordDogs/train/',
            transform=transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(1,1.3)),
                #transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
                ]))
    test_data =  datasets.ImageFolder(root='../../../../../../DATASETS/IMAGE/StanfordDogs/test/',
            transform=transforms.Compose([
                #transforms.RandomResizedCrop(448, scale=(0.8, 1.0), ratio=(1,1.3)),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                normalize
                ]))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True, drop_last=True)
    return train_loader, test_loader, 120

def load_ucsd_birds(batch_size, img_size):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5])
    train_data = datasets.ImageFolder(root='../../../../../../DATASETS/IMAGE/CUB_200_2011/train/',
            transform=transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(1,1.3)),
                #transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
                ]))
    test_data =  datasets.ImageFolder(root='../../../../../../DATASETS/IMAGE/CUB_200_2011/test/',
            transform=transforms.Compose([
                #transforms.RandomResizedCrop(448, scale=(0.8, 1.0), ratio=(1,1.3)),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                normalize
                ]))


    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True, drop_last=True)
    return train_loader, test_loader, 200

def plot_samples_from_images_fixedRange(images, batch_size, plot_path, filename, isRange01=False):
    ''' Plot images
    Changed 2020.11.23
        isRange01 is added to normalize image in different way. 

    Args: 
        images: (b, c, h, w), tensor in any range. (c=3 or 1)
        batch_size: int
        plot_path: string
        filename: string
        isRange01: True/False, Normalization will be different. 
    '''
    #print(torch.max(images), torch.min(images))
    if isRange01:
        max_pix = torch.max(torch.abs(images))
        images = images/max_pix
        #print(max_pix, torch.min(torch.abs(images)))
        #print(torch.max(images))
    else:
        #print(max_pix)
        images = ((images*0.2) + 0.5)
        
    if(images.size()[1] == 1): # binary image
        images = torch.cat((images, images, images), 1)
    
    images = np.swapaxes(np.swapaxes(images.cpu().numpy(), 1, 2), 2, 3)

    fig = plt.figure(figsize=(batch_size/4+5, batch_size/4+5))
    for idx in np.arange(batch_size):
        ax = fig.add_subplot(batch_size/8, 8, idx+1, xticks=[], yticks=[])
        ax.imshow(images[idx])
    plt.tight_layout(pad=1, w_pad=0, h_pad=0)
    if plot_path:
        plt.savefig(os.path.join(plot_path, filename))
    else:
        plt.show()
    plt.close()
    pass


def plot_samples_from_images(images, batch_size, plot_path, filename, isRange01=False, nCols=8):
    ''' Plot images
    Changed 2020.11.23
        isRange01 is added to normalize image in different way. 

    Args: 
        images: (b, c, h, w), tensor in any range. (c=3 or 1)
        batch_size: int
        plot_path: string
        filename: string
        isRange01: True/False, Normalization will be different. 
    '''
    #print(torch.max(images), torch.min(images))
    if isRange01:
        max_pix = torch.max(torch.abs(images))
        images = images/max_pix
        #print(max_pix, torch.min(torch.abs(images)))
        #print(torch.max(images))
    else:
        max_pix = torch.max(torch.abs(images))
        #print(max_pix)
        if max_pix != 0.0:
            images = ((images/max_pix) + 1.0)/2.0
        else:
            images = (images + 1.0) / 2.0
            #print('inside')
    if(images.size()[1] == 1): # binary image
        images = torch.cat((images, images, images), 1)
    
    images = np.swapaxes(np.swapaxes(images.cpu().numpy(), 1, 2), 2, 3)

    fig = plt.figure(figsize=(int(batch_size/4)+5, int(batch_size/4)+5))
    for idx in np.arange(batch_size):
        ax = fig.add_subplot(int(batch_size/nCols), nCols, idx+1, xticks=[], yticks=[])
        ax.imshow(images[idx])
    plt.tight_layout(pad=1, w_pad=0, h_pad=0)
    if plot_path:
        plt.savefig(os.path.join(plot_path, filename))
    else:
        plt.show()
    plt.close()
    pass

def plot_one_sample_from_images_old(images, plot_path, filename, isRange01=False):
    ''' Plot images
    Changed 2020.11.23
        isRange01 is added to normalize image in different way. 

    Args: 
        images: (b, c, h, w), tensor in any range. (c=3 or 1)
        batch_size: int
        plot_path: string
        filename: string
        isRange01: True/False, Normalization will be different. 
    '''
    #print(torch.max(images), torch.min(images))
    if isRange01:
        max_pix = torch.max(torch.abs(images))
        images = images/max_pix
        #print(max_pix, torch.min(torch.abs(images)))
        #print(torch.max(images))
    else:
        max_pix = torch.max(torch.abs(images))
        #print(max_pix)
        if max_pix != 0.0:
            images = ((images/max_pix) + 1.0)/2.0
        else:
            images = (images + 1.0) / 2.0
            #print('inside')
    if(images.size()[1] == 1): # binary image
        images = torch.cat((images, images, images), 1)
    
    images = np.swapaxes(np.swapaxes(images.cpu().numpy(), 1, 2), 2, 3)

    idx=0
    fig = plt.figure()
    plt.imshow(images[idx])
    plt.savefig(os.path.join(plot_path, filename))
    plt.close()

def plot_one_sample_from_images(images, plot_path, filename, isRange01=False):
    ''' Plot images
    Changed 2020.11.23
        isRange01 is added to normalize image in different way. 

    Args: 
        images: (c, h, w), tensor in any range. (c=3 or 1)
        batch_size: int
        plot_path: string
        filename: string
        isRange01: True/False, Normalization will be different. 
    '''
    #print(torch.max(images), torch.min(images))
    if isRange01:
        '''max_pix = torch.max(torch.abs(images))
        images = images/max_pix'''
        images = images
        #print(max_pix, torch.min(torch.abs(images)))
        #print(torch.max(images))
    else:
        max_pix = torch.max(torch.abs(images))
        #print(max_pix)
        if max_pix != 0.0:
            images = ((images/max_pix) + 1.0)/2.0
        else:
            images = (images + 1.0) / 2.0
            #print('inside')
    if(images.size()[1] == 1): # binary image
        images = torch.cat((images, images, images), 1)
    
    images = np.swapaxes(np.swapaxes(torch.squeeze(images).cpu().numpy(), 0, 1), 1, 2)
    #print(np.shape(images))

    #print(filename)
    #print(os.path.join(plot_path, filename))
    idx=0
    plt.imsave(os.path.join(plot_path, filename), images)
    

    #fig = plt.figure()
    #plt.imshow(images[idx])
    #plt.savefig(os.path.join(plot_path, filename))
    #plt.close()


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            #correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def get_classification_TF(output, target):
    with torch.no_grad():
        _, pred = output.topk(1, 1, True, True)
        #print(pred.size(), pred.t().size())
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        #print(correct.size())
        return correct

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        ''' Multiplying n to val before summation is done because
        it is usually used for loss which is already mean with respect to batch size. 
        '''
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def get_PE_cart(batch_s, pe_s):
	'''
	Generates cartesian Positional Encoding. 
	Args: 
		batch_s: int, batch size. 
		pe_s: (int h, int w), size of Positional encoding feature 
	Return:
		pe: (batch_s, 2, pe_s[0], pe_s[1])
	'''
	lin_x = torch.unsqueeze(torch.linspace(-1.0, 1.0, steps=pe_s[1], device='cuda'), 0).repeat(pe_s[0], 1) # (192, 256)
	lin_y = torch.unsqueeze(torch.linspace(-1.0, 1.0, steps=pe_s[0], device='cuda'), 0).repeat(pe_s[1], 1) # (, 256)
	lin_y = lin_y.t()
	
	lin_x = torch.unsqueeze(lin_x, 0).repeat(batch_s, 1, 1) # (batch, fm_s, fm_s)
	lin_y = torch.unsqueeze(lin_y, 0).repeat(batch_s, 1, 1)
	# (b, h, w)

	pe = torch.cat((lin_x.unsqueeze(1), lin_y.unsqueeze(1)), 1)# (b, 2, h, w)
	return pe


def get_coord_feature(batch_s, input_s, pe_s, polar_grid=None):
    '''
    20201124
    Changed to have an option to return Cart PE. 

    code from (base) min@libigpu1:~/research_mk/attention_model_biliniear_localGlobal_6_reinforce/detach_recurrency_group/rl_base_recon_corHM_absHM_corrREINFORCE_corInitRNN_detachTD_hlr_lowResD_detachVD_removeBottleNeckV2I_conventionalResInc_contVer110

    Generate Positional encoding
    PE feature will first be generated in cartesian space with same size of input. 
    And then it will be transformed to polar space, if polar_grid not nont. 
    After that, it will be resized to pe_s. 

    Args:
        batch_s: int. Batch size
        input_s: (int h, int w), Size of loaded input image in cartesian space. 
        pe_s: (int h', int w'), Size of positional encoding feature in polar space. This size of feature will be returned. 
        polar_grid: Grid for polar transformation. It must be the same grid used in polar CNN. Fixation points must be already added to this grid.
    return:
        polar_pe_resized: (b, 2, h', w')
    '''

    with torch.no_grad():
        lin_x = torch.unsqueeze(torch.linspace(-1.0, 1.0, steps=input_s[1], device='cuda'), 0).repeat(input_s[0], 1) # (192, 256)
        lin_y = torch.unsqueeze(torch.linspace(-1.0, 1.0, steps=input_s[0], device='cuda'), 0).repeat(input_s[1], 1) # (, 256)
        lin_y = lin_y.t()
        #lin_r = torch.sqrt(lin_x**2 + lin_y**2)
        
        lin_x = torch.unsqueeze(lin_x, 0).repeat(batch_s, 1, 1) # (batch, fm_s, fm_s)
        lin_y = torch.unsqueeze(lin_y, 0).repeat(batch_s, 1, 1)
        #lin_r = torch.unsqueeze(lin_r, 0).repeat(batch_s, 1, 1) # (batch, fm_s, fm_s)
        # (b, h, w)

        cart_pe = torch.cat((lin_x.unsqueeze(1), lin_y.unsqueeze(1)), 1)# (b, 2, h, w)

        if polar_grid is not None:
            pe = torch.nn.functional.grid_sample(cart_pe, polar_grid, align_corners=False) # (b, 2, h', w')
        else:
            pe = cart_pe
        
        pe_resized =  torch.nn.functional.interpolate(pe, pe_s)
    
    return pe_resized

def set_w_requires_no_grad(model, tf=False):
    for param in model.parameters():
        param.requires_grad = tf

def load_state_dict_removing_string(model, load_dir, str_remove):
    modified_params = {}
    import torch
    params_dict = torch.load(load_dir)
    for k, v in params_dict.items():
        if str_remove in k:
            modified_params[k.replace(str_remove+'.', '')] = v
        else:
            modified_params[k] = v
    #del params_dict
    
    model.load_state_dict(modified_params)#, strict=False)



    '''for k, v in modified_params.items():
        print(k)
    
    diff = torch.sum(torch.abs(modified_params['encoder.0.weight'].cuda() - params_dict['module.model.encoder.0.weight'].cuda()))
    #diff = torch.sum(torch.abs(model.encoder[0].weight.cuda() - params_dict['module.model.encoder.0.weight'].cuda()))
    print('weight difference must be 0.0  in load : ', diff)'''

    return model

def check_load_params(model_after_load, load_dir):
    params_dict = torch.load(load_dir)

    #diff = torch.sum(torch.abs(model_after_load.encoder[0].weight.cuda() - params_dict['module.model.encoder.0.weight'].cuda()))
    diff = torch.sum(torch.abs(model_after_load.conv1_1.conv.weight.cuda() - params_dict['module.resnet.conv1_1.conv.weight'].cuda()))
    print('weight difference must be 0.0: ', diff)

def compare_params(param1, param2):
    for k, v in param1.items():
        print(k)
        print(torch.sum(torch.abs(param1[k] - param2[k])))

def compare_params_old(basedir):
    #basedir = '/home/libiadm/export/HDD1/minkyu/train/polar/pp_repl_fem_femb_ts'

    init_dir = os.path.join(basedir, 'Attn_e0b0.pt')
    train_dir = os.path.join(basedir, 'Attn_e20b0.pt')

    params_init = torch.load(init_dir)
    params_train = torch.load(train_dir)


    for k, v in params_init.items():
        print(k)
        print(torch.sum(torch.abs(params_init[k] - params_train[k])))


def normalize_min_max(fms):
    ''' Normalize input fms range from 0 to 1. 


    2023.01.24. Minkyu
        I made this function more stable by regulating (max_val - min_val) not close to 0. 

    Args: 
        fms: (b, c, h, w)
    return: 
        fms_norm: (b, c, h, w)
    '''
    fms_s = fms.size()
    if len(fms_s) == 3:
        fms = fms.unsqueeze(1)
        fms_s = fms.size()

    #print(fms_s)
    #print(torch.min(fms.view(fms_s[0], -1), 1)[0].size())
    min_val = torch.min(fms.view(fms_s[0], -1), 1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    max_val = torch.max(fms.view(fms_s[0], -1), 1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)

    max_min = max_val - min_val
    max_min[torch.logical_and(max_min>=0, max_min<=1e-4)] = 1

    fms_norm = (fms - min_val) / max_min
    assert not torch.isnan(fms_norm).any(), f"[Assert] from noralize_min_max in misc, nan is detected. "
    return fms_norm

def noralize_min_max(fms, isStable=False):
    ''' Normalize input fms range from 0 to 1. 


    2023.01.24. Minkyu
        I made this function more stable by regulating (max_val - min_val) not close to 0. 

    Args: 
        fms: (b, c, h, w)
    return: 
        fms_norm: (b, c, h, w)
    '''
    fms_s = fms.size()
    if len(fms_s) == 3:
        fms = fms.unsqueeze(1)
        fms_s = fms.size()

    #print(fms_s)
    #print(torch.min(fms.view(fms_s[0], -1), 1)[0].size())
    #min_val = torch.min(fms.view(fms_s[0], -1), 1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    #max_val = torch.max(fms.view(fms_s[0], -1), 1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    min_val = torch.min(fms.reshape(fms_s[0], -1), 1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    max_val = torch.max(fms.reshape(fms_s[0], -1), 1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)

    max_min = max_val - min_val
    #max_min[torch.logical_and(max_min>=0, max_min<=1e-4)] = 1

    fms_norm = (fms - min_val) / max_min
    #fms_norm = torch.nan_to_num(fms_norm)
    fms_norm[fms_norm != fms_norm] = 0
    assert not torch.isnan(fms_norm).any(), f"[Assert] from noralize_min_max in misc, nan is detected. "
    return fms_norm

def mark_point(imgs, fixs, ds=7, isRed=True):
    '''
    Mark a point in the given image. 
    Args:
        imgs: (b, 3, h, w), tensor, any range
        fixs: (b, 2), (float x, float y), tensor, -1~1
    return:
        img_marked: (b, 3, h, w)
    '''
    img_s = imgs.size()
    
    fixs = (fixs + 1)/2.0 # 0~1
    fixs[:,0] = fixs[:,0] * img_s[-1]
    fixs[:,1] = fixs[:,1] * img_s[-2]
    fixs = fixs.to(torch.int)

    #imgs = imgs * 0.5
    '''Edited 20200811. 
    Code below had wrong order of x, y coordinates. I simply changed 0-->1 and 1-->0 of fixs[b,n]. '''
    for b in range(img_s[0]):
        if isRed:
            imgs[b, :, fixs[b,1]-ds:fixs[b,1]+ds, fixs[b,0]-ds:fixs[b,0]+ds] = 0.0
            imgs[b, 0, fixs[b,1]-ds:fixs[b,1]+ds, fixs[b,0]-ds:fixs[b,0]+ds] = 2.0
        else:
            imgs[b, :, fixs[b,1]-ds:fixs[b,1]+ds, fixs[b,0]-ds:fixs[b,0]+ds] = 0.0
            imgs[b, 2, fixs[b,1]-ds:fixs[b,1]+ds, fixs[b,0]-ds:fixs[b,0]+ds] = 2.0
    return imgs


def mark_fixations(imgs, fixs, ds=7, isRed=True):
    '''
    Mark fixation points in the given images. This function is used to mark a fixation. 
    Args:
        imgs: (b, 3, h, w), tensor, any range
        fixs: (b, 2), (float x, float y), tensor, -1~1
    return:
        img_marked: (b, 3, h, w)
    '''

    imgs = noralize_min_max(imgs)
    imgs = mark_point(imgs, fixs, ds=ds, isRed=isRed)

    return (imgs -0.5)*2.0

def mark_fixations_history(imgs, fixs_h, ds=21, isLastRed=True):
    '''
    Mark fixation history in the given images. This function is used to mark fixation history. 
    Args:
        imgs: (b, 3, h, w), tensor, any range
        fixs: (b, step, 2), (float x, float y), tensor, -1~1
    return:
        img_marked: (b, 3, h, w)
    '''
    n_steps = fixs_h.size(1)
    imgs = noralize_min_max(imgs)
    img_m = imgs
    for step in range(n_steps):
        if step == n_steps-1:
            imgs = mark_point(imgs, fixs_h[:, step, :], ds=ds, isRed=isLastRed)
        else:
            imgs = mark_point(imgs, fixs_h[:, step, :], ds=ds, isRed=False)

    return (imgs -0.5)*2.0

def heatmap_generator(pred, feature, fc_weight, top_N_map=1):
    ### 1. get top-N class index (self.top_N_map)
    # pred: (batch, #class)
    # feature: (batch, #inputFM, w, h)
    sorted_logit, sorted_index = torch.sort(torch.squeeze(pred), dim=1,  descending=True)
    # sorted_index: (batch, #top_N_map)

    selected_weight = torch.cat([torch.index_select(fc_weight, 0, idx).unsqueeze(0)
        for idx in  sorted_index[:, 0:top_N_map]])
    # weight: (#class, #inputFM)
    # selected_weight: (batch, top_N_map, #inputFM)

    s = feature.size()
    cams = torch.abs(torch.squeeze(torch.bmm(selected_weight, feature.view(s[0], s[1], s[2]*s[3]))))
    cams_prob = cams.view(s[0], top_N_map, s[2], s[3]) #* sorted_prob.unsqueeze(2).unsqueeze(2)
    heatmap = torch.sum(cams_prob, 1)
    return heatmap, sorted_index
    

def save_caption(captions, ref, epoch, step, vocab, isTrain=True):
    ''' 
    Convert predicted captions in idx to words
    ref: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/sample.py
    Args:
        captions: (batch, max_seq_length), list of list, in word idx, predicted caption
        ref: (batch, max_seq_length), list of list, in word idx, reference caption
        epoch: int
        step: int
        vocab: vocab
    return:
        None
    '''
    if isTrain:
        fp = open('caption_train_e'+str(epoch)+'s'+str(step), 'w')
    else:
        fp = open('caption_test_e'+str(epoch)+'s'+str(step), 'w')
    bs = len(captions)
    #captions = captions.cpu().numpy()
    
    for b in range(bs):
        s_pred = convert_idx2word(captions[b], vocab)
        s_ref = convert_idx2word(ref[b][0], vocab)
        
        fp.write('batch: ' + str(b) + '\n')
        fp.write('    Pred: ' + s_pred + '\n')
        fp.write('    Ref:  ' + s_ref + '\n')
    fp.close()

            
def convert_idx2word(caption, vocab):
    '''
    convert given sentence in idx to words.
    Args:
        caption: (length, ), numpy
        vocab: vocab
    return:
        words: string
    '''
    sentence = []
    for word_id in caption:
        word = vocab.idx2word[word_id]
        sentence.append(word)
        if word == '<end>':
            break
    sentence_j = ' '.join(sentence)
    return sentence_j

def remove_pads_sentence(caption, vocab):
    '''remove pads in a given sentence
    Args: 
        captions: (max_seq_length), list, including word idx
    return:
        caption_clear: (seq_length_without_pad), list
    '''
    caption_clear = [w for w in caption if w not in [0, 1, 2, 3]]
    '''for i, word_id in enumerate(caption):
        word = vocab.idx2word[word_id]
        if word == '<end>':
            break'''
    return caption_clear

def remove_pads_batch(captions, vocab, isTarget=False):
    '''remove pads in a given sentence batch
    Args: 
        captions: (batch, max_seq_length), numpy, including word idx
    return:
        captions_clear: (batch, varying seq_length_without_pad), list of list
    '''
    bs = len(captions)
    captions_clear = list()
    for b in range(bs):
        cap_clear = remove_pads_sentence(captions[b], vocab)
        if isTarget:
            captions_clear.append([cap_clear])
        else:
            captions_clear.append(cap_clear)


    return captions_clear

def make_sequential_mask(lengths, device='cuda'):
    ''' make sequential mask
    see http://juditacs.github.io/2018/12/27/masked-attention.html
    Args:
        lengths: (batch, ), including length of sequence
    return 
        mask: (batch, max_seq_length)
    '''
    maxlen = np.max(lengths)
    lengths = torch.tensor(lengths, device=device)
    mask = torch.arange(maxlen, device=device)[None, :] < lengths[:, None]
    return mask


def load_image(image_path, transform=None):
    from PIL import Image
    image = Image.open(image_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)
            
    return image


def add_heatmap_on_image(heatmap, image):
    '''Visualize heatmap on image. This function is not based on batch.
    Args:
        heatmap: (h, w), 0~1 ranged numpy array
        image: (3, h, w), 0~1 ranged numpy array
        heatmap and image must be in the same size. 
    return:
        hm_img: (h, w, 3), 0~255 ranged numy array
    '''
    #print(np.shape(image), np.shape(heatmap))
    heatmap_cv = heatmap * 255
    heatmap_cv = cv2.applyColorMap(heatmap_cv.astype(np.uint8), cv2.COLORMAP_JET) #(h, w, 3)
    heatmap_cv = cv2.cvtColor(heatmap_cv, cv2.COLOR_BGR2RGB)
    image_cv = np.swapaxes(np.swapaxes(image, 0, 1), 1, 2)*255
    #print(np.shape(image_cv), np.shape(heatmap_cv))
    hm_img = cv2.addWeighted(heatmap_cv, 0.7, image_cv.astype(np.uint8), 0.3, 0)
    
    return hm_img

def add_heatmap_on_image_tensor(heatmap, image, resize_s=(112,112), isNormHM=True, device='cpu'):
    '''Visualize heatmap on image. This function works based on batched tensors
    Args:
        heatmap: (b, h, w), any ranged tensor
        image: (b, 3, h, w), any ranged tensor
        resize_s: (int, int), heatmap and image will be resized to this size
        isNormHM: True/False, if True, heatmap will be normalized to 0~1
    return:
        hm_img: (b, 3, h, w), 0~1 ranged tensor
    '''
    ret = []
    bs = image.size(0)

    #heatmap = torch.squeeze(heatmap)
    heatmap = heatmap.unsqueeze(1) #(b, 1, h, w)
    if resize_s is not None:
        #print(heatmap.size())
        heatmap = torch.nn.functional.interpolate(heatmap, resize_s, mode='bilinear') 
        image = torch.nn.functional.interpolate(image, resize_s, mode='bilinear') 

    if isNormHM:
        heatmap = noralize_min_max(heatmap)
    image = noralize_min_max(image)

    for b in range(bs):
        hm_i = torch.squeeze(heatmap[b]).cpu().numpy()
        image_i =image[b].cpu().numpy()
        hmimg = add_heatmap_on_image(hm_i, image_i)
        ret.append(hmimg)
    ret = np.stack(ret, axis=0) # 0~255 ranged numpy array (b, h, w, 3)
    ret = np.swapaxes(np.swapaxes(ret.astype(np.float32), 2, 3), 1, 2) # 0~255 ranged numpy array(b, 3, h, w)
    ret = torch.tensor(ret/255.0, device=device)

    #print(np.shape(ret))
    return ret



def wrapper_hm(hm, image):
    ret = []
    hm = misc.noralize_min_max(hm)
    image = misc.noralize_min_max(image)

    for b in range(n_plots):
        hm_i = torch.squeeze(hm[b]).cpu().numpy()
        image_i =image[b].cpu().numpy()
        hmimg = misc.add_heatmap_on_image(hm_i, image_i)
        ret.append(hmimg)
    ret = np.stack(ret, axis=0)
    #print(np.shape(ret))
    return ret

def check_gradients(model):
    '''20201124
    return model's gradient stats
    https://discuss.pytorch.org/t/get-the-gradient-of-the-network-parameters/50575
    '''

    max_max = 0
    max_mean = 0
    max_val = 0
    for p in model.parameters():
        mx = torch.max(torch.abs(p.grad))
        if max_max < mx:
            max_max = mx

        mn = torch.mean(torch.abs(p.grad))
        if max_mean < mn:
            max_mean = mn
    return max_max, max_mean, max_val

def get_one_patch(images, action_sequence, patch_size):
    """
    Get small patch of the original image
    image: tensor, (b, 3, h, w)
    action_sequence: tensor, (b, 2), -1~1, (x, y)
    patch_size: int, pixel level size
    """
    batch_size = images.size(0)
    image_size = images.size(2)
    patch_size_half = int(patch_size/2)

    patch_coordinate = ((action_sequence + 1.0)/2.0 * image_size).int()
    #patch_coordinate = torch.floor(action_sequence * (image_size - patch_size)).int()
    patches = []
    for i in range(batch_size):
        per_patch = images[i, :,
                ((patch_coordinate[i, 1] - patch_size_half).item()).int(): ((patch_coordinate[i, 1] + patch_size_half).item()).int(),
                ((patch_coordinate[i, 0] - patch_size_half).item()).int(): ((patch_coordinate[i, 0] + patch_size_half).item()).int()]

        patches.append(per_patch.view(1, per_patch.size(0), per_patch.size(1), per_patch.size(2)))

    return torch.cat(patches, 0)


def get_theta(attn, scale_attn, batch_s):
    '''
    code from libigpu3:~/research_mk/imagenet_new/sumReluHeatmap_fullD_oneshot/train.py
    attn (tensor): (batchx2), (-1~1)
    '''
    a1x = torch.ones(batch_s, 1, 1, requires_grad=False, device='cuda')*scale_attn#(self.v_img_s[2]/self.img_size_o) # (batch, 1, 1)
    a1y = torch.zeros(batch_s, 1, 1, requires_grad=False, device='cuda')
    a2x = torch.zeros(batch_s, 1, 1, requires_grad=False, device='cuda')
    a2y = torch.ones(batch_s, 1, 1, requires_grad=False, device='cuda')*scale_attn#(self.v_img_s[1]/self.img_size_o)

    a1 = torch.cat((a1x, a1y), 1)#.cuda()#, requires_grad=False) # (b, 2, 1)
    a2 = torch.cat((a2x, a2y), 1)#.cuda()#, requires_grad=False)
    #print('get_theta', a1.is_cuda, a2.is_cuda, attn.is_cuda)
    #print(a1.size(), a2.size(), attn.size()) 
    #theta = torch.cat((a1, a2, torch.unsqueeze(torch.squeeze(attn.cuda()), 2)), 2) # (b, 2, 3)
    theta = torch.cat((a1, a2, torch.unsqueeze(attn.cuda(), 2)), 2) # (b, 2, 3)
    return theta#.cuda()

def get_glimpses_old(images, fixs_xy, patch_s, crop_ratio=[0.25, 0.5, 1.0]):
    '''
    code from libigpu3:~/research_mk/imagenet_new/sumReluHeatmap_fullD_oneshot/train.py
    fixs_xy: tensor, (b, 2), -1~1

    2021/6/22: After Neurips submission, this code is changed to use mode=bilinear, instead of nearest. 
        It was nearest but nearest seems to obfuscate gradients. 
    '''
    mode = 'nearest'
    batch_s = images.size(0)
    theta_highres = get_theta(fixs_xy, crop_ratio[0], batch_s)
    grid_highres = torch.nn.functional.affine_grid(theta_highres, torch.Size((batch_s, 3, patch_s*2, patch_s*2)))#, device='cuda')
    img_highres = torch.nn.functional.grid_sample(images, grid_highres, mode='bilinear')
    img_highres = torch.nn.functional.interpolate(img_highres, (patch_s, patch_s), mode=mode)

    theta_midres = get_theta(fixs_xy, crop_ratio[1], batch_s)
    grid_midres = torch.nn.functional.affine_grid(theta_midres, torch.Size((batch_s, 3, patch_s*2, patch_s*2)))#, device='cuda')
    img_midres = torch.nn.functional.grid_sample(images, grid_midres, mode='bilinear')
    img_midres = torch.nn.functional.interpolate(img_midres, (patch_s, patch_s), mode=mode)

    theta_lowres = get_theta(fixs_xy, crop_ratio[2], batch_s)
    grid_lowres = torch.nn.functional.affine_grid(theta_lowres, torch.Size((batch_s, 3, patch_s*2, patch_s*2)))#, device='cuda')
    img_lowres = torch.nn.functional.grid_sample(images, grid_lowres, mode='bilinear')
    img_lowres = torch.nn.functional.interpolate(img_lowres, (patch_s, patch_s), mode=mode)

    return img_highres, img_midres, img_lowres

def get_glimpses_old_old(images, fixs_xy, patch_s, crop_ratio=[0.25, 0.5, 1.0]):
    '''
    code from libigpu3:~/research_mk/imagenet_new/sumReluHeatmap_fullD_oneshot/train.py
    fixs_xy: tensor, (b, 2), -1~1

    2021/6/22: After Neurips submission, this code is changed to use mode=bilinear, instead of nearest. 
        It was nearest but nearest seems to obfuscate gradients. 
    '''
    mode = 'nearest'
    batch_s = images.size(0)
    theta_highres = get_theta(fixs_xy, crop_ratio[0], batch_s)
    grid_highres = torch.nn.functional.affine_grid(theta_highres, torch.Size((batch_s, 3, patch_s*2, patch_s*2)))#, device='cuda')
    img_highres = torch.nn.functional.grid_sample(images, grid_highres, mode='nearest')
    img_highres = torch.nn.functional.interpolate(img_highres, (patch_s, patch_s), mode=mode)

    theta_midres = get_theta(fixs_xy, crop_ratio[1], batch_s)
    grid_midres = torch.nn.functional.affine_grid(theta_midres, torch.Size((batch_s, 3, patch_s*2, patch_s*2)))#, device='cuda')
    img_midres = torch.nn.functional.grid_sample(images, grid_midres, mode='nearest')
    img_midres = torch.nn.functional.interpolate(img_midres, (patch_s, patch_s), mode=mode)

    theta_lowres = get_theta(fixs_xy, crop_ratio[2], batch_s)
    grid_lowres = torch.nn.functional.affine_grid(theta_lowres, torch.Size((batch_s, 3, patch_s*2, patch_s*2)))#, device='cuda')
    img_lowres = torch.nn.functional.grid_sample(images, grid_lowres, mode='nearest')
    img_lowres = torch.nn.functional.interpolate(img_lowres, (patch_s, patch_s), mode=mode)

    return img_highres, img_midres, img_lowres

def get_glimpses(images, fixs_xy, patch_s, crop_ratio=[0.25, 0.5, 1.0]):
    '''
    code from libigpu3:~/research_mk/imagenet_new/sumReluHeatmap_fullD_oneshot/train.py
    fixs_xy: tensor, (b, 2), -1~1

    2021/6/22: After Neurips submission, this code is changed to use mode=bilinear, instead of nearest. 
        It was nearest but nearest seems to obfuscate gradients. 
    '''
    mode = 'bilinear'
    batch_s = images.size(0)
    theta_highres = get_theta(fixs_xy, crop_ratio[0], batch_s)
    grid_highres = torch.nn.functional.affine_grid(theta_highres, torch.Size((batch_s, 3, patch_s*2, patch_s*2)))#, device='cuda')
    img_highres = torch.nn.functional.grid_sample(images, grid_highres, mode='bilinear')
    img_highres = torch.nn.functional.interpolate(img_highres, (patch_s, patch_s), mode=mode)

    theta_midres = get_theta(fixs_xy, crop_ratio[1], batch_s)
    grid_midres = torch.nn.functional.affine_grid(theta_midres, torch.Size((batch_s, 3, patch_s*2, patch_s*2)))#, device='cuda')
    img_midres = torch.nn.functional.grid_sample(images, grid_midres, mode='bilinear')
    img_midres = torch.nn.functional.interpolate(img_midres, (patch_s, patch_s), mode=mode)

    theta_lowres = get_theta(fixs_xy, crop_ratio[2], batch_s)
    grid_lowres = torch.nn.functional.affine_grid(theta_lowres, torch.Size((batch_s, 3, patch_s*2, patch_s*2)))#, device='cuda')
    img_lowres = torch.nn.functional.grid_sample(images, grid_lowres, mode='bilinear')
    img_lowres = torch.nn.functional.interpolate(img_lowres, (patch_s, patch_s), mode=mode)

    return img_highres, img_midres, img_lowres

def get_glimpses_new(images, fixs_xy, patch_s, crop_ratio=[0.25, 0.5, 1.0]):
    '''
    code from libigpu3:~/research_mk/imagenet_new/sumReluHeatmap_fullD_oneshot/train.py
    fixs_xy: tensor, (b, 2), -1~1

    2021/6/22: After Neurips submission, this code is changed to use mode=bilinear, instead of nearest. 
        It was nearest but nearest seems to obfuscate gradients. 
    '''
    mode = 'bilinear'
    batch_s = images.size(0)
    theta_highres = get_theta(fixs_xy, crop_ratio[0], batch_s)
    grid_highres = torch.nn.functional.affine_grid(theta_highres, torch.Size((batch_s, 3, patch_s, patch_s)))#, device='cuda')
    img_highres = torch.nn.functional.grid_sample(images, grid_highres, mode='bilinear')
    #img_highres = torch.nn.functional.interpolate(img_highres, (patch_s, patch_s), mode=mode)

    theta_midres = get_theta(fixs_xy, crop_ratio[1], batch_s)
    grid_midres = torch.nn.functional.affine_grid(theta_midres, torch.Size((batch_s, 3, patch_s, patch_s)))#, device='cuda')
    img_midres = torch.nn.functional.grid_sample(images, grid_midres, mode='bilinear')
    #img_midres = torch.nn.functional.interpolate(img_midres, (patch_s, patch_s), mode=mode)

    theta_lowres = get_theta(fixs_xy, crop_ratio[2], batch_s)
    grid_lowres = torch.nn.functional.affine_grid(theta_lowres, torch.Size((batch_s, 3, patch_s, patch_s)))#, device='cuda')
    img_lowres = torch.nn.functional.grid_sample(images, grid_lowres, mode='bilinear')
    #img_lowres = torch.nn.functional.interpolate(img_lowres, (patch_s, patch_s), mode=mode)

    return img_highres, img_midres, img_lowres


def spatial_basis(height=28, width=28, channels=64):
    '''
    code from 
        https://github.com/cjlovering/interpretable-reinforcement-learning-using-attention/blob/master/torchbeast/attention_net.py
    '''
    """
    NOTE: The `height` and `weight` depend on the inputs' size and its resulting size
    after being processed by the vision network.
    """

    h, w, d = height, width, channels

    p_h = torch.mul(
            torch.arange(1, h + 1).unsqueeze(1).float(), torch.ones(1, w).float()
            ) * (np.pi / h)
    p_w = torch.mul(
            torch.ones(h, 1).float(), torch.arange(1, w + 1).unsqueeze(0).float()
            ) * (np.pi / w)

    # TODO: I didn't quite see how U,V = 4 made sense given that the authors form the spatial
    # basis by taking the outer product of the values.
    # I am not confident in this step.
    U = V = 8  # size of U, V.
    u_basis = v_basis = torch.arange(1, U + 1).unsqueeze(0).float()
    a = torch.mul(p_h.unsqueeze(2), u_basis)
    b = torch.mul(p_w.unsqueeze(2), v_basis)
    out = torch.einsum("hwu,hwv->hwuv", torch.cos(a), torch.cos(b)).reshape(h, w, d)

    return out.permute(2, 0, 1)

