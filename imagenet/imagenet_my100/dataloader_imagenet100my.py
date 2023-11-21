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
plt.switch_backend('agg') 
#from mpl_toolkits.axes_grid1 import ImageGrid
#from sklearn.utils import linear_assignment_
#from scipy.stats import itemfreq
#from sklearn.cluster import KMeans
#from itertools import chain

# This is required for salicon dataset
#import datasetSALICON as ds

def load_imagenet_myclass100_noNormalize(batch_size, img_s_load=512, img_s_return=448, server_type='libigpu5', isRandomResize=True, num_workers=4, num_workers_t=None, shuffle_test=True):
    if server_type == 'libigpu4':
        path = '/home/choi574/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu5':
        path = '/home/choi574/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu6':
        path = '/home/choi574/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu7':
        path = '/home/choi574/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu0':
        path = '/home/min/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu1':
        path = '/home/min/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu3':
        path = '/home/min/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu2':
        path = '/home/choi574/datasets/ImageNet2012_myclass100/'
    elif server_type == 'greatlake':
        path = '/tmp/minkyu/ImageNet2012_myclass100/'
    else:
        print("undefined server type")
    print('================ MY CLASS 100 IMGNET =================== ')

    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if isRandomResize:
        print('load imagenet with RandomResize')
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(img_s_return),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            #normalize
            ]) 
    else:
        print('load imagenet without RandomResize')
        train_transforms = transforms.Compose([
            transforms.Resize(img_s_load),
            transforms.CenterCrop(img_s_return),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            #normalize
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
            transforms.ToTensor()
            #normalize
        ]))

    if num_workers_t == None:
        num_workers_t = num_workers
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=shuffle_test,
        num_workers=num_workers_t, pin_memory=True, drop_last=True)
    return train_loader, test_loader, 100



def load_imagenet_full1000(batch_size, img_s_load=512, img_s_return=448, server_type='libigpu5', isRandomResize=True, num_workers=4, num_workers_t=None, shuffle_test=True):
    if server_type == 'libigpu4':
        path = '/home/choi574/datasets/ImageNet_full_downloaded_20230129/'
    if server_type == 'libigpu8':
        path = '/data/datasets/ImageNet2012/'
    if server_type == 'libilab':
        path = '/datasets/ImageNet2012/'
    if server_type == 'libigpu6':
        path = '/home/choi574/datasets/ImageNet2012/'
    if server_type == 'libigpu5' or server_type == 'libigpu7':
        path = '/home/choi574/datasets/ImageNet_full_downloaded_20230129/'
        #path = '/home/choi574/datasets/ImageNet2012/'
    if server_type == 'libilab':
        path = '/datasets/ImageNet_full_downloaded_20230129/'
    elif server_type != 'greatlake':
        path = '/home/libiadm/datasets/ImageNet2012/'
    elif server_type == 'greatlake':
        path = '/tmpssd/minkyu/ImageNet2012/'
    else:
        print("undefined server type")
    print('================ MY CLASS 100 IMGNET =================== ')

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
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=shuffle_test,
        num_workers=num_workers_t, pin_memory=True, drop_last=True)
    return train_loader, test_loader, 1000





class TransformsSimCLR:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    
    from https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/transformations/simclr.py
    """

    def __init__(self, size):
        s = 1
        color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=size),
                transforms.RandomHorizontalFlip(),  # with 0.5 probability
                #torchvision.transforms.RandomApply([color_jitter], p=0.8),
                #torchvision.transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.Resize(size=size),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)

def load_imagenet_full1000_contrastiveLearning(batch_size, img_s_load=512, img_s_return=448, server_type='libigpu5', num_workers=4, num_workers_t=None, shuffle_test=True):
    if server_type == 'libigpu8':
        print('libigpu8--------------------')
        path = '/data/datasets/ImageNet2012/'
        print(path, server_type)
    if server_type == 'libigpu4':
        path = '/home/choi574/datasets/ImageNet_full_downloaded_20230129/'
    if server_type == 'libilab':
        path = '/datasets/ImageNet2012/'
    if server_type == 'libigpu6':
        path = '/home/choi574/datasets/ImageNet2012/'
    if server_type == 'libigpu5' or server_type == 'libigpu7':
        path = '/home/choi574/datasets/ImageNet_full_downloaded_20230129/'
    if server_type == 'libilab':
        path = '/datasets/ImageNet_full_downloaded_20230129/'
    #elif server_type != 'greatlake':
    #    path = '/home/libiadm/datasets/ImageNet2012/'
    elif server_type == 'greatlake':
        path = '/tmpssd/minkyu/ImageNet2012/'
    else:
        print("undefined server type")



    train_data = datasets.ImageFolder(root=os.path.expanduser(path + 'train/'),
                                        transform=TransformsSimCLR(img_s_return))
    test_data = datasets.ImageFolder(root=os.path.expanduser(path + 'val/'),
                                        transform=TransformsSimCLR(img_s_return))
    
    if num_workers_t == None:
        num_workers_t = num_workers
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=shuffle_test,
        num_workers=num_workers_t, pin_memory=True, drop_last=True)
    return train_loader, test_loader, 1000



def load_imagenet_myclass100(batch_size, img_s_load=512, img_s_return=448, server_type='libigpu5', isRandomResize=True, num_workers=4, num_workers_t=None, shuffle_test=True):
    if server_type == 'libigpu4':
        path = '/home/choi574/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu5':
        path = '/home/choi574/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu6':
        path = '/home/choi574/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu7':
        path = '/home/choi574/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu0':
        path = '/home/min/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu1':
        path = '/home/min/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu3':
        path = '/home/min/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu2':
        path = '/home/choi574/datasets/ImageNet2012_myclass100/'
    elif server_type == 'lambda':
        path = '/home/ubuntu/datasets/ImageNet2012_myclass100/'
    elif server_type == 'greatlake':
        path = '/home/minkyu/ImageNet2012_myclass100/'
    else:
        print("undefined server type")
    print('================ MY CLASS 100 IMGNET =================== ')

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
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=shuffle_test,
        num_workers=num_workers_t, pin_memory=True, drop_last=True)
    return train_loader, test_loader, 100

def load_imagenet_myclass100_DDP(batch_size, img_s_load=512, img_s_return=448, server_type='libigpu5', isRandomResize=True, num_workers=4, num_workers_t=None, shuffle_test=True):
    if server_type == 'libigpu4':
        path = '/home/choi574/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu2':
        path = '/home/choi574/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu5':
        path = '/home/choi574/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu6':
        path = '/home/choi574/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu7':
        path = '/home/choi574/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu0':
        path = '/home/min/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu1':
        path = '/home/min/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu3':
        path = '/home/min/datasets/ImageNet2012_myclass100/'
    else:
        print("undefined server type")

    print('================ MY CLASS 100 IMGNET =================== ')
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

    if num_workers_t == None:
        num_workers_t = num_workers

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=shuffle_test,
        num_workers=num_workers_t, pin_memory=True, drop_last=True)
    return train_loader, test_loader, 100, train_sampler


def load_imagenet_myclass100_for_AdvAttacks(batch_size=1, img_s_load=256+128, img_s_return=224+112, server_type='libigpu5', isRandomResize=True, 
        num_workers=1, shuffle_test=False, isReturnInfo=False):
    if server_type == 'libigpu4':
        path = '/home/choi574/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu5':
        path = '/home/choi574/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu6':
        path = '/home/choi574/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu7':
        path = '/home/choi574/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu0':
        path = '/home/min/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu1':
        path = '/home/min/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu3':
        path = '/home/min/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu2':
        path = '/home/choi574/datasets/ImageNet2012_myclass100/'
    elif server_type == 'greatlake':
        path = '/tmp/minkyu/ImageNet2012_myclass100/'
    
    else:
        print("undefined server type")
    print('================ MY CLASS 100 IMGNET =================== ')

    test_data =  datasets.ImageFolder(root=os.path.expanduser(path + 'val/'),
        transform=transforms.Compose([
            transforms.Resize(img_s_load),
            transforms.CenterCrop(img_s_return),
            transforms.ToTensor()
        ]))
    lu_c2i = test_data.class_to_idx
    lu_c = test_data.classes
    #print(lu_c2i)
    #print(lu_c)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=shuffle_test,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    if isReturnInfo:
        return test_loader, lu_c2i, lu_c 
    else:
        return test_loader 



def load_imagenet_myclass100_for_AdvAttacks_crossTest(path, batch_size=144, img_s_return=224+112, server_type='libigpu5', 
        num_workers=2, shuffle_test=False):
    '''if server_type == 'libigpu4':
        path = '/home/choi574/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu5':
        path = '/home/choi574/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu6':
        path = '/home/choi574/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu7':
        path = '/home/choi574/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu0':
        path = '/home/min/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu1':
        path = '/home/min/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu3':
        path = '/home/min/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu2':
        path = '/home/choi574/datasets/ImageNet2012_myclass100/'
    else:
        print("undefined server type")'''
    print('================ MY CLASS 100 IMGNET =================== ')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_data =  datasets.ImageFolder(root=os.path.expanduser(path),
        transform=transforms.Compose([
            transforms.Resize(img_s_return),
            transforms.ToTensor(), 
            normalize
        ]))

    #print(test_data.class_to_idx)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=shuffle_test,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    return test_loader 

def load_imagenet_myclass100_for_AdvAttacks_crossTest_noResize(path, batch_size=144, server_type='libigpu5', 
        num_workers=2, shuffle_test=False):
    '''if server_type == 'libigpu4':
        path = '/home/choi574/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu5':
        path = '/home/choi574/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu6':
        path = '/home/choi574/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu7':
        path = '/home/choi574/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu0':
        path = '/home/min/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu1':
        path = '/home/min/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu3':
        path = '/home/min/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu2':
        path = '/home/choi574/datasets/ImageNet2012_myclass100/'
    else:
        print("undefined server type")'''
    print('================ MY CLASS 100 IMGNET =================== ')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_data =  datasets.ImageFolder(root=os.path.expanduser(path),
        transform=transforms.Compose([
            transforms.ToTensor(), 
            normalize
        ]))

    #print(test_data.class_to_idx)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=shuffle_test,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    return test_loader 


def test(batch_size=144, img_s_return=224+112, server_type='libigpu5', 
        num_workers=2, shuffle_test=False):
    if server_type == 'libigpu4':
        path = '/home/choi574/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu5':
        path = '/home/choi574/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu6':
        path = '/home/choi574/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu7':
        path = '/home/choi574/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu0':
        path = '/home/min/datasets/ImageNet2012_myclass100/val/'
    elif server_type == 'libigpu1':
        path = '/home/min/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu3':
        path = '/home/min/datasets/ImageNet2012_myclass100/'
    elif server_type == 'libigpu2':
        path = '/home/choi574/datasets/ImageNet2012_myclass100/'
    else:
        print("undefined server type")
    print('================ MY CLASS 100 IMGNET =================== ')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_data =  datasets.ImageFolder(root=os.path.expanduser(path),
        transform=transforms.Compose([
            transforms.Resize(img_s_return),
            transforms.ToTensor(), 
            normalize
        ]))

    print(test_data.class_to_idx)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=shuffle_test,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    return test_loader 

