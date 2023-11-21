import os
import torch
import torch.utils.data as data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import numpy as np
import cv2

import sys
sys.path.append('/home/choi574/coco/PythonAPI')
#sys.path.append('/mnt/lls/local_export/3/home/choi574/cocoapi/PythonAPI')
from pycocotools.coco import COCO
sys.path.append('/mnt/lls/home/choi574/git_libs/misc/')
sys.path.append('/home/cminkyu/git_libs/misc/')
import datasetSALICON


'''
2023.01.17. Minkyu
This code is for loading MSCOCO and SALICON at the same time. 
Because salicon is from MSCOCO and smaller than MSCOCO, 
Images shared for both MSCOCO and salicon will be loaded here. 
'''
def inspect_MSCOCO():
    '''
    2023.01.17. Minkyu
    This function is an example of inspecting MSCOCO's json file. 
    Based on the json file structure and info, the point to change in the code will be determined.  
    run it from libigpu5'''
    
    import sys
    import json

    f = open('/home/choi574/datasets/mscoco/data/annotations/instances_val2014.json')
    data = json.load(f)
    
    # Check what is in json file (keys)
    print(data.keys())
    
    # check annotations' length
    print(len(data['annotations']))
    # It seems the length is same as the number of all images
    
    # check the info of the first image
    print(data['annotations'][0]) 
    # >> {'segmentation': [[239.97, 260.24, 222.04, 270.49, 199.84, 253.41, 213.5, 227.79, 259.62, 200.46, 274.13, 202.17, 277.55, 210.71, 249.37, 253.41, 237.41, 264.51, 242.54, 261.95, 228.87, 271.34]], 
    #       'area': 2765.1486500000005, 
    #       'iscrowd': 0, 
    #       'image_id': 558840, 
    #       'bbox': [199.84, 200.46, 77.71, 70.88], 
    #       'category_id': 58, 
    #       'id': 156}
    print(data['annotations'][1])
    # >> {'segmentation': [[247.71, 354.7, 253.49, 346.99, 276.63, 337.35, 312.29, 333.49, 364.34, 331.57, 354.7, 327.71, 369.16, 325.78, 376.87, 333.49, 383.61, 330.6, 379.76, 321.93, 365.3, 320.0, 356.63, 317.11, 266.02, 331.57, 260.24, 334.46, 260.24, 337.35, 242.89, 338.31, 234.22, 338.31, 234.22, 348.92, 239.04, 353.73, 248.67, 355.66, 252.53, 353.73]], 
    #       'area': 1545.4213000000007, 
    #       'iscrowd': 0, 
    #       'image_id': 200365, 
    #       'bbox': [234.22, 317.11, 149.39, 38.55], 
    #       'category_id': 58, 
    #       'id': 509}

    # Here, image_id is the one that appears on the file name. 
    print(data['category']) # this is just the full categories (info for all 80 categories)
    print(data['images']) # this is not important. 
    
    
    
    
class CocoDetection(datasets.coco.CocoDetection):
    ''' 2021.02.06
    https://github.com/allenai/elastic/blob/master/multilabel_classify.py
    '''
    def __init__(self, root, annFile, 
                 path_salicon_in, path_salicon_target,
                 path_imgIds, 
                 transform=None, target_transform=None, 
                 size_img=(480, 640)):
        '''
        2023.01.17. Minkyu
        Made modification for loading salicon. 
        For path_imgIds, see choi574@libigpu5:/mnt/lls/home/choi574/git_libs/misc/salicon_MSCOCO. 
        
        Args:
            root: str, path to MSCOCO images. It already includes 'train', or 'val'
            annFile: str, path to annotation files. 
            path_salicon_in: str, path to salicon dataset's input images. 
            path_salicon_target: str, path to salicon dataset's target images. 
            path_imgIds: str, path to imgIds collected from salicon. 
        '''
        self.size_img = size_img
        self.path_salicon_in = path_salicon_in
        self.path_salicon_target = path_salicon_target
        self.root = root
        self.coco = COCO(annFile)
        self.ids = np.load(path_imgIds)
        #self.ids = list(self.coco.imgs.keys())
        # from this, self.ids contains all the image_ids that are included in the dataset. 
        # self.ids can be changed to only include the image_ids that I want to load. 
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)
        # print(self.cat2cat)
        # >> {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79}
        
    def __getitem__(self, index):
        coco = self.coco
        img_id = int(self.ids[index])
        '''
        2023.01.17. It took me some time to debug an issue. It keeps returning None from the code below. 
            coco.loadImgs(img_id)
        The issue was caused because img_id was not int. Its type is numpy int, not int. 
        So, type conversion is needed here. 
        '''
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        output = torch.zeros((3, 80), dtype=torch.long)
        for obj in target:
            if obj['area'] < 32 * 32:
                output[0][self.cat2cat[obj['category_id']]] = 1
            elif obj['area'] < 96 * 96:
                output[1][self.cat2cat[obj['category_id']]] = 1
            else:
                output[2][self.cat2cat[obj['category_id']]] = 1
        target = output

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            
            
        ### Load Salicon Data ###
        dir_sal_in = os.path.join(self.path_salicon_in, path)
        dir_sal_target = os.path.join(self.path_salicon_target, path.replace('jpg', 'png'))
        image = cv2.imread(dir_sal_in)
        saliency = cv2.imread(dir_sal_target, 0)
        img_sal, target_sal = datasetSALICON.imageProcessing(image, saliency, self.size_img, self.size_img)
        
        return img, target, img_sal, target_sal


def load_mscoco_multilabel_salicon(batch_size, img_s_return=224, server_type='libigpu5', 
                           num_workers=4, num_workers_t=None, shuffle_val=True):
    if 'libigpu5' in server_type:
        path = '/home/choi574/datasets/mscoco/data'
        path_imgIds = '/mnt/lls/home/choi574/git_libs/misc/salicon_MSCOCO'
        path_salicon_in = '/home/choi574/datasets/salicon_original/image/images/'
        path_salicon_target = '/home/choi574/datasets/salicon_original/maps/'
    elif 'libigpu' in server_type:
        path = '/home/choi574/datasets/mscoco/'
        path_imgIds = '/mnt/lls/home/choi574/git_libs/misc/salicon_MSCOCO'
        path_salicon_in = '/home/choi574/datasets/salicon_original/image/images/'
        path_salicon_target = '/home/choi574/datasets/salicon_original/maps/'
    elif 'libilab' in server_type:
        path = '/datasets/mscoco/'
        path_imgIds = '/home/choi574/git_libs/misc/salicon_MSCOCO'
        path_salicon_in = '/datasets/salicon_original/image/images/'
        path_salicon_target = '/datasets/salicon_original/maps/'
    elif server_type == 'greatlake':
        path = '/tmpssd/minkyu/mscoco/'
        path_imgIds = '/home/cminkyu/git_libs/misc/salicon_MSCOCO'
        path_salicon_in = '/tmpssd/minkyu/salicon_original/image/images/'
        path_salicon_target = '/tmpssd/minkyu/salicon_original/maps/'
    else:
        print("undefined server type")
    path = os.path.expanduser(path)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    
    
    '''
    2023.01.17. 
    For the consistency with the salicon loader, 
    I removed random resized crop and random horizontal flip are removed. 
    '''
    train_transforms = transforms.Compose([
        transforms.Resize(img_s_return),
        #transforms.RandomResizedCrop(img_s_return),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
        ]) 



    train_data = CocoDetection(os.path.join(path, 'train2014/'),
            os.path.join(path, 'annotations/instances_train2014.json'),
            path_salicon_in=os.path.join(path_salicon_in, 'train'),
            path_salicon_target=os.path.join(path_salicon_target, 'train'),
            path_imgIds=os.path.join(path_imgIds, 'salicon_image_ids_train.npy'),
            transform=train_transforms, 
            size_img=(img_s_return, img_s_return)
            )
    val_data =  CocoDetection(os.path.join(path, 'val2014/'),
            os.path.join(path, 'annotations/instances_val2014.json'),
            path_salicon_in=os.path.join(path_salicon_in, 'val'),
            path_salicon_target=os.path.join(path_salicon_target, 'val'),
            path_imgIds=os.path.join(path_imgIds, 'salicon_image_ids_val.npy'),
            size_img=(img_s_return, img_s_return),
            transform=transforms.Compose([
                transforms.Resize((img_s_return, img_s_return)),
                transforms.ToTensor(),
                normalize
            ]))

    if num_workers_t == None:
        num_workers_t = num_workers

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=shuffle_val,
        num_workers=num_workers_t, pin_memory=True, drop_last=True)
    return train_loader, val_loader, 80



