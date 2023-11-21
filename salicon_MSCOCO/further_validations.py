import os
import torch
import torch.utils.data as data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image

import sys
sys.path.append('/home/choi574/coco/PythonAPI')
#sys.path.append('/mnt/lls/local_export/3/home/choi574/cocoapi/PythonAPI')
from pycocotools.coco import COCO

import numpy as np

'''
2023.03.20. Minkyu
I here validate the data. 
Check if
    1. the image IDs from salicon train / validation sets are from train / validation sets from MSCOCO
        (if there is any interchanges between train and validation sets)
'''


path = '/home/choi574/datasets/mscoco/data/'
root = os.path.join(path, 'train2014/')
annFile_train = os.path.join(path, 'annotations/instances_train2014.json')
annFile_val = os.path.join(path, 'annotations/instances_val2014.json')

coco_train = COCO(annFile_train)
coco_ids_train = list(coco_train.imgs.keys())

coco_val = COCO(annFile_val)
coco_ids_val = list(coco_val.imgs.keys())




path_imgIds = '/mnt/lls/home/choi574/git_libs/misc/salicon_MSCOCO'
path_imgIds_train = os.path.join(path_imgIds, 'salicon_image_ids_train.npy')
sal_ids_train = np.load(path_imgIds_train)

path_imgIds_val = os.path.join(path_imgIds, 'salicon_image_ids_val.npy')
sal_ids_val = np.load(path_imgIds_val)


salTrain_cocoTrain = 0
salVal_cocoTrain = 0
salTrain_cocoVal = 0
salVal_cocoVal = 0

for sal_id_train in sal_ids_train:
    sal_id_train = int(sal_id_train)
    if sal_id_train in coco_ids_train:
        salTrain_cocoTrain += 1
    elif sal_id_train in coco_ids_val:
        salTrain_cocoVal += 1
    else:
        print('something is wrong: ', sal_id_train)
        
        
for sal_id_val in sal_ids_val:
    sal_id_val = int(sal_id_val)
    if sal_id_val in coco_ids_train:
        salVal_cocoTrain += 1
    elif sal_id_val in coco_ids_val:
        salVal_cocoVal += 1
    else:
        print('something is wrong: ', sal_id_val)
        
print(len(sal_ids_train))
print(len(sal_ids_val))
print(salTrain_cocoTrain, salVal_cocoTrain, salTrain_cocoVal, salVal_cocoVal)
'''
Results
    10000
    5000
    10000 0 0 5000
There are no mixed image ids, which means the images from the salicon training are all from training set of MSCOCO. 
And the validation images of salicon are all from validation set of MSCOCO. 
'''

val_samples = [554828, 528411, 425324, 192714]
for val_sample in val_samples:
    print(val_sample in coco_ids_val)
    
train_samples = [189223, 427958, 254775, 299029]
for train_sample in train_samples:
    print(train_sample in coco_ids_train)
    
    