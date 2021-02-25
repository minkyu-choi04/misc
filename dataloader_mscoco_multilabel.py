import os
import torch
import torch.utils.data as data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image

import sys
#sys.path.append('/home/choi574/cocoapi/PythonAPI')
sys.path.append('/mnt/lls/local_export/3/home/choi574/cocoapi/PythonAPI')
from pycocotools.coco import COCO

class CocoDetection(datasets.coco.CocoDetection):
    ''' 2021.02.06
    https://github.com/allenai/elastic/blob/master/multilabel_classify.py
    '''
    def __init__(self, root, annFile, transform=None, target_transform=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)
        # print(self.cat2cat)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
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
        return img, target


def load_mscoco_multilabel_ddp(batch_size, img_s_load=256, img_s_return=224, server_type='libigpu5', isRandomResize=True, num_workers=4):
    if server_type == 'libigpu0':
        path = '/home/min/DATASET/mscoco/'
    elif server_type == 'libigpu1':
        path = '/home/min/datasets/mscoco/'
    elif server_type == 'home':
        path = '~/datasets/ImageNet2012/'
    elif server_type == 'libigpu2':
        path = '/home/choi574/datasets/mscoco/'
    elif server_type == 'libigpu3':
        path = '/home/min/datasets/mscoco/'
    elif server_type == 'libigpu4':
        path = '/home/libiadm/datasets/ImageNet2012/'
    elif server_type == 'libigpu5':
        path = '/home/choi574/datasets/mscoco/'
    elif server_type == 'libigpu6':
        path = '/home/choi574/datasets/mscoco/'
    elif server_type == 'libigpu7':
        path = '/home/libiadm/datasets/ImageNet2012/'
    else:
        print("undefined server type")
    path = os.path.expanduser(path)

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


    train_data = CocoDetection(os.path.join(path, 'train2014/'),
            os.path.join(path, 'annotations/instances_train2014.json'),
            transform=train_transforms)
    val_data =  CocoDetection(os.path.join(path, 'val2014/'),
            os.path.join(path, 'annotations/instances_val2014.json'),
            transform=transforms.Compose([
                transforms.Resize((img_s_return, img_s_return)),
                transforms.ToTensor(),
                normalize
            ]))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    return train_loader, val_loader, 80, train_sampler

def load_mscoco_multilabel(batch_size, img_s_load=256, img_s_return=224, server_type='libigpu5', isRandomResize=True, num_workers=4):
    if server_type == 'libigpu0':
        path = '/home/libi/datasets/ImageNet2012/'
    elif server_type == 'libigpu1':
        path = '/home/min/datasets/mscoco/'
    elif server_type == 'home':
        path = '~/datasets/ImageNet2012/'
    elif server_type == 'libigpu2':
        path = '/home/choi574/datasets/mscoco/'
    elif server_type == 'libigpu3':
        path = '/home/min/datasets/mscoco/'
    elif server_type == 'libigpu4':
        path = '/home/libiadm/datasets/ImageNet2012/'
    elif server_type == 'libigpu5':
        path = '/home/choi574/datasets/mscoco/data/'
    elif server_type == 'libigpu6':
        path = '/home/choi574/datasets/mscoco/'
    elif server_type == 'libigpu7':
        path = '/home/libiadm/datasets/ImageNet2012/'
    else:
        print("undefined server type")
    path = os.path.expanduser(path)

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


    train_data = CocoDetection(os.path.join(path, 'train2014/'),
            os.path.join(path, 'annotations/instances_train2014.json'),
            transform=train_transforms)
    val_data =  CocoDetection(os.path.join(path, 'val2014/'),
            os.path.join(path, 'annotations/instances_val2014.json'),
            transform=transforms.Compose([
                transforms.Resize((img_s_return, img_s_return)),
                transforms.ToTensor(),
                normalize
            ]))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    return train_loader, val_loader, 80



