'''
2020.05.21.
https://raw.githubusercontent.com/imatge-upc/saliency-2019-SalBCE/master/src/dataloader/datasetSALICON.py
'''


import os
import sys
import cv2

import numpy as np
#from IPython import embed
from PIL import Image
from random import randint
from scipy import ndimage

import matplotlib.pylab as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

#size = (192, 256)
# BGR MODE

def imageProcessing(image, saliency, size_img, size_sal):
    image = cv2.resize(image, (size_img[1], size_img[0]), interpolation=cv2.INTER_AREA).astype(np.float32)
    saliency = cv2.resize(saliency, (size_sal[1], size_sal[0]), interpolation=cv2.INTER_AREA).astype(np.float32)

    # remove mean value
    #image -= mean
    augmentation = randint(0,2)
    '''if augmentation == 0: #horizental flip
        image = image[:,::-1,:]
        saliency = saliency[:,::-1]
    elif augmentation == 1:# vertical flip
        image = image[::-1,:,:]
        saliency = saliency[::-1,:]
    elif augmentation == 2: # rotation
        image = ndimage.rotate(image, 45)
        saliency = ndimage.rotate(saliency, 45)
        sqr = image.shape[0]
        start1 = int((sqr-192)/2)+1
        end1 = sqr-int((sqr-192)/2)
        start2 = int((sqr-256)/2)+1
        end2 = sqr-int((sqr-256)/2)
        image = image[start1:end1, start2:end2,:]
        saliency = saliency[start1:end1, start2:end2]'''
    # convert to torch Tensor
    image = np.ascontiguousarray(image)/255.
    saliency = np.ascontiguousarray(saliency)/255.

    transform_img = transforms.Compose([transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_sal = transforms.ToTensor()
    #image = torch.FloatTensor(image)

    # swap channel dimensions
    #image = image.transpose(2,0,1)
    #image = image -0.5
    image = transform_img(image)
    #image = torch.cat((image[1, :, :].unsqueeze(0), image[0, :, :].unsqueeze(0), image[2, :, :].unsqueeze(0)), 0)
    #image = torch.cat((image[0, :, :].unsqueeze(0), image[2, :, :].unsqueeze(0), image[1, :, :].unsqueeze(0)), 0)
    #image = torch.cat((image[1, :, :].unsqueeze(0), image[2, :, :].unsqueeze(0), image[0, :, :].unsqueeze(0)), 0)
    #image = torch.cat((image[2, :, :].unsqueeze(0), image[0, :, :].unsqueeze(0), image[1, :, :].unsqueeze(0)), 0)
    image = torch.cat((image[2, :, :].unsqueeze(0), image[1, :, :].unsqueeze(0), image[0, :, :].unsqueeze(0)), 0)
    saliency = transform_sal(saliency)
    return image,saliency

class SALICON(Dataset):
    def __init__(self, path_dataset, size_img=(480, 640), size_sal=(480, 640), mode='train', N=None):
        self.path_dataset = path_dataset#os.path.expanduser('~/datasets/salicon_original')
        self.size_img = size_img
        self.size_sal = size_sal
        self.path_images = os.path.join(self.path_dataset,'image', 'images', mode)
        self.path_saliency = os.path.join(self.path_dataset, 'maps', mode)

        # get list images
        list_names = os.listdir( os.path.join(self.path_dataset, 'image', 'images', mode) )
        list_names = np.array([n.split('.')[0] for n in list_names])
        self.list_names = list_names

        if N is not None:
            self.list_names = list_names[:N]
        # embed()
        print("Init dataset in mode {}".format(mode))
        print("\t total of {} images.".format(self.list_names.shape[0]))

    def __len__(self):
        return self.list_names.shape[0]

    def __getitem__(self, index):

        # Image and saliency map paths
        rgb_ima = os.path.join(self.path_images, self.list_names[index]+'.jpg')
        sal_path = os.path.join(self.path_saliency, self.list_names[index]+'.png')

        image = cv2.imread(rgb_ima)
        saliency = cv2.imread(sal_path, 0)
        return imageProcessing(image, saliency, self.size_img, self.size_sal)

if __name__ == '__main__':
	s = SALICON(mode='val', N=100)

	image, saliency = s[0]
