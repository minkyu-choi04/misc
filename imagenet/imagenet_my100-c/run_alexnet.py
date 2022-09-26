import torch
import torch.nn as nn
import numpy as np
import os
import sys
import torchvision.models as models

import imagenetC_getData

class ResnetData(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.alexnet(pretrained=True)
        #self.model = models.alexnet(weights='IMAGENET1K_V1')
        
        
    def forward(self, img):
        return_dict = {}

        pred = self.model(img)

        return_dict['pred'] = [pred]

        return return_dict

model = ResnetData()
imagenetC_getData.test_ImageNetC(model, 1, batch_size=64, img_size=(256,227), server_type='libigpus_new')

