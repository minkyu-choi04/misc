import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import numpy as np
import time
import torch.nn.functional as F
from torch.nn import init
import scipy.io
import os
import argparse

import sys
sys.path.append('/home/cminkyu/git_libs/misc/')
sys.path.append('/mnt/lls/local_export/3/home/choi574/git_libs/misc/')
import misc

def load_imagenetC_myclass100(batch_size,
                                curr_type, curr_type_sub, lev, 
                                img_s_load=256, img_s_return=224, 
                                server_type='libigpu5', 
                                num_workers=1, 
                                shuffle_test=False, isReturnInfo=False):
    if server_type == 'libigpus_new':
        path = '/home/choi574/datasets/ImageNetC_myclass100/'
    elif server_type == 'greatlake-hdd':
        path = '/tmp/minkyu/ImageNetC_myclass100/'
    else:
        print("undefined server type")
    print('================ MY CLASS 100 IMGNET =================== ')

    path = os.path.join(path, curr_type, curr_type_sub, lev)
    test_data =  datasets.ImageFolder(root=os.path.expanduser(path),
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



def test_ImageNetC(model, n_steps, batch_size=64, img_size=(256,224), server_type='libigpus_new'):
    currs = {}
    currs['blur'] = ['defocus_blur',  'glass_blur',  'motion_blur',  'zoom_blur']
    currs['digital'] = ['contrast',  'elastic_transform',  'jpeg_compression',  'pixelate']
    currs['extra'] = ['gaussian_blur',  'saturate',  'spatter',  'speckle_noise']
    currs['noise'] = ['gaussian_noise',  'impulse_noise',  'shot_noise']
    currs['weather'] = ['brightness',  'fog',  'frost',  'snow']

    records_t1 = {}
    records_t5 = {}

    for curr_type in currs.keys():
        for curr_type_sub in currs[curr_type]:
            for lev in range(5):
                time_s = time.perf_counter()

                test_loader = load_imagenetC_myclass100(batch_size,
                        curr_type, curr_type_sub, str(lev+1), 
                        img_s_load=img_size[0], img_s_return=img_size[1],
                        server_type=server_type, num_workers=2)
            
                acc_top1 = []
                acc_top5 = []
                for step in range(n_steps):
                    acc_top1.append(misc.AverageMeter('Acc@1', ':6.2f'))
                    acc_top5.append(misc.AverageMeter('Acc@5', ':6.2f'))

                model.eval()
                with torch.no_grad():
                    for batch_i, data in enumerate(test_loader):
                        inputs, labels = data
                        inputs, labels = inputs.cuda(), labels.cuda().long()
                        
                        return_dict = model(inputs)

                        for step in range(n_steps):        
                            acc1, acc5 = misc.accuracy(torch.squeeze(return_dict['pred'][step]), torch.squeeze(labels), topk=(1,5))
                            acc_top1[step].update(acc1[0], len(inputs))
                            acc_top5[step].update(acc5[0], len(inputs))
                            
                    k = [curr_type, curr_type_sub, str(lev+1)].join('/')
                    records_t1[k] = []
                    for step in range(n_steps):
                        records_t1[k].append(acc_top1[step].avg)
                        records_t5[k].append(acc_top5[step].avg)

                    time_e = time.perf_counter() - time_s
                    print(curr_type, curr_type_sub, lev, time_e)
    
    np.save('records_t1.npy', records_t1)
    np.save('records_t5.npy', records_t5)
    return records_t1, records_t5
    