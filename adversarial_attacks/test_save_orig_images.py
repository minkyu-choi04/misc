import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import numpy as np
import time
import torch.nn.functional as F
from torch.nn import init
import os
import argparse

import sys
sys.path.append('/mnt/lls/local_export/3/home/choi574/git_libs/misc/')
sys.path.append('/home/choi574/research_mk/git_libs_onlyForLibi2/misc/')
import misc
sys.path.append('/mnt/lls/local_export/3/home/choi574/git_libs/misc/imagenet/imagenet_my100/')
sys.path.append('/home/choi574/research_mk/git_libs_onlyForLibi2/misc/imagenet/imagenet_my100/')
import dataloader_imagenet100my as dl
sys.path.append('/mnt/lls/local_export/3/home/choi574/git_libs/misc/adversarial_attacks/')
sys.path.append('/home/choi574/research_mk/git_libs_onlyForLibi2/misc/adversarial_attacks/')
import utils 

import logging

from advertorch.attacks import LinfSPSAAttack, LinfPGDAttack
from advertorch.utils import NormalizeByChannelMeanStd
from advertorch_examples.utils import get_panda_image, bchw2bhwc, bhwc2bchw




def attack_LinfPGD(args, model_o, eps_list, img_s_load=256+128, img_s_return=224+112, cudnn_backends=True, imgsavedir='/home/libi/HDD1/minkyu/collect_AA_imgs'):
    #attack = LinfPGDAttack(model, eps=2.0 / 255, nb_iter=20, eps_iter=0.2 / 255)
    #attack = LinfPGDAttack(
    #    model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=2.0/255,
    #    nb_iter=20, eps_iter=0.2/255, rand_init=True, clip_min=-2.5, clip_max=2.5,
    #    targeted=False)
    torch.backends.cudnn.enabled = cudnn_backends

    normalize = NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model = torch.nn.Sequential(normalize, model_o).cuda()
    model_o.eval()
    model.eval()

    val_loader, lu_c2i, lu_c = dl.load_imagenet_myclass100_for_AdvAttacks(img_s_load=img_s_load, img_s_return=img_s_return, server_type=args.server_type, shuffle_test=False)
    print('\n\n=============================')
    print(' Image will be loaded by ')
    print('      {}  /  {} '.format(img_s_load, img_s_return))
    print('=============================\n\n')
    print('\n\n=============================')
    print(' attack_nsteps ')
    print('=============================\n\n')

    if not os.path.exists('plots'):
        os.makedirs('plots')


    fr = []
    acc_list = []

    #for eps in [0.001, 0.002, 0.003, 0.005, 0.007, 0.009, 0.01, 0.03, 0.05]:
    #for eps in [ 0.005, 0.01, 0.03, 0.05]:
    #for eps in [ 0.07, 0.09, 0.1, 0.3, 0.5]:
    for eps in eps_list:
    #for eps in [0.001, 0.002, 0.003, 0.005, 0.01]:
        #eps = 0.001
        print('eps: ', eps)
        success = 0
        failure = 0
        incorrect = 0
        total_n = 0

        cnt = np.zeros(100)


        time_s = time.perf_counter()

        model.eval()
        for batch_i, data in enumerate(val_loader):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda().long()
            loss_b = 0.0

            total_n += 1

            ''' Test Original Image on the model '''

            gt_labels = labels.cpu().numpy()[0]
            utils.save_AA_imgs_range01_lookup(imgsavedir, eps, gt_labels, cnt, inputs, lu_c[gt_labels])
            cnt[gt_labels] = cnt[gt_labels] + 1
            #print('instance')

        
    
    
    '''fd = open('print_adv_eps_failureRate', 'a')
    fd.write('=============\n')
    for i in range(len(fr)):
        fd.write('{:.4f},  '.format(fr[i]))
    fd.write('\n')
    fd.close()

    fd = open('print_adv_eps_failureRate8', 'a')
    fd.write('=============\n')
    for i in range(len(fr8)):
        fd.write('{:.4f},  '.format(fr8[i]))
    fd.write('\n')
    fd.close()'''




