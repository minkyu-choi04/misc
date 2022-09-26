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

#from advertorch.attacks import LinfSPSAAttack, LinfPGDAttack
#from advertorch.utils import NormalizeByChannelMeanStd
#from advertorch_examples.utils import get_panda_image, bchw2bhwc, bhwc2bchw
import foolbox
from foolbox.attacks import AdamRandomStartProjectedGradientDescentAttack
import eagerpy as ep
'''
Foolbox version Attack script
https://github.com/bethgelab/foolbox/blob/master/examples/single_attack_pytorch_resnet18.py
https://foolbox.readthedocs.io/en/v2.3.0/user/examples.html
https://foolbox.readthedocs.io/en/v3.3.1/modules/attacks.html?highlight=LinfPGD#
'''

def attack_LinfPGDAdam_Foolbox(args, model_o, eps_list, batch_size=32, n_iter=40, eps_iter=5, img_s_load=256+128, img_s_return=224+112, cudnn_backends=True, es=None, n_steps=16):
    #attack = LinfPGDAttack(model, eps=2.0 / 255, nb_iter=20, eps_iter=0.2 / 255)
    #attack = LinfPGDAttack(
    #    model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=2.0/255,
    #    nb_iter=20, eps_iter=0.2/255, rand_init=True, clip_min=-2.5, clip_max=2.5,
    #    targeted=False)
    torch.backends.cudnn.enabled = cudnn_backends
    model_o.eval()

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    preprocessing = dict(mean=mean, std=std, axis=-3)
    model = foolbox.models.PyTorchModel(model_o, bounds=(0, 1), num_classes=100, preprocessing=preprocessing)
    #normalize = transforms.Normalize(mean, std)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    mean_t = torch.tensor(mean).cuda().view(1, -1, 1, 1)
    std_t = torch.tensor(std).cuda().view(1, -1, 1, 1)

    val_loader = dl.load_imagenet_myclass100_for_AdvAttacks(img_s_load=img_s_load, img_s_return=img_s_return, server_type=args.server_type, batch_size=batch_size)
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
    fr8 = []
    acc_list8 = []
    attack = AdamRandomStartProjectedGradientDescentAttack(model, distance=foolbox.distances.Linf)
    #for eps in [0.001, 0.002, 0.003, 0.005, 0.007, 0.009, 0.01, 0.03, 0.05]:
    #for eps in [ 0.005, 0.01, 0.03, 0.05]:
    #for eps in [ 0.07, 0.09, 0.1, 0.3, 0.5]:
    for eps in eps_list:
    #for eps in [0.001, 0.002, 0.003, 0.005, 0.01]:
        #eps = 0.001
        acc_top1 = []
        acc_top5 = []
        for step in range(n_steps):
            acc_top1.append(misc.AverageMeter('Acc@1', ':6.2f'))
            acc_top5.append(misc.AverageMeter('Acc@5', ':6.2f'))

        print('eps: ', eps)
        #attack = LinfPGDAttack(
        #    model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps,
        #    nb_iter=5, eps_iter=eps/3, rand_init=True, clip_min=0.0, clip_max=1.0,
        #    targeted=False)

        time_s = time.perf_counter()

        #model.eval()
        for batch_i, data in enumerate(val_loader):
            inputs, labels = data
            #inputs, labels = inputs.cuda(), labels.cuda().long()
            loss_b = 0.0


            ''' Test Original Image on the model '''
            
            
            tic = time.time()
            if eps != 0.0:
                advs = attack(inputs.numpy(), labels.numpy(), binary_search=False, epsilon=eps, iterations=n_iter, stepsize=eps/eps_iter, unpack=False)

                advs = [a.perturbed for a in advs]
                advs = [
                    p if p is not None else u
                    for p, u in zip(advs, inputs)
                ]
                advs = np.stack(advs)
            else:
                inputs, labels = inputs.cuda(), labels.cuda().long()
                advs = inputs
            toc = time.time()


            ''' long steps '''
            with torch.no_grad():
                advs_n = (torch.tensor(advs).cuda() - mean_t)/std_t
                return_dict = model_o(advs_n, n_steps=n_steps, isReturnAllStep=True)

            for step in range(n_steps):
                acc1, acc5 = misc.accuracy(torch.squeeze(return_dict['pred'][step]), torch.squeeze(torch.tensor(labels).cuda()), topk=(1, 5))
                acc_top1[step].update(acc1[0], batch_size)
                acc_top5[step].update(acc5[0], batch_size)

            if es != None:
                if batch_i>es:
                    break


        print('{} : \t'.format(eps), end='')
        for s in range(n_steps):
            print('{}, '.format(acc_top1[s].avg), end='')
        print('\n')

        fd = open('print_adv_acc', 'a')
        fd.write('{} : \t'.format(eps))#, end='')
        for s in range(n_steps):
            fd.write('{}, '.format(acc_top1[s].avg))#, end='')
        fd.write('\n')
        fd.close()


