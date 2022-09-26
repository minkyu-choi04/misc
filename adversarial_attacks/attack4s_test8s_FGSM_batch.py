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
sys.path.append('/home/choi574/git_libs/misc/')
#sys.path.append('/home/choi574/research_mk/git_libs_onlyForLibi2/misc/')
import misc
sys.path.append('/mnt/lls/local_export/3/home/choi574/git_libs/misc/imagenet/imagenet_my100/')
sys.path.append('/home/choi574/git_libs/imagenet/imagenet_my100/')
#sys.path.append('/home/choi574/research_mk/git_libs_onlyForLibi2/misc/imagenet/imagenet_my100/')
import dataloader_imagenet100my as dl
sys.path.append('/mnt/lls/local_export/3/home/choi574/git_libs/misc/adversarial_attacks/')
sys.path.append('/home/choi574/git_libs/misc/adversarial_attacks/')
#sys.path.append('/home/choi574/research_mk/git_libs_onlyForLibi2/misc/adversarial_attacks/')
import utils 

import logging

import advertorch
from advertorch.attacks import LinfSPSAAttack, LinfPGDAttack
from advertorch.utils import NormalizeByChannelMeanStd
from advertorch_examples.utils import get_panda_image, bchw2bhwc, bhwc2bchw


def attack_LinfPGD(args, model_o, eps_list, n_steps=16, batch_size=128, img_s_load=256+128, img_s_return=224+112, cudnn_backends=True, n_iter=5, eps_iter=3, es=None):
    '''
    To run this function,
    - eps_list: must inlcude 0.0 at the beginning of the eps_list
    - model: must be able to return all steps' predictions as a dictionary form when a flag is given to the forward(). 

    Change train.py
    1. add isReturnAllStep=False to the forward()
    2. at the end of the forward(), add 
        if isReturnAllStep:
            return return_dict
        else:
            return pred
    3. Add 0.0 at the beginning of the eps_list
        eps_list = [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.03, 0.05, 0.07]
    4. add
        sys.path.append('/mnt/lls/local_export/3/home/choi574/git_libs/misc/adversarial_attacks/')
        import attack4s_test8s_batch as attack
        attack.attack_LinfPGD(args, model, eps_list, n_steps=16, batch_size=128, img_s_load=256+128, img_s_return=224+112)
        return

    example code:
        min@libigpu3:~/research_mk/polar/pretrain/cart_warping2/align_comp_params/train_imagenet100_myclasses/foveatedImage_myClass/adversarial_attack/eps_range/furtuer_tests/two_stream_stochasticFixationsTest/corrected_downsample_model/corrected_eps_img_range/single_stream_correct_trained_rndInitFixs_modes/non_stochastic_fixs/convrnn_cw75_b10p0_miniGRUwState_FOVEATIONsimple_vgg13v4_4blksSiz2_2222_imgn100MY_VDwarpBilinDownsclNearest_rndInitFixs_lrAdje80e110_fixedInitFixs_attack1s_testMultiAccs
    '''

    if 0.0 not in eps_list:
        eps_list.append(0.0)

    torch.backends.cudnn.enabled = cudnn_backends


    normalize = NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model = torch.nn.Sequential(normalize, model_o).cuda()
    model_o.eval()
    model.eval()

    val_loader = dl.load_imagenet_myclass100_for_AdvAttacks(batch_size=batch_size, img_s_load=img_s_load, img_s_return=img_s_return, server_type=args.server_type)
    print('\n\n=============================')
    print(' Image will be loaded by ')
    print('      {}  /  {} '.format(img_s_load, img_s_return))
    print('=============================\n\n')
    print('\n\n=============================')
    print(' attack4s_test8s ')
    print('=============================\n\n')

    if not os.path.exists('plots'):
        os.makedirs('plots')

    fr = []
    acc_list = []
    fr8 = []
    acc_list8 = []

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
        attack = advertorch.attacks.GradientSignAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps,
            clip_min=0.0, clip_max=1.0,
            targeted=False)

        time_s = time.perf_counter()

        model.eval()
        for batch_i, data in enumerate(val_loader):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda().long()
            loss_b = 0.0


            ''' Test Original Image on the model '''
            #pred = model(inputs)

            tic = time.time()
            if eps != 0.0:              
                x_adv = attack.perturb(inputs, labels)
            else:
                x_adv = inputs
            toc = time.time()
            #print("elapsed time: {} sec".format(toc - tic))
            #logging.info("elapsed time: {} sec".format(toc - tic))

            ###################### 4 steps ####################
            #prediction_on_purturbed_img = model(x_adv)
            #logging.info("argmax(y^) = {}".format(torch.max(prediction_on_purturbed_img, 1)))
            #logging.info("prediction of purturbed image: {}".format(prediction_on_purturbed_img))
            #print("argmax(y^) = {}".format(torch.max(prediction_on_purturbed_img, 1)))
            ####print("prediction of purturbed image: {}".format(prediction_on_purturbed_img))

            ##################### 8 steps #######################
            with torch.no_grad():
                x_adv_norm = normalize(x_adv)
                return_dict = model_o(x_adv_norm, n_steps=n_steps, isReturnAllStep=True)
            #logging.info("argmax(y^) = {}".format(torch.max(prediction_on_purturbed_img, 1)))
            #logging.info("prediction of purturbed image: {}".format(prediction_on_purturbed_img))
            #print("argmax(y^) = {}".format(torch.max(prediction_on_purturbed_img, 1)))
            ####print("prediction of purturbed image: {}".format(prediction_on_purturbed_img))

            for step in range(n_steps):
                acc1, acc5 = misc.accuracy(torch.squeeze(return_dict['pred'][step]), torch.squeeze(labels), topk=(1, 5))
                acc_top1[step].update(acc1[0], batch_size)
                acc_top5[step].update(acc5[0], batch_size)

            #print(batch_i)
            if batch_i == 10:
                n_plots = 1 # min(56, int(args.batch_size/int(torch.cuda.device_count())))
                misc.plot_one_sample_from_images(normalize(inputs[:n_plots]),
                        'plots', 'test_e'+str(0)+'b'+str(batch_i)+str(int(eps*1000))+'_in.jpg')
                misc.plot_one_sample_from_images(normalize(x_adv[:n_plots]), 
                        'plots', 'test_e'+str(0)+'b'+str(batch_i)+str(int(eps*1000))+'_in_adv.jpg')
            if False:#es != None:
                if batch_i>es:
                    break
                    #print('test data saved')
        
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


