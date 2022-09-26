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

from advertorch.attacks import LinfSPSAAttack, LinfPGDAttack
from advertorch.utils import NormalizeByChannelMeanStd
from advertorch_examples.utils import get_panda_image, bchw2bhwc, bhwc2bchw


def attack_LinfPGD(args, model_o, eps_list, img_s_load=256+128, img_s_return=224+112, cudnn_backends=True):
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

    val_loader = dl.load_imagenet_myclass100_for_AdvAttacks(img_s_load=img_s_load, img_s_return=img_s_return, server_type=args.server_type)
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
        print('eps: ', eps)
        attack = LinfPGDAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps,
            nb_iter=5, eps_iter=eps/3, rand_init=True, clip_min=0.0, clip_max=1.0,
            targeted=False)
        success = 0
        failure = 0
        incorrect = 0
        success8 = 0
        failure8 = 0
        incorrect8 = 0
        total_n = 0

        time_s = time.perf_counter()

        model.eval()
        for batch_i, data in enumerate(val_loader):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda().long()
            loss_b = 0.0

            total_n += 1

            ''' Test Original Image on the model '''
            pred = model(inputs)
            if torch.max(pred, 1)[1].cpu().numpy()[0] == labels.cpu().numpy()[0]:
                #print("original prediction matches the label.")
                logging.info("original prediction matches the label.")
            else:
                #print("original prediction does not match the label.")
                logging.info("original prediction does not match the label.")
                incorrect += 1
                continue

            tic = time.time()
            x_adv = attack.perturb(inputs, labels)
            toc = time.time()
            #print("elapsed time: {} sec".format(toc - tic))
            #logging.info("elapsed time: {} sec".format(toc - tic))

            ###################### 4 steps ####################
            prediction_on_purturbed_img = model(x_adv)
            #logging.info("argmax(y^) = {}".format(torch.max(prediction_on_purturbed_img, 1)))
            #logging.info("prediction of purturbed image: {}".format(prediction_on_purturbed_img))
            #print("argmax(y^) = {}".format(torch.max(prediction_on_purturbed_img, 1)))
            ####print("prediction of purturbed image: {}".format(prediction_on_purturbed_img))

            if (torch.max(prediction_on_purturbed_img, 1)[1].cpu().numpy()[0] == labels.cpu().numpy()[0]):
                failure += 1
                logging.info("attack failed.")
                #print("attack failed.")
            else:
                success += 1
                logging.info("attack succeeded.")
                logging.info( "success : failure : incorrect = {} : {} : {}".format(success, failure, incorrect))
                #print("attack succeeded.")
                #print( "success : failure : incorrect = {} : {} : {}".format(success, failure, incorrect))

            ##################### 8 steps #######################
            x_adv_norm = normalize(x_adv)
            prediction_on_purturbed_img = model_o(x_adv_norm, n_steps=8)
            #logging.info("argmax(y^) = {}".format(torch.max(prediction_on_purturbed_img, 1)))
            #logging.info("prediction of purturbed image: {}".format(prediction_on_purturbed_img))
            #print("argmax(y^) = {}".format(torch.max(prediction_on_purturbed_img, 1)))
            ####print("prediction of purturbed image: {}".format(prediction_on_purturbed_img))

            if (torch.max(prediction_on_purturbed_img, 1)[1].cpu().numpy()[0] == labels.cpu().numpy()[0]):
                failure8 += 1
                logging.info("attack failed.")
                #print("attack failed.")
            else:
                success8 += 1
                logging.info("attack succeeded.")
                logging.info( "success : failure : incorrect = {} : {} : {}".format(success8, failure8, incorrect))
                #print("attack succeeded.")
                #print( "success : failure : incorrect = {} : {} : {}".format(success, failure, incorrect))

            print(batch_i)
            if batch_i > 10:
                n_plots = 1 # min(56, int(args.batch_size/int(torch.cuda.device_count())))
                misc.plot_one_sample_from_images(normalize(inputs[:n_plots]),
                        'plots', 'test_e'+str(0)+'b'+str(batch_i)+str(int(eps*1000))+'_in')
                misc.plot_one_sample_from_images(normalize(x_adv[:n_plots]), 
                        'plots', 'test_e'+str(0)+'b'+str(batch_i)+str(int(eps*1000))+'_in_adv')
                print('check')
                break
            #if batch_i>10:
            #    break
            #    #print('test data saved')
        
        ss = success / (failure+success)
        ff = failure / (failure+success)
        accr = 1 - (success+incorrect) / total_n
        accr1 = failure / total_n
        print( "success : failure : incorrect = {} : {} : {}  ||  {}  {}  {}".format(success, failure, incorrect, ss, ff, accr1))
        fd = open('print_adv_eps', 'a')
        fd.write('eps: {}  |  s : f: i =  {}  :  {}  :  {}   ||  {}  {}  {}\n'.format(eps, success, failure, incorrect, ss, ff, accr1))
        fd.close()
        #if accr != accr1:
        #    print('wrong calculation in accs: ', accr, accr1)

        
        ss8 = success8 / (failure8+success8)
        ff8 = failure8 / (failure8+success8)
        accr8 = failure8 / total_n
        print( "success : failure : incorrect = {} : {} : {}  ||  {}  {}  {}".format(success8, failure8, incorrect, ss8, ff8, accr8))
        fd = open('print_adv_eps8', 'a')
        fd.write('eps: {}  |  s : f: i =  {}  :  {}  :  {}   ||  {}  {}  {}\n'.format(eps, success8, failure8, incorrect, ss8, ff8, accr8))
        fd.close()

        fr.append(ff)
        acc_list.append(accr)
        fr8.append(ff8)
        acc_list8.append(accr8)
    

    utils.write_scores_in_a_row('print_adv_eps_failureRate', fr)
    utils.write_scores_in_a_row('print_adv_eps_failureRate8', fr8)
    utils.write_scores_in_a_row('print_adv_eps_acc', acc_list)
    utils.write_scores_in_a_row('print_adv_eps_acc8', acc_list8)
    
    
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




