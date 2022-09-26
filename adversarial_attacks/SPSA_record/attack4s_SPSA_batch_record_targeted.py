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
sys.path.append('/mnt/lls/local_export/3/home/choi574/git_libs/misc/imagenet/imagenet_my100/imgnet_my100_val100/')
sys.path.append('/home/cminkyu/git_libs/misc/imagenet/imagenet_my100/imgnet_my100_val100/')
#sys.path.append('/home/choi574/research_mk/git_libs_onlyForLibi2/misc/imagenet/imagenet_my100/')
import dataloader_imagenet100my_val100 as dl
sys.path.append('/mnt/lls/local_export/3/home/choi574/git_libs/misc/adversarial_attacks/')
sys.path.append('/home/cminkyu/git_libs/misc/adversarial_attacks/')
#sys.path.append('/home/choi574/research_mk/git_libs_onlyForLibi2/misc/adversarial_attacks/')
import utils 

import logging

from advertorch.attacks import LinfSPSAAttack_allSteps, LinfPGDAttack_EOT
from advertorch.utils import NormalizeByChannelMeanStd
from advertorch_examples.utils import get_panda_image, bchw2bhwc, bhwc2bchw


def attack_LinfSPSAAttack_allSteps_targeted(args, model_o, eps_list, n_sample=8192, n_steps=16, batch_size=32, img_s_load=256+128, img_s_return=224+112, cudnn_backends=True, 
        n_iter=100, delta=0.01, max_batch_size=None, es=None, compensation=0):
    '''
    SPSA 
    Looks like n_iter=100, n_sample=8192 are the widely used settings. 
    See Saccader paper. 
    
    '''
    logging.basicConfig(filename='logs.log', filemode='w', level=logging.DEBUG, format='%(asctime)s %(message)s')
    print('n_sample: {}'.format(n_sample))
    print('n_iter: {}'.format(n_iter))
    print('Current attack script requires model to return [return_dict]')
    print('Use [Val100] dataset')
    print('use AvgStep: [{}]'.format(True))

    if 0.0 not in eps_list:
        eps_list.append(0.0)

    torch.backends.cudnn.enabled = cudnn_backends


    normalize = NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model = torch.nn.Sequential(normalize, model_o).cuda()
    model_o.eval()
    model.eval()

    #val_loader = dl.load_imagenet_myclass100_val1000_for_AdvAttacks(batch_size=batch_size, img_s_load=img_s_load, img_s_return=img_s_return, server_type=args.server_type)
    val_loader = dl.load_imagenet_myclass100_val100_for_AdvAttacks(batch_size=batch_size, img_s_load=img_s_load, img_s_return=img_s_return, server_type=args.server_type)
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

    for eps in eps_list:
        acc_top1 = []
        acc_top5 = []
        for step in range(n_steps):
            acc_top1.append(misc.AverageMeter('Acc@1', ':6.2f'))
            acc_top5.append(misc.AverageMeter('Acc@5', ':6.2f'))

        print('eps: ', eps)
        if max_batch_size == None:
            max_batch_size = batch_size
        attack = LinfSPSAAttack_allSteps(model, eps=eps, delta=delta, lr=0.01, nb_iter=n_iter, nb_sample=n_sample, max_batch_size=max_batch_size, targeted=True)
        '''
        max_batch_size is a little bit confusing. It is not same as batch_size that I defined here for data loader. 
        batch_size is what I used for dataloader to form one mini-batch. This is what I know. 
        max_batch_size is for iterating nb_sample. 
        A mini-batch of input images, x is in size (b, c, h, w). Inside SPSA, x will be resized to (max_batch_size, b, c, h, w). 
        This max_batch_size is for iterating nb_sample and it is nothing to do with the actual batch_size. 
        
        If you don't understand it, see the code
            https://github.com/BorealisAI/advertorch/blob/master/advertorch/attacks/spsa.py
        Line 77: x: (1, b, c, h, w)
        Line 82: x: (max_batch_size, b, c, h, w)
        Line 93: x: (max_batch_size*b, c, h, w)

        Internally, nb_sample and batch_size are considered as extended batch size and parallelized. 

        But setting max_batch_size equals to batch_size is not wrong. It can be any value. I think it will not make a big difference in comp. time. 
        '''

        time_s = time.perf_counter()
        model.eval()
        for batch_i, data in enumerate(val_loader):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda().long()
            loss_b = 0.0

            labels_gt = labels
            labels = torch.randint(0, 100, labels.size(), device='cuda').long()


            ''' Test Original Image on the model '''
            with torch.no_grad():
                inputs_norm = normalize(inputs)
                rd = model_o(inputs_norm, n_steps=n_steps, isReturnAllStep=True)
                pred = rd['pred']
                tf_clean_steps = []
                for step in range(n_steps-compensation):
                    tf = misc.get_classification_TF(pred[step], labels_gt)
                    #print(tf)
                    ss = torch.squeeze(tf)
                    tf_clean_steps.append(ss)
                tf_clean_steps = torch.stack(tf_clean_steps).t()
                #print(tf_clean_steps.size())

                if batch_i == 0:
                    tf_clean = tf_clean_steps
                else:
                    tf_clean = torch.cat((tf_clean, tf_clean_steps), 0)

            ''' Attack image '''
            tic = time.time()
            if eps != 0.0:              
                x_adv = attack.perturb(inputs, labels)
            else:
                x_adv = inputs
            toc = time.time()
            #print("elapsed time: {} sec".format(toc - tic))
            logging.info("batch: {},   elapsed time: {} sec".format(batch_i, toc - tic))


            ##################### inference long steps #######################
            with torch.no_grad():
                x_adv_norm = normalize(x_adv)
                return_dict = model_o(x_adv_norm, n_steps=n_steps, isReturnAllStep=True)

            tf_adv_steps = []
            for step in range(n_steps-compensation):
                pred = torch.squeeze(return_dict['pred'][step])
                tf = misc.get_classification_TF(pred, torch.squeeze(labels))
                #print(tf)
                ss = torch.squeeze(tf)
                tf_adv_steps.append(ss)
            tf_adv_steps = torch.stack(tf_adv_steps).t()
            if batch_i == 0:
                tf_adv = tf_adv_steps
            else:
                tf_adv = torch.cat((tf_adv, tf_adv_steps), 0)

            ''' linf '''
            linf = torch.squeeze(torch.norm(torch.abs(inputs.view(inputs.size(0), -1)-x_adv.view(x_adv.size(0), -1)), p=float('inf'), dim=1))
            ''' accum imgs '''
            if batch_i == 0:
                linf_accum = linf
                x_adv_accum = x_adv
                inputs_accum = inputs
            else:
                linf_accum = torch.cat((linf_accum, linf), 0)
                x_adv_accum = torch.cat((x_adv_accum, x_adv), 0)
                inputs_accum = torch.cat((inputs_accum, inputs), 0)


            if eps == 0.0:
                ls = labels_gt
            else:
                ls = labels
            for step in range(n_steps-compensation):
                acc1, acc5 = misc.accuracy(torch.squeeze(return_dict['pred'][step]), torch.squeeze(ls), topk=(1, 5))
                acc_top1[step].update(acc1[0], batch_size)
                acc_top5[step].update(acc5[0], batch_size)

            #print(batch_i)
            if batch_i == 10:
                n_plots = 1 # min(56, int(args.batch_size/int(torch.cuda.device_count())))
                misc.plot_one_sample_from_images(normalize(inputs[:n_plots]),
                        'plots', 'test_e'+str(0)+'b'+str(batch_i)+str(int(eps*1000))+'_in.jpg')
                misc.plot_one_sample_from_images(normalize(x_adv[:n_plots]), 
                        'plots', 'test_e'+str(0)+'b'+str(batch_i)+str(int(eps*1000))+'_in_adv.jpg')
            if es != None:
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

        np.save('tf_adv_e'+str(int(eps*10000))+'.npy', tf_adv.cpu().numpy())
        np.save('tf_clean_e'+str(int(eps*10000))+'.npy', tf_clean.cpu().numpy())
        np.save('perturb_e'+str(int(eps*10000))+'.npy', linf_accum.cpu().numpy())
        np.save('img_adv_e'+str(int(eps*10000))+'.npy', x_adv_accum.cpu().numpy())
        np.save('img_clean_e'+str(int(eps*10000))+'.npy', inputs_accum.cpu().numpy())

