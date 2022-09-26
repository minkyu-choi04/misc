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
sys.path.append('/home/choi574/git_libs/imagenet/imagenet_my100/imgnet_my100_val100/')
#sys.path.append('/home/choi574/research_mk/git_libs_onlyForLibi2/misc/imagenet/imagenet_my100/')
import dataloader_imagenet100my_val100 as dl
sys.path.append('/mnt/lls/local_export/3/home/choi574/git_libs/misc/adversarial_attacks/')
sys.path.append('/home/choi574/git_libs/misc/adversarial_attacks/')
#sys.path.append('/home/choi574/research_mk/git_libs_onlyForLibi2/misc/adversarial_attacks/')
import utils 

import logging

from advertorch.attacks import LinfSPSAAttack, LinfPGDAttack_EOT
from advertorch.utils import NormalizeByChannelMeanStd
from advertorch_examples.utils import get_panda_image, bchw2bhwc, bhwc2bchw


def attack_LinfPGD_EOT(args, model_o, eps_list, n_steps=16, batch_size=128, img_s_load=256+128, img_s_return=224+112, cudnn_backends=True, n_iter=5, eps_iter=3, es=None, compensation=0, n_EOT=1, isAvgSteps=False):
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
    logging.basicConfig(filename='logs.log', filemode='w', level=logging.DEBUG, format='%(asctime)s %(message)s')
    print('PGD-EOT: {} iterations'.format(n_EOT))
    print('Current attack script requires model to return [return_dict]')
    print('Use [Val1000] dataset')
    print('use AvgStep: [{}]'.format(isAvgSteps))

    if 0.0 not in eps_list:
        eps_list.append(0.0)

    torch.backends.cudnn.enabled = cudnn_backends


    normalize = NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model = torch.nn.Sequential(normalize, model_o).cuda()
    model_o.eval()
    model.eval()

    val_loader = dl.load_imagenet_myclass100_val1000_for_AdvAttacks(batch_size=batch_size, img_s_load=img_s_load, img_s_return=img_s_return, server_type=args.server_type)
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
        attack = LinfPGDAttack_EOT(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps,
            nb_iter=n_iter, eps_iter=eps/eps_iter, rand_init=True, clip_min=0.0, clip_max=1.0,
            targeted=False, n_EOT=n_EOT, isAvgSteps=isAvgSteps)

        time_s = time.perf_counter()

        model.eval()
        for batch_i, data in enumerate(val_loader):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda().long()
            loss_b = 0.0


            ''' Test Original Image on the model '''
            with torch.no_grad():
                inputs_norm = normalize(inputs)
                rd = model_o(inputs_norm, n_steps=n_steps, isReturnAllStep=True)
                pred = rd['pred']
                tf_clean_steps = []
                for step in range(n_steps-compensation):
                    tf = misc.get_classification_TF(pred[step], labels)
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
                #x_adv_accum = x_adv
                #inputs_accum = inputs
            else:
                linf_accum = torch.cat((linf_accum, linf), 0)
                #x_adv_accum = torch.cat((x_adv_accum, x_adv), 0)
                #inputs_accum = torch.cat((inputs_accum, inputs), 0)


            for step in range(n_steps-compensation):
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
        #np.save('img_adv_e'+str(int(eps*10000))+'.npy', x_adv_accum.cpu().numpy())
        #np.save('img_clean_e'+str(int(eps*10000))+'.npy', inputs_accum.cpu().numpy())

