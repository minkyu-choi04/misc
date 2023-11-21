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
sys.path.append('/home/cminkyu/git_libs/misc/')
#sys.path.append('/home/choi574/research_mk/git_libs_onlyForLibi2/misc/')
import misc
sys.path.append('/mnt/lls/local_export/3/home/choi574/git_libs/misc/adversarial_attacks/')
sys.path.append('/home/cminkyu/git_libs/misc/adversarial_attacks/')
import utils 

import logging

from advertorch.attacks import LinfSPSAAttack, LinfPGDAttack_EOT_mem
from advertorch.utils import NormalizeByChannelMeanStd
from advertorch_examples.utils import get_panda_image, bchw2bhwc, bhwc2bchw


def attack_LinfPGD_EOT(args, model_o, eps_list, n_steps=4, batch_size=128, cudnn_backends=True, embedding_size=224, 
        n_iter=5, eps_iter=3, es=None, compensation=0, n_EOT=1, isAvgSteps=False, 
        shuffle_test=False, save_imgs=False, save_more=False):
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
    print('Save more info to npy: [{}]'.format(save_more))

    if 0.0 not in eps_list:
        #eps_list.append(0.0)
        eps_list = [0.0] + eps_list

    torch.backends.cudnn.enabled = cudnn_backends


    model = model_o.cuda()
    model_o.eval()
    model.eval()



    transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),  # Ensure the images are grayscale
                transforms.ToTensor(),
                            ])
    # Load the dataset from the directory
    test_dataset = datasets.ImageFolder(root='~/datasets/mnist23_test_topk/top20/test/', transform=transform)
    # Create a DataLoader
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    print('=============================\n\n')
    print('\n\n=============================')

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
        attack = LinfPGDAttack_EOT_mem(
            model, loss_fn=nn.CrossEntropyLoss(reduction="mean"), eps=eps,
            nb_iter=n_iter, eps_iter=eps/eps_iter, rand_init=True, clip_min=0.0, clip_max=1.0,
            targeted=True, n_EOT=n_EOT, isAvgSteps=isAvgSteps)
        ''' reduction=mean is set to fair comparison. 2021.12.28 '''

        time_s = time.perf_counter()

        model.eval()
        for batch_i, data in enumerate(val_loader):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda().long()
            
            # Embed images
            inputs = utils.embed_mnist_images(inputs, embedding_size, isCenter=True)
            inputs = inputs.expand(-1, 3, -1, -1)

            loss_b = 0.0
            labels_gt = labels

            #labels = torch.randint(0, 10, labels.size(), device='cuda').long()
            labels = (labels + 1) % 10


            ''' Test Original Image on the model '''
            with torch.no_grad():
                inputs_norm = inputs
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
                x_adv_norm = x_adv
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
            linf = torch.squeeze(torch.norm(torch.abs(inputs.reshape(inputs.size(0), -1)-x_adv.reshape(x_adv.size(0), -1)), p=float('inf'), dim=1))
            ''' accum imgs '''
            if batch_i == 0:
                linf_accum = linf
                if save_more:
                    x_adv_accum = x_adv
                    fixs_accum = torch.stack(return_dict['fixs_xy']).permute(1, 0, 2) # (b, step, 2)
                #inputs_accum = inputs
            else:
                linf_accum = torch.cat((linf_accum, linf), 0)
                if save_more:
                    x_adv_accum = torch.cat((x_adv_accum, x_adv), 0)
                    fixs_accum = torch.cat((fixs_accum, torch.stack(return_dict['fixs_xy']).permute(1, 0, 2)), 0)
                #inputs_accum = torch.cat((inputs_accum, inputs), 0)


            if eps == 0.0:
                ls = labels_gt
            else:
                ls = labels
            for step in range(n_steps-compensation):
                acc1, acc5 = misc.accuracy(torch.squeeze(return_dict['pred'][step]), torch.squeeze(ls), topk=(1, 5))
                acc_top1[step].update(acc1[0], batch_size)
                acc_top5[step].update(acc5[0], batch_size)

            #print(batch_i)
            if save_imgs and batch_i==0:
                #n_plots = min(56, int(args.batch_size))
                n_plots = min(16, int(batch_size))
                '''misc.plot_one_sample_from_images(inputs[:n_plots],
                        'plots', 'test_e'+str(0)+'b'+str(batch_i)+str(int(eps*1000))+'_in.jpg')
                misc.plot_one_sample_from_images(x_adv[:n_plots], 
                        'plots', 'test_e'+str(0)+'b'+str(batch_i)+str(int(eps*1000))+'_in_adv.jpg')
                '''
                misc.plot_samples_from_images(inputs[:n_plots], n_plots, 
                        'plots', 'test_e'+str(0)+'b'+str(batch_i)+str(int(eps*1000))+'_in.jpg')
                misc.plot_samples_from_images(x_adv[:n_plots], n_plots,
                        'plots', 'test_e'+str(0)+'b'+str(batch_i)+str(int(eps*1000))+'_in_adv.jpg')
                for step in range(n_steps):
                    fixs_until = torch.stack(return_dict['fixs_xy'][:step+1]).permute(1, 0, 2) # (b, step, 2)
                    img_fixs = misc.mark_fixations_history(inputs[:n_plots], fixs_until[:n_plots])
                    misc.plot_samples_from_images(img_fixs, n_plots,
                           'plots', 'test_e'+str(0)+'b'+str(batch_i)+str(int(eps*1000))+'_fix_hist'+'_s'+str(step))
                    #print('np_e'+str(0)+'b'+str(batch_i)+'_fix_hist'+'_s'+str(step)+'.npy')
                    #np.save('./plots/np_e'+str(0)+'b'+str(batch_i)+'_fix_hist'+str(int(eps*1000))+'_s'+str(step)+'.npy', img_fixs.cpu().numpy())
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
        if save_more:
            np.save('img_adv_e'+str(int(eps*10000))+'.npy', x_adv_accum.cpu().numpy())
            np.save('fixs_e'+str(int(eps*10000))+'.npy', fixs_accum.cpu().numpy())
        #np.save('img_clean_e'+str(int(eps*10000))+'.npy', inputs_accum.cpu().numpy())

