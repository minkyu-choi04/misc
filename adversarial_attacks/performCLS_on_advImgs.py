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
sys.path.append('/mnt/lls/local_export/3/home/choi574/git_libs/misc/imagenet/imagenet_my100/imgnet_my100_val100/')
sys.path.append('/home/cminkyu/git_libs/misc/imagenet/imagenet_my100/imgnet_my100_val100/')
#sys.path.append('/home/choi574/research_mk/git_libs_onlyForLibi2/misc/imagenet/imagenet_my100/')
import dataloader_imagenet100my_val100 as dl
sys.path.append('/mnt/lls/local_export/3/home/choi574/git_libs/misc/adversarial_attacks/')
sys.path.append('/home/cminkyu/git_libs/misc/adversarial_attacks/')
#sys.path.append('/home/choi574/research_mk/git_libs_onlyForLibi2/misc/adversarial_attacks/')
import utils 

import logging

from advertorch.attacks import LinfSPSAAttack, LinfPGDAttack_EOT_mem
from advertorch.utils import NormalizeByChannelMeanStd
from advertorch_examples.utils import get_panda_image, bchw2bhwc, bhwc2bchw


def perform_gridFixations(args, model_o, path_advImgs, path_advLabels, path_cleanImgs, path_cleanLabels, path_out, cudnn_backends=True): 
    '''
    2022.02.17.
    See example code
        (pytorch_mk_131) choi574@libigpu7:~/research_mk/foveated_adv/unified_stage12/get_data_sharedAdvLabels/
    This function receives adversarial examples and a model using fixations. 
    Then, this function makes a grid of fixations and test its robustness. 
    This function assumes targeted attack. 
    '''
    torch.backends.cudnn.enabled = cudnn_backends


    normalize = NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model = torch.nn.Sequential(normalize, model_o).cuda()
    model_o.eval()
    model.eval()

    adv_imgs = torch.tensor(np.load(path_advImgs), device='cuda')
    adv_labels = torch.tensor(np.load(path_advLabels), device='cuda')
    clean_imgs = torch.tensor(np.load(path_cleanImgs), device='cuda')
    clean_labels = torch.tensor(np.load(path_cleanLabels), device='cuda')
    batch_s = adv_imgs.size(0)

    map_clean_TF = torch.zeros((batch_s, 32,32), device='cuda')
    map_clean_confid = torch.zeros((batch_s, 32,32), device='cuda')
    map_adv_TF = torch.zeros((batch_s, 32,32), device='cuda')
    map_adv_confid = torch.zeros((batch_s, 32,32), device='cuda')
    map_adv_TF_gtL = torch.zeros((batch_s, 32,32), device='cuda') # confidence on GT label
    map_adv_confid_gtL = torch.zeros((batch_s, 32,32), device='cuda')

    xlocs = np.linspace(-1.0, 1.0, num=32)
    ylocs = np.linspace(-1.0, 1.0, num=32)


    # normalize images
    adv_imgs_norm = normalize(adv_imgs)
    clean_imgs_norm = normalize(clean_imgs)

    # print accuracy with center fixations
    fixs_xy = torch.zeros((batch_s, 2), device='cuda')
    adv_rd = model_o(adv_imgs_norm, fixs_xy_given=fixs_xy)
    clean_rd = model_o(clean_imgs_norm, fixs_xy_given=fixs_xy)

    adv_tf = misc.get_classification_TF(adv_rd['pred'][-1], adv_labels)
    clean_tf = misc.get_classification_TF(clean_rd['pred'][-1], clean_labels)
    print('accs adv / clean ::: {}% / {}%'.format(torch.mean(adv_tf.int().float()), torch.mean(clean_tf.int().float())))



    with torch.no_grad():
        for ix, x in enumerate(xlocs): # -1~1
            for iy, y in enumerate(ylocs): # -1~1
                fixs_x = torch.ones((batch_s,1), device='cuda')*x
                fixs_y = torch.ones((batch_s,1), device='cuda')*y
                fixs_xy = torch.cat((fixs_x, fixs_y), 1)
                
                adv_rd = model_o(adv_imgs_norm, fixs_xy_given=fixs_xy)
                # for adv images on adv labels
                adv_tf = misc.get_classification_TF(adv_rd['pred'][0], adv_labels)
                map_adv_TF[:, iy, ix] = torch.squeeze(adv_tf)
                adv_confid = torch.squeeze(torch.gather(torch.nn.functional.softmax(adv_rd['pred'][0], dim=1), dim=1, index=adv_labels.unsqueeze(-1)))
                map_adv_confid[:, iy, ix] = adv_confid
                # for adv images on GT labels
                adv_tf_gtL = misc.get_classification_TF(adv_rd['pred'][0], clean_labels)
                map_adv_TF_gtL[:, iy, ix] = torch.squeeze(adv_tf_gtL)
                adv_confid_gtL = torch.squeeze(torch.gather(torch.nn.functional.softmax(adv_rd['pred'][0], dim=1), dim=1, index=clean_labels.unsqueeze(-1)))
                map_adv_confid_gtL[:, iy, ix] = adv_confid_gtL
                

                clean_rd = model_o(clean_imgs_norm, fixs_xy_given=fixs_xy)
                clean_tf = misc.get_classification_TF(clean_rd['pred'][0], clean_labels)
                map_clean_TF[:, iy, ix] = torch.squeeze(clean_tf)
                clean_confid = torch.squeeze(torch.gather(torch.nn.functional.softmax(clean_rd['pred'][0], dim=1), dim=1, index=clean_labels.unsqueeze(-1)))
                map_clean_confid[:, iy, ix] = clean_confid

    results = {}
    results['map_adv_TF'] = map_adv_TF.cpu().numpy()
    results['map_adv_TF_gtL'] = map_adv_TF_gtL.cpu().numpy()
    results['map_clean_TF'] = map_clean_TF.cpu().numpy()
    results['map_adv_confid'] = map_adv_confid.cpu().numpy()
    results['map_adv_confid_gtL'] = map_adv_confid_gtL.cpu().numpy()
    results['map_clean_confid'] = map_clean_confid.cpu().numpy()
    results['adv_labels'] = adv_labels.cpu().numpy()
    results['clean_labels'] = clean_labels.cpu().numpy()
    results['adv_imgs'] = np.load(path_advImgs)
    results['clean_imgs'] = np.load(path_cleanImgs)


    misc.make_dir(path_out)
    fn_results = os.path.join(path_out, 'results_' + path_advImgs.split('/')[-1])
    np.save(fn_results, results)


