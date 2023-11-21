#from utils import add_flops_counting_methods
from flops_benchmark import add_flops_counting_methods
import torch
import numpy as np
''' code from https://github.com/allenai/elastic/blob/master/multilabel_classify.py 
exmaple codes:
    (pytorch_mk_131) min@libigpu0:~/research_minkyu/polar_cnn/pretrain/cart_warping2/align_comp_params/train_imagenet100/crnn/fixation_source_state_fms/vgg13v1/convrnn_cw84_b11p9_fullGRU_vgg13v1_attnSourceTdBu_pretrained120eWlrl_lrAdje100e140_countFlops
    (pytorch_mk_131) min@libigpu0:~/research_minkyu/polar_cnn/pretrain/cart_warping2/align_comp_params/train_imagenet100/count_flops/baseline_cartCNN_imgnet100_res224v1_lrAdj80e_countFlops
'''


def count_stats(model, image_s=224, image_c=3):
    ''' 2021.02.28.
    This function counts 1)#parameters, 2)flops.
    In order to use this function, the model has to be changed. 
        - The arguments of forward of the model must have image as the first argument. 
        - Return of the model does not matter. 
    '''
    # count number of parameters
    count = 0
    params = list()
    for n, p in model.named_parameters():
        if '.ups.' not in n:
            params.append(p)
            count += np.prod(p.size())
    print('Parameters:', count)

    # count flops
    model = add_flops_counting_methods(model)
    model.eval()
    image = torch.randn(1, image_c, image_s, image_s).cuda()

    model.start_flops_count()
    #model(args, image, isTrain=False).sum()
    #with torch.no_grad():
    #    _ = model(image, args, isTrain=False)
    with torch.no_grad():
        _ = model(image)
    print("GFLOPs", model.compute_average_flops_cost() / 1000000000.0)
    model.stop_flops_count()
    return

def count_stats_params(model):
    ''' 2021.02.28.
    This function counts #parameters.
    '''
    # count number of parameters
    count = 0
    params = list()
    for n, p in model.named_parameters():
        if '.ups.' not in n:
            params.append(p)
            count += np.prod(p.size())
    print('Parameters:', count)

