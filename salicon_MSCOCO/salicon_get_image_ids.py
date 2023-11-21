import os
import sys
import numpy as np

sys.path.append('/mnt/lls/home/choi574/git_libs/misc/')
import utils_dirs



'''
2023.01.17, Minkyu
This code is used to collect and save image_ids of salicon dataset. 
Because the salicon images are all from MSCOCO, there are shared images from both MSCOCO and salicon. 
This code collects image_ids and save it for later use. 
Run this code on libigpu5. 
'''

dir_train = '/home/choi574/datasets/salicon_original/image/images/train/'
dir_val = '/home/choi574/datasets/salicon_original/image/images/val/'

fns_train = utils_dirs.get_fns_recursively(dir_train)
fns_val = utils_dirs.get_fns_recursively(dir_val)

ids_train = []
for fn in fns_train:
    fn_only = fn.split('/')[-1] # ex) COCO_val2014_000000580248.jpg
    fn_only_nums = fn_only.split('_')[2][:-4]
    print(fn_only_nums)
    if len(fn_only_nums) != len('000000580248'):
        print('[Error] parsing file name is not correct')
        break
    ids_train.append(int(fn_only_nums))

ids_val = []
for fn in fns_val:
    fn_only = fn.split('/')[-1]
    fn_only_nums = fn_only.split('_')[2][:-4]
    if len(fn_only_nums) != len('000000580248'):
        print('[Error] parsing file name is not correct')
        break
    ids_val.append(int(fn_only_nums))


np.save('salicon_image_ids_train.npy', ids_train)
np.save('salicon_image_ids_val.npy', ids_val)

