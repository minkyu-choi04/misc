import os
import sys
import numpy as np

sys.path.append('/mnt/lls/home/choi574/git_libs/misc/')
import utils_dirs


dir_img = '/home/choi574/datasets/salicon_original/image/images/train/'
dir_map = '/home/choi574/datasets/salicon_original/maps/train/'

fns_img = utils_dirs.get_fns_recursively(dir_img)
fns_map = utils_dirs.get_fns_recursively(dir_map)

ids_img = []
for fn in fns_img:
    fn_only = fn.split('/')[-1].split('.')[0] # ex) COCO_val2014_000000580248
    ids_img.append(fn_only)

ids_map = []
for fn in fns_map:
    fn_only = fn.split('/')[-1].split('.')[0] # ex) COCO_val2014_000000580248
    ids_map.append(fn_only)





cnt1 = 0
for fn in ids_map:
    if fn not in ids_img:
        print('1. ', fn)
        cnt1 += 1

cnt2 = 0
for fn in ids_img:
    if fn not in ids_map:
        print('2. ', fn)
        cnt2 += 1

print(cnt1, cnt2)

