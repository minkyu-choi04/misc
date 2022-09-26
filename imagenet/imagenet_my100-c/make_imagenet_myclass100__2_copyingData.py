import shutil, errno
import numpy as np
import os

import sys
sys.path.append('/mnt/lls/local_export/3/home/choi574/git_libs/misc/')
import misc


'''20210315
This code will copy data and make imagenet100 with my own classes. 
The class sampling 100 from 1000 must be done before running this code. 
This code will be run at min@libigpu3:/home/min/datasets/
    python /mnt/lls/local_export/3/home/choi574/git_libs/misc/imagenet/imagenet_my100/make_imagenet_myclass100__2_copyingData.py
'''

'''20220730
Run on (pytorch_mk_141) choi574@libigpu3:~/imagenet-c
'''


def copyanything(src, dst):
    '''2020.11.14
    https://stackoverflow.com/a/1994840
    '''
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise


currs = {}
currs['blur'] = ['defocus_blur',  'glass_blur',  'motion_blur',  'zoom_blur']
currs['digital'] = ['contrast',  'elastic_transform',  'jpeg_compression',  'pixelate']
currs['extra'] = ['gaussian_blur',  'saturate',  'spatter',  'speckle_noise']
currs['noise'] = ['gaussian_noise',  'impulse_noise',  'shot_noise']
currs['weather'] = ['brightness',  'fog',  'frost',  'snow']

source_dir = '/home/choi574/imagenet-c/'
dest_dir = '/home/choi574/datasets/ImageNetC_myclass100/'

# from Emergent Properties of Foveated Perceptual Systems
classes100 = np.load('/mnt/lls/local_export/3/home/choi574/git_libs/misc/imagenet/imagenet_my100/imagenet_classes100.npy')
''' copy and make a new dataset of 100 from 365 '''




#### for test ####

for c in currs.keys():
    for c_sub in currs[c]:
        for lev in range(5):
            cnt = 0
            for myclass in classes100:
                source_inst = os.path.join(source_dir, c, c_sub, str(lev+1), myclass)
                dest_inst_parents = os.path.join(dest_dir, c, c_sub, str(lev+1))
                misc.make_dir(dest_inst_parents, parents=True)
                dest_inst = os.path.join(dest_dir, c, c_sub, str(lev+1), myclass)
                copyanything(source_inst, dest_inst)
                cnt += 1
                #print(dest_inst)
        print(c, c_sub, lev, cnt)

'''
cnt = 0
for n in classes100:
    print('test: ', n, cnt)
    source_inst = os.path.join(source_dir, 'val', n)
    dest_inst = os.path.join(dest_dir, 'val', n)
    copyanything(source_inst, dest_inst)
    cnt += 1
'''
print('end test')

