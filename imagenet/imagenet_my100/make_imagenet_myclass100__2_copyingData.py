import shutil, errno
import numpy as np
import os

'''20210315
This code will copy data and make imagenet100 with my own classes. 
The class sampling 100 from 1000 must be done before running this code. 
This code will be run at min@libigpu3:/home/min/datasets/
    python /mnt/lls/local_export/3/home/choi574/git_libs/misc/imagenet/imagenet_my100/make_imagenet_myclass100__2_copyingData.py
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


source_dir = '/home/libiadm/data/ImageNet2012/'
dest_dir = '/home/min/datasets/ImageNet2012_myclass100/'

# from Emergent Properties of Foveated Perceptual Systems
classes100 = np.load('/mnt/lls/local_export/3/home/choi574/git_libs/misc/imagenet/imagenet_my100/imagenet_classes100.npy')
''' copy and make a new dataset of 100 from 365 '''
#### for train ####
cnt = 0
for n in classes100:
    print('train: ', n, cnt)
    source_inst = os.path.join(source_dir, 'train', n)
    dest_inst = os.path.join(dest_dir, 'train', n)
    copyanything(source_inst, dest_inst)
    cnt += 1
print('end train')

#### for test ####
cnt = 0
for n in classes100:
    print('test: ', n, cnt)
    source_inst = os.path.join(source_dir, 'val', n)
    dest_inst = os.path.join(dest_dir, 'val', n)
    copyanything(source_inst, dest_inst)
    cnt += 1
print('end test')

