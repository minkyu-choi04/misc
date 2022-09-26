import shutil, errno
import numpy as np
import os
from os import listdir
from os.path import isfile, join
'''20210608
This code is for sampling only 1 imgae from each class of validation set. 
The resulting validation set will have 100 image from 100 classes. 
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


source_dir = '/home/choi574/datasets/ImageNet2012_myclass100/val/'
dest_dir = '/home/choi574/datasets/ImageNet2012_myclass100_val1000/val/'


dirs = [x[0] for x in os.walk(source_dir)][1:]
print(dirs)
print(len(dirs))


for c in dirs:
    fn_list = [f for f in listdir(c) if isfile(join(c, f))]
    #print(len(fn_list))
    print(fn_list)
    img_sample = np.random.choice(np.array(fn_list), 10, replace=False)
    print(img_sample)

    for ii in range(len(img_sample)):
        source_dir_img = os.path.join(c, img_sample[ii])
        class_dir = c.split('/')[-1]
        dest_dir_class = os.path.join(dest_dir, class_dir)
        dest_dir_img = os.path.join(dest_dir_class, img_sample[ii])
        print(dest_dir_class)
        print(source_dir_img)
        print(dest_dir_img)

        os.makedirs(os.path.dirname(dest_dir_img), exist_ok=True)
        shutil.copyfile(source_dir_img, dest_dir_img)

