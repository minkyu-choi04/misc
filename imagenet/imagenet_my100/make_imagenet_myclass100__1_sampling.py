import shutil, errno
import numpy as np
import os

'''20210315
This code is used to draw 100 samples from 1000 classes of imagenet. 
This code must be drawn only once, otherwise different samples will be drawn. 
Previously, I used imagenet100 with classes following other paper. 
But I will use my own 100 classes. 
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


source_dir = '/home/libi/'
dest_dir = '/home/libi/'

classes = []
classes_name = []
classes_idx = []
# from Emergent Properties of Foveated Perceptual Systems
''' get all classes of ImageNet1000
    this code is run at libilab, /export/home/choi574/git_libs/misc/imagenet/imagenet_my100
    Once this is run, it will be disabled and the saved classes will be used. '''
fn = 'map_clsloc.txt'
with open(fn) as f:  
    labels = f.readlines()
    for label in labels: 
        ls = label.split(' ')[0]
        if ls not in classes:
            classes.append(ls)
            #classes_name.append(label.split(' ')[2]
            #classes_idx.append(label.split(' ')[1]
print('total labels collected: ', len(classes))

''' sample 100 classes from total 1000 classes '''
classes100 = np.random.choice(np.array(classes), 100, replace=False)
print('selected labels: ',  len(classes100))
print(classes100)

''' save sampled classes into file '''
np.save('./imagenet_classes100.npy', classes100)
#np.save('/mnt/lls/local_export/3/home/choi574/git_libs/misc/places/classes100.npy', classes100)

#classes100 = np.load('/mnt/lls/local_export/3/home/choi574/git_libs/misc/places/classes100.npy')
''' copy and make a new dataset of 100 from 365 '''
#### for train ####
'''cnt = 0
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
'''
