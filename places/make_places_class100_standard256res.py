import shutil, errno
import numpy as np
import os

''' 2021.02.25
This file is used to generate a subset dataset of places365_standard. 
The sampled classes are saved in the current directory as classes100.npy
This code must not be run twice because it will sample a new set of classes. 
If you want to make a same set of dataset as before, you need to modify this code 
to use the saved sampled classes instead of sampling them again. 

This code is run at min@libigpu0:/home/libi/HDD1/minkyu/Places2/places365_standard
And sampled dataset is saved at /home/libi/HDD1/minkyu/Places2/places100_sampled/

This code is applied to the image dataset resized to 256x256, which called places365_standard'. 

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


source_dir = '/home/libi/HDD1/minkyu/Places2/places365_standard/'
dest_dir = '/home/libi/HDD1/minkyu/Places2/places100_sampled/'

classes = []
#classes = ["aquarium", "badlands", "bedroom", "bridge", "campus", "corridor", "forest_path", "highway", "hospital", "industrial_area", "japanese_garden", "kitchen", "mansion", "mountain", "ocean", "office", "restaurant", "skyscraper", "train_interior", "waterfall"]
# from Emergent Properties of Foveated Perceptual Systems
''' get all classes of Places365
    this code is run at libigpu0, /home/libi/HDD1/minkyu/Places2/places365_standardi
    Once this is run, it will be disabled and the saved classes will be used. '''
'''fn = '/home/libi/HDD1/minkyu/Places2/places365_standard/train.txt'
with open(fn) as f:  
    labels = f.readlines()
    for label in labels: 
        ls = label.split('/')[1]
        if ls not in classes:
            classes.append(ls)
print('total labels collected: ', len(classes))'''

''' sample 100 classes from total 365 classes '''
'''classes100 = np.random.choice(np.array(classes), 100, replace=False)
print('selected labels: ',  len(classes100))'''

''' save sampled classes into file '''
'''np.save('./classes100.npy', classes100)'''
#np.save('/mnt/lls/local_export/3/home/choi574/git_libs/misc/places/classes100.npy', classes100)

classes100 = np.load('/mnt/lls/local_export/3/home/choi574/git_libs/misc/places/classes100.npy')
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

