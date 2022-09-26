import shutil, errno
import numpy as np
import os

''' 2021.02.25
This file is used to generate a subset dataset of places365_larger with 512x512 res. . 
The sampled classes are saved in the current directory as classes100.npy
This code is basically run after the "make_places_class100_standard256res.py". 
The sampled classes are generated from the "make_places_class100_large512res.py" 
and saved in the current dir and will be used in this code. 

This code is run at min@libigpu3:/home/libi/HDD1/users/minkyu/DATASETS/IMAGE/places365_large 
And sampled dataset is saved at the current dir. 

This code is applied to the image dataset resized to 512x512, which called places365_large'. 

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


source_dir = '/home/libi/HDD1/users/minkyu/DATASETS/IMAGE/places365_large/'
dest_dir = '/home/libi/HDD1/users/minkyu/DATASETS/IMAGE/places100_sampled_large/'

classes = []
#classes = ["aquarium", "badlands", "bedroom", "bridge", "campus", "corridor", "forest_path", "highway", "hospital", "industrial_area", "japanese_garden", "kitchen", "mansion", "mountain", "ocean", "office", "restaurant", "skyscraper", "train_interior", "waterfall"]
# from Emergent Properties of Foveated Perceptual Systems
''' get all classes of Places365
    this code is run at libigpu0, /home/libi/HDD1/minkyu/Places2/places365_standard'''
'''fn = '/home/libi/HDD1/minkyu/Places2/places365_standard/train.txt'
with open(fn) as f:  
    labels = f.readlines()
    for label in labels: 
        ls = label.split('/')[1]
        if ls not in classes:
            classes.append(ls)
print('total labels collected: ', len(classes))
'''
''' sample 100 classes from total 365 classes '''
'''classes100 = np.random.choice(np.array(classes), 100, replace=False)
print('selected labels: ',  len(classes100))
'''
''' save sampled classes into file '''
''' np.save('./classes100.npy', classes100)'''
#np.save('/mnt/lls/local_export/3/home/choi574/git_libs/misc/places/classes100.npy', classes100)





''' saved classes will be loaded here '''
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
    print('val: ', n, cnt)
    source_inst = os.path.join(source_dir, 'val', n)
    dest_inst = os.path.join(dest_dir, 'val', n)
    cnt += 1
    copyanything(source_inst, dest_inst)
print('end test')

