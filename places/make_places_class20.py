import shutil, errno
import os

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


source_dir = '/home/choi574/datasets/places365_standard/'
dest_dir = '/home/choi574/datasets/places20/'

classes = ["aquarium", "badlands", "bedroom", "bridge", "campus", "corridor", "forest_path", "highway", "hospital", "industrial_area", "japanese_garden", "kitchen", "mansion", "mountain", "ocean", "office", "restaurant", "skyscraper", "train_interior", "waterfall"]
# from Emergent Properties of Foveated Perceptual Systems


#### for train ####
for n in classes:
    print(n)
    source_inst = os.path.join(source_dir, 'train', n)
    dest_inst = os.path.join(dest_dir, 'train', n)
    copyanything(source_inst, dest_inst)
print('end train')

#### for test ####
for n in classes:
    source_inst = os.path.join(source_dir, 'val', n)
    dest_inst = os.path.join(dest_dir, 'val', n)
    copyanything(source_inst, dest_inst)
print('end test')

