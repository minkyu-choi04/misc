from PIL import Image
from pathlib import Path
import glob
from multiprocessing import Pool

'''
root_dir = f'/tmpssd/minkyu/ImageNet2012/train/'
for filename in glob.iglob(root_dir + '**/*.JPEG', recursive=True):
     print(filename)
root_dir = f'/tmpssd/minkyu/ImageNet2012/val/'
for filename in glob.iglob(root_dir + '**/*.JPEG', recursive=True):
     print(filename)
'''





def verify(p):
    try:
        im = Image.open(p)
        im2 = im.convert('RGB')
    except OSError:
        print("Cannot load : {}".format(p))


if __name__ == '__main__':
    #root_dir = f'/tmpssd/minkyu/ImageNet2012/train/'
    root_dir = f'/home/libiadm/datasets/ImageNet2012/train/'
    #root_dir = f'/home/choi574/datasets/ImageNet2012/train/'
    filename_train = list(glob.iglob(root_dir + '**/*.JPEG', recursive=True))

    #root_dir = f'/tmpssd/minkyu/ImageNet2012/val/'
    root_dir = f'/home/libiadm/datasets/ImageNet2012/val/'
    filename_val = list(glob.iglob(root_dir + '**/*.JPEG', recursive=True))
    
    with Pool() as pool:
        pool.map(verify, filename_train)
        print('finished')

'''
https://stackoverflow.com/questions/2212643/python-recursive-folder-read
https://github.com/lucidrains/stylegan2-pytorch/issues/7
'''
