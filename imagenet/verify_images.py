from PIL import Image
from pathlib import Path
import glob

'''
root_dir = f'/tmpssd/minkyu/ImageNet2012/train/'
for filename in glob.iglob(root_dir + '**/*.JPEG', recursive=True):
     print(filename)
root_dir = f'/tmpssd/minkyu/ImageNet2012/val/'
for filename in glob.iglob(root_dir + '**/*.JPEG', recursive=True):
     print(filename)
'''
#root_dir = f'/tmpssd/minkyu/ImageNet2012/train/'
root_dir = f'/home/libiadm/datasets/ImageNet2012/train/'
filename_train = list(glob.iglob(root_dir + '**/*.JPEG', recursive=True))

#root_dir = f'/tmpssd/minkyu/ImageNet2012/val/'
root_dir = f'/home/libiadm/datasets/ImageNet2012/val/'
filename_val = list(glob.iglob(root_dir + '**/*.JPEG', recursive=True))



for p in filename_train:
    try:
        im = Image.open(p)
        im2 = im.convert('RGB')
    except OSError:
        print("Cannot load : {}".format(p))


for p in filename_val:
    try:
        im = Image.open(p)
        im2 = im.convert('RGB')
    except OSError:
        print("Cannot load : {}".format(p))


'''
https://stackoverflow.com/questions/2212643/python-recursive-folder-read
https://github.com/lucidrains/stylegan2-pytorch/issues/7
'''
