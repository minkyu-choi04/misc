import os 
from os import listdir
from os.path import isfile, join
import numpy as np


def get_fns(mypath: str) -> list:
    """Get and returns list of file names in the given directory. 
    Only files, not dirs are returned. """
    
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    return onlyfiles

def rename_files(root: str) -> None:
    """Rename 1.jpg as 001.jpg."""

    fn_list = get_fns(root)

    # get maximum index
    index_list = [] 
    for fn in fn_list:
        idx_image = fn.split('/')[-1].split('.')[0]
        index_list.append(int(idx_image))
    index_list = np.array(index_list)
    max_len = len(str(np.max(index_list)))

    for idx in index_list:
        fn_revised = str(idx).zfill(max_len)+'.jpg'
        print(idx, fn_revised)
        os.rename(os.path.join(root, str(idx)+'.jpg'), os.path.join(root, fn_revised))