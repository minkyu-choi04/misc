import os 
from os import listdir
from os.path import isfile, join
import numpy as np
from PIL import Image
from multiprocessing import Pool

from pathlib import Path
def make_dir(path, parents=False):
    Path(os.path.expanduser(path)).mkdir(parents=parents, exist_ok=True)

def get_fns_all(mypath: str) -> list:
    """Get and returns list of file names in the given directory. 
    """
    
    allfiles = [f for f in listdir(mypath)]
    return allfiles

def get_fns(mypath: str) -> list:
    """Get and returns list of file names in the given directory. 
    Only files, not dirs are returned. """
    
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    return onlyfiles

def get_fns_recursively(mypath: str) -> list:
    fns = []
    for path, currentDirectory, files in os.walk(mypath):
        for file in files:
            fns.append(os.path.join(path, file))
    return fns

def check_dataset_null_file(mypath: str) -> str:
    '''2022.11.30. Check all the image files in the path recursively and return file name
    when it encounters file size 0. '''
    fns = get_fns_recursively(mypath)
    for fn in fns:
        if os.stat(fn).st_size == 0:
            return fn
        
    print('There is no empty file. ')

def CheckOne(f):
    # https://stackoverflow.com/questions/59155213/checking-for-corrupted-files-in-directory-with-hundreds-of-thousands-of-images-g
    try:
        im = Image.open(f)
        im.verify()
        im.close()
        # DEBUG: print(f"OK: {f}")
        return
    except (IOError, OSError, Image.DecompressionBombError):
        # DEBUG: print(f"Fail: {f}")
        return f

def check_dataset_null_file_PIL(mypath: str) -> str:
    # https://stackoverflow.com/questions/59155213/checking-for-corrupted-files-in-directory-with-hundreds-of-thousands-of-images-g
    # Create a pool of processes to check files
    p = Pool()

    # Create a list of files to process
    files = get_fns_recursively(mypath)

    print(f"Files to be checked: {len(files)}")

    # Map the list of files to check onto the Pool
    result = p.map(CheckOne, files)

    # Filter out None values representing files that are ok, leaving just corrupt ones
    result = list(filter(None, result)) 
    print(f"Num corrupt files: {len(result)}")
    print(result)





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