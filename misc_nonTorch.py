
import os
import argparse
import numpy as np
import random

import matplotlib.pyplot as plt
import sys

plt.switch_backend('agg') 


'''
2023.02.08. Minkyu
This code is copied from misc.py. 
Here, I just copied functions from misc.py to remove dependencies to torch. 
Do not make any modificaion to the functions here.
Make all modification to misc.py and copy them here. 
'''
import numpy as np
from scipy.stats import pearsonr

def pearson_r(x, y):
    """
    Calculate Pearson's correlation coefficient (r) between two numpy arrays,
    for each feature dimension.
    Written by ChatGPT

    Parameters
    ----------
    x : numpy array, size n_samples by feature_dimensions
        First input array.
    y : numpy array, size n_samples by feature_dimensions
        Second input array.

    Returns
    -------
    r : numpy array, size feature_dimensions
        Pearson's correlation coefficient (r) between x and y, for each feature dimension.
    """
    r = np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        r[i], _ = pearsonr(x[:,i], y[:,i])
        
    return r


def check_nan(np_array):
    '''
    2023.02.07. Minkyu
    Return True if any Nan is included'''
    return np.isnan(np_array).any()


from pathlib import Path
def make_dir(path, parents=False, exist_ok=True):
    Path(os.path.expanduser(path)).mkdir(parents=parents, exist_ok=exist_ok)

def remove_nan(fms_norm):
    '''2023.01.25. 
    Remove nan from given tensor. 
    Repalce nan value to 0.0. 
    '''
    
    fms_norm[fms_norm != fms_norm] = 0
    return fms_norm
