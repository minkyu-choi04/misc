import torch
import math
import torch.nn as nn
import numpy as np

def get_gaussian_kernel(attn_p, kernel_size=[192,256], sigma=1, channels=3, norm='max', device='cuda'):
    '''
    This function does not suuport batched sigma.
    For applying different sigmas for each element in a batch, 
        see /export/home/choi574/git_libs/convert_img_coords/Gaussian_RF.py

    Changed 20201124 
    Current sigma is not constrained but it must be positive. To avoid sigma being negative or 0, 
        sigma is wrapped by abs and added by small epsilon. 

    Args:
        attn_p: (b, 2), tensor, fixation center ranged in -1~1
        kernel_size: (int y, int x), size of kernel returned. 
        sigma: (b, 1), float, sigma of Gaussian kernel.
    Return:
        gaussian_kernel: (b, c, h, w)
    '''
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    sigma = np.abs(sigma) + 1e-6
    batch_s = attn_p.size(0)

    x_coord = torch.arange(kernel_size[1], device=device)
    x_grid = x_coord.repeat(kernel_size[0]).view(kernel_size[0], kernel_size[1])

    y_coord = torch.arange(kernel_size[0], device=device)
    y_grid = y_coord.repeat(kernel_size[1]).view(kernel_size[1], kernel_size[0])
    y_grid = y_grid.t()

    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    xy_grid = xy_grid.unsqueeze(0).repeat(batch_s, 1, 1, 1)
    #a = torch.tensor(kernel_size, device=device).unsqueeze(0)
    #a = a.cuda()
    mean = (attn_p+1.0)/2.0 * torch.tensor([kernel_size[1], kernel_size[0]], device=device).unsqueeze(0) #modified 20200712
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                            -torch.sum((xy_grid - mean.unsqueeze(1).unsqueeze(1))**2., dim=-1) /\
                              (2*variance)
                    )
    # Make sure sum of values in gaussian kernel equals 1.
    # Always max value should be 1.
    if norm == 'max':
        gaussian_kernel = gaussian_kernel / torch.max(gaussian_kernel)
    elif norm == 'sum':
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    return gaussian_kernel.unsqueeze(1)
    
def get_gaussian_mask(attn_p, mask_prev=None, heatmap_s=[192,256], sigma=25, device='cuda'):
    '''
    Modified 2020.11.28. Can change sigma as an input args. 
    Modified 2020.07.06. Code from 'Gaussian IOR test.ipynb'
    Modified 2020.07.12. In the function 'get_gaussian_kernel', mean=(attn_p+1.0)... line is changed. 

    Args: 
        attn_p: (float x, float y), range -1~1
        mask_prev: (b, )
        heatmap_s: (int h, int w)
    Returns:
        gaussian_kernel: (b, 1, heatmap_s[0], heatmap_s[1])
    '''
    batch_s = attn_p.size()[0]
    if mask_prev is None:
        mask_prev = torch.ones([batch_s, heatmap_s[0], heatmap_s[1]], device=device)

    region_cur = get_gaussian_kernel(attn_p, heatmap_s, sigma, 1, device=device)
    
    mask_cur = nn.functional.relu(mask_prev - region_cur)
    return mask_cur
