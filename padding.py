import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

def _cyclic_pad(x, pad, axis):
    """Cyclic padding.

    Args:
        x: torch.Tensor, input tensor
        pad: int or (int, int), pad amount
        axis: int, axis along which to pad

    Returns:
        torch.Tensor, padded tensor
    """
    if type(pad) is int:
        pad = (pad, pad)
    if pad[0] == 0 and pad[1] == 0:
        return x
    if pad[1] > 0:
        left = x.narrow(axis, 0, pad[1])
    if pad[0] > 0:
        right = x.narrow(axis, x.shape[axis] - pad[0], pad[0])
    if pad[0] == 0:
        return torch.cat([x, left], axis)
    if pad[1] == 0:
        return torch.cat([right, x], axis)
    return torch.cat([right, x, left], axis)


def _pad2d(x, pad, mode):
    """2D padding.

    Args:
        x: torch.Tensor, input tensor
        pad: int or (int, int), pad amount
        mode: str or (str, str), one of 'constant', 'reflect', 'replicate', or 'cyclic'

    Returns:
        torch.Tensor, padded tensor
    """
    if type(pad) is int:
        pad = (pad, pad)
    if type(mode) is str:
        mode = (mode, mode)

    hmode, wmode = mode # changed 200302
    hpad, wpad = pad
    out = x

    if wmode == 'cyclic':
        out = _cyclic_pad(out, pad=wpad, axis=3)
    elif wmode is not None:
        out = F.pad(out, (wpad, wpad, 0, 0), wmode)

    if hmode == 'cyclic':
        out = _cyclic_pad(out, pad=hpad, axis=2)
    elif hmode is not None:
        out = F.pad(out, (0, 0, hpad, hpad), hmode)

    return out

