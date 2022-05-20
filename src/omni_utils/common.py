# utils.common.py
#
# Author: Changhee Won (changhee.1.won@gmail.com)
#
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import annotations
from typing import *
import sys
import os
import os.path as osp
import traceback
import time
import random

# preload frequently used modules
import numpy as np
import numpy.random as npr
import scipy
import matplotlib
import matplotlib.pyplot as plt
from easydict import EasyDict as Edict

from src.omni_utils.log import LOG_INFO, LOG_ERROR, LOG_WARNING, LOG_DEBUG, LOG_CRITICAL

try:
    import torch
    TORCH_FOUND = True
except ImportError:
    TORCH_FOUND = False

EPS = sys.float_info.epsilon

def argparse(opts: Edict, varargin: None | Edict = None) -> Edict:
    if varargin is not None:
        for k in varargin:
            opts[k] = varargin[k]
    return opts

def random_seed(seed: int) -> None:
    np.random.seed(seed)
    if TORCH_FOUND:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

def random_index(n: int) -> np.ndarray:
    return (np.arange(n) + np.random.randint(n)) % n

def random_index_2x(n: int) -> List[np.ndarray, np.ndarray]:
    x = np.random.randint(n) #
    index1x = (np.arange(n) + x) % n
    index2x = (np.arange(2 * n) + 2 * x) % (2*n)
    return [index1x, index2x]

def isTorchArray(x):
    return TORCH_FOUND and type(x) == torch.Tensor

def rand(min_v, max_v=None, shape=None) -> float:
    if max_v is None:
        v_range = np.array(min_v)
        if v_range.size == 2:
            return rand(v_range[0], v_range[1], shape)
        r = random.random() if shape is None else npr.random(shape)
        return r * min_v
    else:
        r = random.random() if shape is None else npr.random(shape)
        return r * (max_v - min_v) + min_v