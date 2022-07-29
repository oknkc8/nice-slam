# utils.geometry.py
#
# Author: Changhee Won (chwon@hanyang.ac.kr)
#
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import annotations
from typing import *

from numpy import dtype

from src.omni_utils.common import *
from scipy.spatial.transform import Rotation as R

import pdb

def rodrigues(r: np.ndarray) -> np.ndarray:
    if r.size == 3: return R.from_rotvec(r.squeeze()).as_matrix()
    else: return R.from_matrix(r).as_rotvec().reshape((3, 1))

def getRot(transform: np.ndarray) -> np.ndarray:
    if transform.size == 6:
        transform = transform.reshape((6, 1))
        return rodrigues(transform[:3])
    elif transform.shape == (3, 4) or transform.shape == (4, 4):
        return transform[:3, :3]
    else:
        LOG_ERROR(
            'Invalid shape of input transform: {}'.format(transform.shape))
        return None

def getTr(transform: np.ndarray) -> np.ndarray:
    if transform.size == 6:
        transform = transform.reshape((6, 1))
        return transform[3:6].reshape((3, 1))
    elif transform.shape == (3, 4) or transform.shape == (4, 4):
        return transform[:3, 3].reshape((3, 1))
    else:
        LOG_ERROR(
            'Invalid shape of input transform: {}'.format(transform.shape))
        return None
    
def rotateAxis(t: np.ndarray, deg_x: float, deg_y: float, deg_z: float) -> np.ndarray:
    # import pdb
    # pdb.set_trace()
    R1 = R.from_euler('xyz', [deg_x, deg_y, deg_z], degrees=True).as_matrix()
    R2, tr2 = getRot(t), getTr(t)
    new_R = np.matmul(R1, R2)
    # new_tr = R1.dot(tr2)
    new_tr = tr2
    return np.concatenate((new_R, new_tr), axis=1)

def inverseTransform(transform: np.ndarray) -> np.ndarray:
    R, tr = getRot(transform), getTr(transform)
    R_inv = R.transpose()
    tr_inv = -R_inv.dot(tr)
    if transform.size == 6:
        r_inv = rodrigues(R_inv)
        return np.concatenate((r_inv, tr_inv), axis=0) # (6, 1) vector
    else:
        return np.concatenate((R_inv, tr_inv), axis=1) # (3, 4) matrix

def mergedTransform(t2: np.ndarray, t1: np.ndarray) -> np.ndarray: # T2 * T1
    R1, tr1 = getRot(t1), getTr(t1)
    R2, tr2 = getRot(t2), getTr(t2)
    R = np.matmul(R2, R1)
    tr = R2.dot(tr1) + tr2
    if t1.size == 6:
        rot = rodrigues(R)
        return np.concatenate((rot, tr), axis=0)
    else:
        return np.concatenate((R, tr), axis=1)

def applyTransform(transform: np.ndarray, P: torch.Tensor | np.ndarray) \
        -> torch.Tensor | np.ndarray:
    R, tr = getRot(transform), getTr(transform)
    if isTorchArray(P):
        R = torch.tensor(R, dtype=P.dtype).to(P.device)
        tr = torch.tensor(tr, dtype=P.dtype).to(P.device)
        return torch.matmul(R, P) + tr
    else:
        return R.dot(P) + tr