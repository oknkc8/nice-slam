# array_utils.py
#
# Author: Changhee Won (chwon@hanyang.ac.kr)
#
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import annotations
from typing import *

from src.omni_utils.common import *

def toNumpy(arr: torch.Tensor) -> np.ndarray:
    if isTorchArray(arr): arr = arr.cpu().numpy()
    return arr

def sqrt(x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    if isTorchArray(x): return torch.sqrt(x)
    else: return np.sqrt(x)

def cos(x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    if isTorchArray(x): return torch.cos(x)
    else: return np.cos(x)

def sin(x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    if isTorchArray(x): return torch.sin(x)
    else: return np.sin(x)

def tan(x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    if isTorchArray(x): return torch.tan(x)
    else: return np.tan(x)

def atan(x:torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    if isTorchArray(x): return torch.atan(x)
    else: return np.arctan(x)

def atan2(y:torch.Tensor | np.ndarray,
        x:torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    if isTorchArray(x): return torch.atan2(y, x)
    else: return np.arctan2(y, x)

def asin(x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    if isTorchArray(x): return torch.asin(x)
    else: return np.arcsin(x)

def acos(x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    if isTorchArray(x): return torch.acos(x)
    else: return np.arccos(x)

def exp(x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    if isTorchArray(x): return torch.exp(x)
    else: return np.exp(x)

def reshape(x: torch.Tensor | np.ndarray, shape: List[int]) \
        -> torch.Tensor | np.ndarray:
    if isTorchArray(x): return x.view(shape)
    else: return x.reshape(shape)

def concat(arr_list: List[torch.Tensor | np.ndarray], axis: int = 0):
    if isTorchArray(arr_list[0]): return torch.cat(arr_list, dim=axis)
    else: return np.concatenate(arr_list, axis=axis)

def polyval(P:torch.Tensor | np.ndarray, x:torch.Tensor | np.ndarray) \
        -> torch.Tensor | np.ndarray:
    if isTorchArray(x): P = torch.tensor(P).to(x.device)
    if isTorchArray(P):
        npol = P.shape[0]
        val = torch.zeros_like(x)
        for i in range(npol - 1):
            val = (val + P[i]) * x
        val += P[-1]
        return val
    else:
        return np.polyval(P, toNumpy(x))

def isnan(arr: torch.Tensor | np.ndarray):
    if isTorchArray(arr): return torch.isnan(arr)
    else: return np.isnan(arr)


def zeros_like(x: np.ndarray | torch.Tensor) -> torch.Tensor | np.ndarray:
    if isTorchArray(x): return torch.zeros_like(x)
    else: return np.zeros_like(x)

def ones_like(x: np.ndarray | torch.Tensor) -> torch.Tensor | np.ndarray:
    if isTorchArray(x): return torch.ones_like(x)
    else: return np.ones_like(x)

def logical_and(*arrs: List[torch.Tensor | np.ndarray]) \
        -> torch.Tensor | np.ndarray:
    if len(arrs) == 2:
        if isTorchArray(arrs[0]):
            return torch.logical_and(arrs[0], arrs[1])
        else:
            return np.logical_and(arrs[0], arrs[1])
    return logical_and(arrs[0], logical_and(*arrs[1:]))

def logical_or(*arrs: List[torch.Tensor | np.ndarray]) \
        -> torch.Tensor | np.ndarray:
    if len(arrs) == 2:
        if isTorchArray(arrs[0]):
            return torch.logical_or(arrs[0], arrs[1])
        else:
            return np.logical_or(arrs[0], arrs[1])
    return logical_or(arrs[0], logical_or(*arrs[1:]))

def logical_not(arr: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    if isTorchArray(arr): return torch.logical_not(arr)
    else: return np.logical_not(arr)

def hom(arr: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    h = ones_like(arr)
    return concat((arr, h), 0)