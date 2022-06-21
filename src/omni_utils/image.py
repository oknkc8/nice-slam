# utils.image
#
# Author: Changhee Won (changhee.1.won@gmail.com)
#
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import annotations
from typing import *

from src.omni_utils.common import *
from src.omni_utils.array import *

import tifffile
import skimage.io
import skimage.transform
from PIL import Image
if TORCH_FOUND:
    import torch.nn.functional as F

import cv2


## visualize =================================

def colorMap(colormap_name: str,
             arr: np.ndarray,
             min_v: float | None = None,
             max_v: float | None = None,
             alpha: float | None = None) -> np.ndarray:
    arr = toNumpy(arr).astype(np.float64).squeeze()
    if colormap_name == 'oliver': return colorMapOliver(arr, min_v, max_v)
    cmap = matplotlib.cm.get_cmap(colormap_name)
    if max_v is None: max_v = np.nanmax(arr)
    if min_v is None: min_v = np.nanmin(arr)
    arr[arr > max_v] = max_v
    arr[arr < min_v] = min_v
    arr = (arr - min_v) / (max_v - min_v)
    if alpha is None:
        out = cmap(arr)
        if len(out.shape) == 3:
            out = out[:, :, 0:3]
        elif len(out.shape) == 2:
            out = out[:, 0:3]
    else:
        alpha = min(max(alpha, 0), 1)
        out = cmap(arr, alpha=alpha)
    return np.round(255 * out).astype(np.uint8)

#
# code adapted from Oliver Woodford's sc.m
_CMAP_OLIVER = np.array(
    [[0,0,0,114], [0,0,1,185], [1,0,0,114], [1,0,1,174], [0,1,0,114],
     [0,1,1,185], [1,1,0,114], [1,1,1,0]]).astype(np.float64)
#
def colorMapOliver(arr: np.ndarray,
                   min_v: float | None = None,
                   max_v: float | None = None) -> np.ndarray:
    arr = toNumpy(arr).astype(np.float64).squeeze()
    height, width = arr.shape
    arr = arr.reshape([1, -1])
    if max_v is None: max_v = np.max(arr)
    if min_v is None: min_v = np.min(arr)
    arr[arr < min_v] = min_v
    arr[arr > max_v] = max_v
    arr = (arr - min_v) / (max_v - min_v)
    bins = _CMAP_OLIVER[:-1, 3]
    cbins = np.cumsum(bins)
    bins = bins / cbins[-1]
    cbins = cbins[:-1] / cbins[-1]
    ind = np.sum(
        np.tile(arr, [6, 1]) > \
        np.tile(np.reshape(cbins,[-1, 1]), [1, arr.size]), axis=0)
    ind[ind > 6] = 6
    bins = 1 / bins
    cbins = np.array([0.0] + cbins.tolist())
    arr = (arr - cbins[ind]) * bins[ind]
    arr = _CMAP_OLIVER[ind, :3] * np.tile(np.reshape(1 - arr,[-1, 1]),[1,3]) + \
        _CMAP_OLIVER[ind+1, :3] * np.tile(np.reshape(arr,[-1, 1]),[1,3])
    arr[arr < 0] = 0
    arr[arr > 1] = 1
    out = np.reshape(arr, [height, width, 3])
    out = np.round(255 * out).astype(np.uint8)
    return out

## image transform =================================

def rgb2gray(I: np.ndarray, channel_wise_mean=True) -> np.ndarray:
    I = toNumpy(I)
    dtype = I.dtype
    I = I.astype(np.float64)
    if channel_wise_mean:
        return np.mean(I, axis=2).squeeze().astype(dtype)
    else:
        return np.dot(I[...,:3], [0.299, 0.587, 0.114]).astype(dtype)

def imrescale(image: np.ndarray, scale: float) -> np.ndarray:
    image = toNumpy(image)
    dtype = image.dtype
    multi_channel = len(image.shape) == 3
    out = skimage.transform.rescale(image, scale,
        multichannel=multi_channel, preserve_range=True)
    return out.astype(dtype)

imresize = skimage.transform.resize

def interp2DNumpy(I: np.ndarray, grid: np.ndarray) -> np.ndarray:
    def __interpChannel(I, x0, x1, y0, y1, rx, ry, rx1, ry1, invalid):
        I = ry1 * (I[y0, x0] * rx1 + I[y0, x1] * rx) + \
            ry * (I[y1, x0] * rx1 + I[y1, x1] * rx)
        I[invalid] = 0
        return I

    org_dtype = I.dtype
    I = I.astype(np.float64).squeeze()
    multi_channels = len(I.shape) == 3
    if multi_channels:
        src_h, src_w = I[..., 0].shape
    else:
        src_h, src_w = I.shape

    xs = grid[..., 0]
    ys = grid[..., 1]
    target_h, target_w = xs.shape
    xs = (xs + 1) / 2.0 * (src_w - 1) # make -1 ~ 1 to 0 ~ w-1
    ys = (ys + 1) / 2.0 * (src_h - 1) # make -1 ~ 1 to 0 ~ h-1

    invalid = logical_or(xs < 0, xs >= src_w, ys < 0, ys >= src_h,
        isnan(xs), isnan(ys))

    xs[invalid] = 0
    ys[invalid] = 0

    x0 = np.floor(xs).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, src_w - 1)
    y0 = np.floor(ys).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, src_h - 1)

    rx = xs - x0
    ry = ys - y0
    rx1 = 1 - rx
    ry1 = 1 - ry

    if multi_channels:
        chs = [
            __interpChannel(
                I[..., n], x0, x1, y0, y1,
                rx, ry, rx1, ry1, invalid)[..., np.newaxis] \
            for n in range(I.shape[2])]
        return concat(chs, 2)
    else:
        return __interpChannel(I, x0, x1, y0, y1, rx, ry, rx1, ry1, invalid)

 
def interp2D(I: torch.Tensor | np.ndarray, grid: torch.Tensor | np.ndarray, flag) \
        -> torch.Tensor | np.ndarray:
    if not TORCH_FOUND:
        return interp2DNumpy(I, grid)
    istensor = isTorchArray(I)
    I = torch.tensor(I).float().squeeze().unsqueeze(0) # make 1 x C x H x W

    if flag == 0: #RGBimages
        is_flipped = I.shape[3] == 3 and I.shape[1] != 3
        if is_flipped: I = I.permute((0, 3, 1, 2))
    elif flag == 1:
        is_flipped = I.shape[2] == 1 and I.shape[0] != 1
        if is_flipped: I = I.transpose((2, 0))

    if len(I.shape) < 4 : # if 1D channel image
        I = I.unsqueeze(0)
    grid = torch.tensor(
        grid).squeeze().unsqueeze(0).float() # make 1 x npts x 2
    if flag == 0: #RGBimages
        out = F.grid_sample(
            I, grid, mode='bilinear', align_corners=True).squeeze()
    elif flag == 1:
        out = F.grid_sample(
            I, grid, mode='nearest', align_corners=True).squeeze()

    if is_flipped: out = out.permute((1, 2, 0))
    if not istensor: out = out.numpy()
    return out

def pixelToGrid(pts: torch.Tensor | np.ndarray,
                target_resolution: Tuple[int, int],
                source_resolution: Tuple[int, int]) \
                    -> torch.Tensor | np.ndarray:
    h, w = target_resolution
    height, width = source_resolution
    xs = (pts[0,:]) / (width - 1) * 2 - 1
    ys = (pts[1,:]) / (height - 1) * 2 - 1
    xs = xs.reshape((h, w, 1))
    ys = ys.reshape((h, w, 1))
    return concat((xs, ys), 2)

def normalizeImage(image: np.ndarray,
                   mask: np.ndarray | None = None) -> np.ndarray:
    image = toNumpy(image)
    mask = toNumpy(mask) > 0
    def __normalizeImage1D(image, mask):
        image = image.squeeze().astype(np.float32)
        if mask is not None: image[mask] = np.nan
        # normalize intensities
        image = (image - np.nanmean(image.flatten())) / \
            np.nanstd(image.flatten())
        if mask is not None: image[mask] = 0
        return image
    if len(image.shape) == 3 and image.shape[2] == 3:
        # return np.concatenate(
        #     [__normalizeImage1D(image[:,:,i], mask)[..., np.newaxis]
        #         for i in range(3)], axis=2)
        image = image.squeeze().astype(np.float32)
        mask = np.tile(mask[..., np.newaxis], (1, 1, 3))
        if mask is not None: image[mask] = np.nan
        # normalize intensities
        image = (image - np.nanmean(image.flatten())) / \
            np.nanstd(image.flatten())
        if mask is not None: image[mask] = 0
        return image
    else:
        return __normalizeImage1D(image, mask)

## image file I/O =================================

def writeImageFloat(image: np.ndarray, tiff_path: str,
                    thumbnail: np.ndarray | None = None) -> None:
    image = toNumpy(image)
    with tifffile.TiffWriter(tiff_path) as tiff:
        if thumbnail is not None:
            if not thumbnail.dtype == np.uint8:
                thumbnail = thumbnail.astype(np.uint8)
            tiff.save(thumbnail, photometric='RGB',
                bitspersample=8)
        if not image.dtype == np.float32:
            image = image.astype(np.float32)
        tiff.save(image, photometric='MINISBLACK',
                bitspersample=32, compress=9)

def readImageFloat(tiff_path: str, return_thumbnail = False,
                   read_or_die = True) \
                   -> np.ndrray | List[np.ndarray] | \
                      Tuple[np.ndarray, None | np.ndarray] | None:
    try:
        # multi_image = skimage.io.MultiImage(tiff_path)
        multi_image = tifffile.TiffFile(tiff_path)
        num_read_images = len(multi_image.pages)
        if num_read_images == 0:
            raise Exception('No images found.')
        elif num_read_images == 1:
            return multi_image.pages[0].asarray().squeeze()
        elif num_read_images == 2: # returns float, thumnail
            multi_image = [x.asarray().squeeze() for x in multi_image.pages]
            if multi_image[0].dtype == np.uint8:
                if not return_thumbnail: return multi_image[1].squeeze()
                else: return multi_image[1].squeeze(), multi_image[0].squeeze()
            else:
                if not return_thumbnail: return multi_image[0].squeeze()
                else: return multi_image[0].squeeze(), multi_image[1].squeeze()
        else: # returns list of images
            return [im.squeeze() for im in multi_image]
    except Exception as e:
        LOG_ERROR('Failed to read image float: "%s"' %(e))
        if read_or_die:
            traceback.print_tb(e.__traceback__)
            sys.exit()
        return None

def writeImage(image: np.ndarray, path: str) -> None:
    image = skimage.img_as_ubyte(toNumpy(image))
    image = Image.fromarray(image)
    # skimage.io.imsave(path, image, check_contrast=False)
    image.save(path)

def readImage(path: str, read_or_die = True) -> np.ndarray | None:
    try:
        return skimage.io.imread(path)
    except Exception as e:
        LOG_ERROR('Failed to read image: "%s"' % (e))
        if read_or_die:
            traceback.print_tb(e.__traceback__)
            sys.exit()
        return None