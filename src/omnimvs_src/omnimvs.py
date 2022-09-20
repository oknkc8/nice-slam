# omnimvs.py
#
# Author: Changhee Won (changhee.1.won@gmail.com)
#
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import annotations
from tkinter import Variable
from turtle import pos
from typing import *
import os.path as osp
import cv2

import numpy as np
import scipy.ndimage
from easydict import EasyDict as Edict
import torch
import torch.utils
from torch.autograd import Variable

import sys
from src.omni_utils.common import *
from src.omni_utils.array import *
from src.omni_utils.geometry import *
from src.omni_utils.log import *
from src.omni_utils.camera import *
from src.omni_utils.image import *
from src.omni_utils.sensor_data import SensorDataReader, SensorDataType
import src.omni_utils.dbhelper

import pdb
# torch.autograd.set_detect_anomaly(True)

def getEquirectCoordinate(
        pts: np.ndarray, equirect_size: Tuple[int, int],
        phi_deg: float, phi_max_deg: float = -1.0) -> np.ndarray:
    h, w = equirect_size
    d = sqrt((pts**2).sum(0)).reshape((1, -1))
    nx = pts[0,:] / d
    ny = pts[1,:] / d
    nz = pts[2,:] / d
    phi = asin(ny)
    sign = cos(phi); sign[sign < 0] = -1.0; sign[sign >= 0] = 1.0
    theta = atan2(nz * sign, -nx * sign)
    equi_x = ((theta - np.pi / 2) / np.pi + 1) * w / 2
    equi_x[equi_x < 0] += w
    equi_x[equi_x >= w] -= w
    if (phi_max_deg < 0):
        equi_y = (phi / np.deg2rad(phi_deg) + 1) * h / 2
    else:
        med = np.deg2rad((phi_max_deg - phi_deg) / 2)
        med2 = np.deg2rad((phi_max_deg + phi_deg) / 2)
        equi_y = ((phi + med2) / med + 1) * h / 2
    return concat([equi_x, equi_y], axis=0)

def makeSphericalRays(equirect_size: Tuple[int, int],
                      phi_deg: float, phi_max_deg: float = -1.0,
                      out_thetas = False) -> np.ndarray:
    h, w = equirect_size
    xs, ys = np.meshgrid(range(w), range(h)) # row major
    w_2, h_2 = w / 2.0, (h - 1) / 2.0
    xs = (xs - w_2) / w_2 * np.pi + (np.pi / 2.0)
    if phi_max_deg > 0.0:
        med = np.deg2rad((phi_max_deg - phi_deg) / 2.0)
        med2 = np.deg2rad((phi_max_deg + phi_deg) / 2.0)
        ys = (ys - h_2) / h_2 * med - med2
    else:
        ys = (ys - h_2) / h_2 * np.deg2rad(phi_deg)

    X = -np.cos(ys) * np.cos(xs)
    Y = np.sin(ys) # sphere
    # Y = np.sin(ys) / np.cos(ys) # cylinder
    # Y = ys / np.deg2rad(phi_deg) # perspective cylinder
    Z = np.cos(ys) * np.sin(xs)
    rays = np.concatenate((np.reshape(X, [1, -1]),
        np.reshape(Y, [1,-1]), np.reshape(Z, [1,-1]))).astype(np.float64)
    if out_thetas:
        return rays, xs, ys
    else:
        return rays

def buildSphereLookupTable(
        equirect_size: Tuple[int, int],
        num_invdepth: int, min_invdepth: float,
        step_invdepth: float | List[float],
        phi_deg: float, phi_max_deg: float, ocams: List[OcamModel],
        output_gpu_tensor = False, transform: np.ndarray | None = None,
        device_id: int = 0) \
            -> List[np.ndarray | torch.Tensor]:
    h, w = equirect_size
    num_cams = len(ocams)
    rays = makeSphericalRays(equirect_size, phi_deg, phi_max_deg)
    if output_gpu_tensor:
        grids = [torch.zeros((num_invdepth, h, w, 2),
                    requires_grad=False, dtype=torch.float32).to(device_id) \
                    for _ in range(num_cams)]
    else:
        grids = [np.zeros((num_invdepth, h, w, 2), dtype=np.float32)
            for _ in range(num_cams)]
    for d in range(num_invdepth):
        if d == 0:
            depth = 1 / min_invdepth
        elif type(step_invdepth) == list or type(step_invdepth) == np.ndarray:
            depth = 1.0 / (min_invdepth + step_invdepth[d - 1])
        else:
            depth = 1.0 / (min_invdepth + d * step_invdepth)
        pts = depth * rays
        if output_gpu_tensor:
            pts = torch.tensor(pts.astype(np.float32),
                requires_grad=False).to(device_id)

        if transform is not None:
            pts = applyTransform(transform, pts)

        for i in range(num_cams):
            P = applyTransform(ocams[i].rig2cam, pts)
            p = ocams[i].rayToPixel(P)
            p[p < 0] = -1e5
            grid = pixelToGrid(p, equirect_size,
                (ocams[i].height, ocams[i].width))
            if output_gpu_tensor: grids[i][d, ...] = grid
            else: grids[i][d, ...] = grid.astype(np.float32)
    for i in range(num_cams):
        grids[i] = grids[i].unsqueeze(0)
    return grids

def buildIndividualLookupTable(
        ocams: List[CameraModel],
        num_invdepth: int, min_invdepth: float,
        step_invdepth: Union[List[float], float],
        omnimvs: 'OmniMVS', output_gpu_tensor: bool = True, device_id: int = 0):
    if output_gpu_tensor:
        grids = [torch.zeros((num_invdepth, cam.height, cam.width, 3),
                    requires_grad=False, dtype=torch.float32).to(device_id) \
                    for cam in ocams]
    else:
        grids = [np.zeros((num_invdepth, cam.height, cam.width, 3),
            dtype=np.float32) for cam in ocams]

    for i, cam in enumerate(ocams):
        rays = cam.pixelToRay(cam.getPixelGrid())
        for d in range(num_invdepth):
            if d == 0:
                depth = 1 / min_invdepth
            elif type(step_invdepth) == list or \
                    type(step_invdepth) == np.ndarray:
                depth = 1.0 / (min_invdepth + step_invdepth[d - 1])
            else:
                depth = 1.0 / (min_invdepth + d * step_invdepth)
            pts = depth * rays

            if output_gpu_tensor:
                pts = torch.tensor(pts.astype(np.float32),
                    requires_grad=False).to(device_id)
            pts_rig = applyTransform(cam.cam2rig, pts)
            invdepth_rig = 1.0 / sqrt(torch.sum(pts_rig**2, 0))
            
            equi_pix  = getEquirectCoordinate(
                pts_rig, omnimvs.equirect_size, omnimvs.phi_deg, omnimvs.phi_max_deg) 
            equi_didx = omnimvs.invdepthToIndex(invdepth_rig)

            equi_didx[isnan(equi_didx)] = -1
            equi_didx = torch.clamp_max(equi_didx, num_invdepth - 1)
            # consider circular interloation from equirectangular image
            equi_grid_hw = pixelToGrid(
                equi_pix, cam.image_size,
                (omnimvs.equirect_size[0], omnimvs.equirect_size[1] + 1))
            equi_grid_d = (equi_didx / (num_invdepth - 1)) * 2 - 1
            equi_grid_d = equi_grid_d.reshape((cam.height, cam.width, 1))
            equi_grid = concat((equi_grid_hw, equi_grid_d), 2)
            # equi_grid = concat((equi_grid_d, equi_grid_hw), 2)
            if output_gpu_tensor:
                grids[i][d, ...] = equi_grid
            else:
                grids[i][d, ...] = equi_grid.astype(np.float32)
        grids[i] = grids[i].unsqueeze(0)
    return grids

def inverseCuboidRadiusToDepth(
        invradii: np.ndarray,
        cuboid_size: Tuple[float, float, float],
        thetas: np.ndarray,
        phis: np.ndarray) -> np.ndarray:
    if invradii.size != thetas.size or invradii.size != phis.size:
        LOG_ERROR('size of invdepth is not equal to thetas or phis')
        sys.exit(-1)
    invradii = invradii.reshape(thetas.shape)
    l, h, w = cuboid_size
    w2, h2, l2 = w*w, h*h, l*l
    abs_cos_theta, abs_sin_theta = abs(cos(thetas)), abs(sin(thetas))
    abs_cos_phi, abs_sin_phi = abs(cos(phis)), abs(sin(phis))
    abs_tan_theta, abs_tan_phi = abs(tan(thetas)), abs(tan(phis))
    is_right = abs_tan_theta <= (l / w)
    is_front = logical_not(is_right)

    radii = 1 / invradii
    r2 = radii**2
    lb_tan_xz = l / (w + radii)
    ub_tan_xz = (l + radii) / w
    is_round_xz = logical_and(
        abs_tan_theta >= lb_tan_xz, abs_tan_theta <= ub_tan_xz)
    is_front_xz = abs_tan_theta > ub_tan_xz
    is_right_xz = abs_tan_theta < lb_tan_xz

    d_xz = zeros_like(thetas)
    d_xz[is_front_xz] = (l + radii[is_front_xz]) / \
        (abs_sin_theta[is_front_xz] + EPS)
    d_xz[is_right_xz] = (w + radii[is_right_xz]) / \
        (abs_cos_theta[is_right_xz] + EPS)

    w_cx_l_sx = w * abs_cos_theta[is_round_xz] + \
                l * abs_sin_theta[is_round_xz]
    d_xz[is_round_xz] = w_cx_l_sx + sqrt(
        (w_cx_l_sx)**2 - w2 - l2 + r2[is_round_xz])

    lb_tan_y = h / d_xz
    ub_tan_y_front = (h + radii) / (l / (abs_sin_theta + EPS))
    ub_tan_y_right = (h + radii) / (w / (abs_cos_theta + EPS))

    is_round_y = logical_and(abs_tan_phi > lb_tan_y,
        logical_or(
            logical_and(is_right, abs_tan_phi < ub_tan_y_right),
            logical_and(is_front, abs_tan_phi < ub_tan_y_front)))
    is_side_y = abs_tan_phi <= lb_tan_y
    is_up_y = logical_and(logical_not(is_round_y), logical_not(is_side_y))

    d = np.zeros_like(thetas)
    d[is_side_y] = d_xz[is_side_y] / abs_cos_phi[is_side_y]
    d[is_up_y] = (h + radii[is_up_y]) / abs_sin_phi[is_up_y]

    is_round_y_right_xz = logical_and(is_round_y, is_right)
    is_round_y_front_xz = logical_and(is_round_y, is_front)

    cx = abs_cos_theta[is_round_y_right_xz]
    sx = abs_sin_theta[is_round_y_right_xz]
    cy = abs_cos_phi[is_round_y_right_xz]
    sy = abs_sin_phi[is_round_y_right_xz]

    # takes root
    a = cy**2 * cx**2 + sy**2
    b = w * cy * cx + h * sy
    c = w2 + h2 - r2[is_round_y_right_xz]
    d_tmp = (b + sqrt(b**2 - (a * c))) / a
    # set invalid
    valid_mask = (cy * sx) < l / (d_tmp + EPS)
    is_round_y_right_xz[is_round_y_right_xz] = valid_mask
    d[is_round_y_right_xz] = d_tmp[valid_mask]

    cx = abs_cos_theta[is_round_y_front_xz]
    sx = abs_sin_theta[is_round_y_front_xz]
    cy = abs_cos_phi[is_round_y_front_xz]
    sy = abs_sin_phi[is_round_y_front_xz]

    # takes root
    a = cy**2 * sx**2 + sy**2
    b = l * cy * sx + h * sy
    c = l2 + h2 - r2[is_round_y_front_xz]
    d_tmp = (b + sqrt(b**2 - (a * c))) / a
    # set invalid
    valid_mask = (cy * cx) < w / d_tmp
    is_round_y_front_xz[is_round_y_front_xz] = valid_mask
    d[is_round_y_front_xz] = d_tmp[valid_mask]

    is_round_y_corner = logical_and(is_round_y,
        logical_not(is_round_y_front_xz), logical_not(is_round_y_right_xz))

    cx = abs_cos_theta[is_round_y_corner]
    sx = abs_sin_theta[is_round_y_corner]
    cy = abs_cos_phi[is_round_y_corner]
    sy = abs_sin_phi[is_round_y_corner]

    # takes root
    b = h * sy + w * cx * cy + l * cy * sx
    c = h2 + w2 + l2 - r2[is_round_y_corner]
    d[is_round_y_corner] = b + sqrt(b**2 - c)
    return 1 / d

def inverseDepthToCuboidRadius(
        invdepth : np.ndarray | torch.Tensor,
        cuboid_size : Tuple[float, float, float],
        thetas: np.ndarray,
        phis: np.ndarray) -> np.ndarray | torch.Tensor:
    if invdepth.size != thetas.size or invdepth.size != phis.size:
        LOG_ERROR('size of invdepth is not equal to thetas or phis')
        sys.exit(-1)
    invdepth = invdepth.reshape(thetas.shape)
    cos_thetas, sin_thetas = cos(thetas), sin(thetas)
    cos_phis, sin_phis = cos(phis), sin(phis)
    xs = abs(cos_phis * cos_thetas / invdepth).reshape((1, -1))
    ys = abs(sin_phis / invdepth).reshape((1, -1))
    zs = abs(cos_phis * sin_thetas / invdepth).reshape((1, -1))
    pts = concat((xs, ys, zs), 0)
    to_center = zeros_like(pts)
    l, h, w = cuboid_size
    pts[0, (xs < w).flatten()] = EPS
    pts[1, (ys < h).flatten()] = EPS
    pts[2, (zs < l).flatten()] = EPS
    to_center[0, (xs > w).flatten()] = w
    to_center[1, (ys > h).flatten()] = h
    to_center[2, (zs > l).flatten()] = l
    radius = sqrt(((pts - to_center)**2).sum(0)).reshape(invdepth.shape)
    return 1 / radius

def buildRoundedCuboidLookUpTable(
        cuboid_size : Tuple[float, float, float], # (length, height, width)
        equirect_size : Tuple[int, int],
        num_invradii : int,
        min_invradius: float, step_invradius: float | List[float],
        phi_deg: float, phi_max_deg: float,
        ocams: List[OcamModel],
        transform: np.ndarray | None = None, output_gpu_tensor=True,
        device_id: int = 0):
    equirect_h, equirect_w = equirect_size
    num_cams = len(ocams)
    if output_gpu_tensor:
        grids = [
            torch.zeros((num_invradii, equirect_h, equirect_w, 2),
                requires_grad=False, dtype=torch.float32).to(device_id) \
                    for _ in range(num_cams)]
    else:
        grids = [
            np.zeros((num_invradii, equirect_h, equirect_w, 2),
                dtype=np.float32) for _ in range(num_cams)]

    thetas, phis = np.meshgrid(range(equirect_w), range(equirect_h))
    equi_w_2, equi_h_2 = equirect_w / 2.0, (equirect_h - 1) / 2.0
    thetas = (thetas - equi_w_2) / equi_w_2 * np.pi + (np.pi / 2.0)
    if phi_max_deg > 0.0:
        med = np.deg2rad((phi_max_deg - phi_deg) / 2.0)
        med2 = np.deg2rad((phi_max_deg + phi_deg) / 2.0)
        phis = (phis - equi_h_2) / equi_h_2 * med - med2
    else:
        phis = (phis - equi_h_2) / equi_h_2 * np.deg2rad(phi_deg)

    X = -np.cos(phis) * np.cos(thetas)
    Y = np.sin(phis) # sphere
    Z = np.cos(phis) * np.sin(thetas)
    rays = np.concatenate((np.reshape(X, [1, -1]),
        np.reshape(Y, [1,-1]), np.reshape(Z, [1,-1])))

    l, h, w = cuboid_size
    w2, h2, l2 = w*w, h*h, l*l
    abs_cos_theta, abs_sin_theta = abs(cos(thetas)), abs(sin(thetas))
    abs_cos_phi, abs_sin_phi = abs(cos(phis)), abs(sin(phis))
    abs_tan_theta, abs_tan_phi = abs(tan(thetas)), abs(tan(phis))
    is_right = abs_tan_theta <= (l / w)
    is_front = logical_not(is_right)

    for n in range(num_invradii):
        if n == 0:
            r = 1 / min_invradius
        elif type(step_invradius) == np.ndarray or type(step_invradius) == list:
            r = 1.0 / (min_invradius + step_invradius[n - 1])
        else:
            r = 1.0 / (min_invradius + n * step_invradius)
        r2 = r*r
        lb_tan_xz = l / (w + r)
        ub_tan_xz = (l + r) / w
        is_round_xz = logical_and(
            abs_tan_theta >= lb_tan_xz, abs_tan_theta <= ub_tan_xz)
        is_front_xz = abs_tan_theta > ub_tan_xz
        is_right_xz = abs_tan_theta < lb_tan_xz

        d_xz = np.zeros_like(thetas)
        d_xz[is_front_xz] = (l + r) / (abs_sin_theta[is_front_xz] + EPS)
        d_xz[is_right_xz] = (w + r) / (abs_cos_theta[is_right_xz] + EPS)

        w_cx_l_sx = w * abs_cos_theta[is_round_xz] + \
                    l * abs_sin_theta[is_round_xz]
        d_xz[is_round_xz] = w_cx_l_sx + sqrt((w_cx_l_sx)**2 - w2 - l2 + r2)

        lb_tan_y = h / d_xz
        ub_tan_y_front = (h + r) / (l / (abs_sin_theta + EPS))
        ub_tan_y_right = (h + r) / (w / (abs_cos_theta + EPS))

        is_round_y = logical_and(abs_tan_phi > lb_tan_y,
            logical_or(
                logical_and(is_right, abs_tan_phi < ub_tan_y_right),
                logical_and(is_front, abs_tan_phi < ub_tan_y_front)))
        is_side_y = abs_tan_phi <= lb_tan_y
        is_up_y = logical_and(logical_not(is_round_y), logical_not(is_side_y))

        d = np.zeros_like(thetas)
        d[is_side_y] = d_xz[is_side_y] / (abs_cos_phi[is_side_y] + EPS)
        d[is_up_y] = (h + r) / (abs_sin_phi[is_up_y] + EPS)

        is_round_y_right_xz = logical_and(is_round_y, is_right)
        is_round_y_front_xz = logical_and(is_round_y, is_front)

        cx = abs_cos_theta[is_round_y_right_xz]
        sx = abs_sin_theta[is_round_y_right_xz]
        cy = abs_cos_phi[is_round_y_right_xz]
        sy = abs_sin_phi[is_round_y_right_xz]
        # takes root
        a = cy**2 * cx**2 + sy**2
        b = w * cy * cx + h * sy
        c = w2 + h2 - r2
        d_tmp = (b + sqrt(b**2 - (a * c))) / a
        # set invalid
        valid_mask = (cy * sx) < l / d_tmp
        is_round_y_right_xz[is_round_y_right_xz] = valid_mask
        d[is_round_y_right_xz] = d_tmp[valid_mask]

        cx = abs_cos_theta[is_round_y_front_xz]
        sx = abs_sin_theta[is_round_y_front_xz]
        cy = abs_cos_phi[is_round_y_front_xz]
        sy = abs_sin_phi[is_round_y_front_xz]
        # takes root
        a = cy**2 * sx**2 + sy**2
        b = l * cy * sx + h * sy
        c = l2 + h2 - r2
        d_tmp = (b + sqrt(b**2 - (a * c))) / a
        # set invalid
        valid_mask = (cy * cx) < w / d_tmp
        is_round_y_front_xz[is_round_y_front_xz] = valid_mask
        d[is_round_y_front_xz] = d_tmp[valid_mask]

        is_round_y_corner = logical_and(is_round_y,
            logical_not(is_round_y_front_xz), logical_not(is_round_y_right_xz))

        cx = abs_cos_theta[is_round_y_corner]
        sx = abs_sin_theta[is_round_y_corner]
        cy = abs_cos_phi[is_round_y_corner]
        sy = abs_sin_phi[is_round_y_corner]
        # takes root
        b = h * sy + w * cx * cy + l * cy * sx
        c = h2 + w2 + l2 - r2
        d[is_round_y_corner] = b + sqrt(b**2 - c)

        pts = d.reshape((1, -1)) * rays
        if output_gpu_tensor:
            pts = torch.tensor(pts.astype(np.float32),
                requires_grad=False).to(device_id)
        if transform is not None:
            pts = applyTransform(transform, pts)
        for i in range(num_cams):
            P = applyTransform(ocams[i].rig2cam, pts)
            p = ocams[i].rayToPixel(P)
            p[p < 0] = -1e5
            grid = pixelToGrid(p, equirect_size,
                (ocams[i].height, ocams[i].width))
            if output_gpu_tensor: grids[i][n, ...] = grid
            else: grids[i][n, ...] = grid.astype(np.float32)
    for i in range(num_cams):
        grids[i] = grids[i].unsqueeze(0)
    return grids

def getValidGridMask(grids: List[torch.Tensor]) -> torch.Tensor:
    h, w = grids[0].shape[-3], grids[0].shape[-2]
    grid_masks = torch.zeros((h, w)).bool().to(grids[0].device)
    for grid in grids:
        # 1 N H W 2
        visible = grid[...,0] >= -1 # 1 N H W
        mask = visible.sum((0,1)).bool() # H W
        grid_masks = torch.logical_or(grid_masks, mask)
    return grid_masks

def makeInterleavingGrids(
        w1: float, w2: float,
        grids1: torch.Tensor, grids2: torch.Tensor) -> \
            List[torch.Tensor]:
    w1 = float(w1)
    w2 = float(w2)
    w = w1 + w2

    grid = -1e5 * torch.ones_like(grids1)
    valid_fov1 = grids1 >= -1
    valid_fov2 = grids2 >= -1

    overlap = torch.logical_and(valid_fov1, valid_fov2)
    not_overlap = torch.logical_not(overlap)
    valid_fov1 = torch.logical_and(valid_fov1, not_overlap)
    valid_fov2 = torch.logical_and(valid_fov2, not_overlap)
    dist_from_center1 = torch.sum(grids1**2, -1)
    dist_from_center2 = torch.sum(grids2**2, -1)
    inside_fov1 = (
        dist_from_center1 <= dist_from_center2).unsqueeze(-1).expand(
            -1,-1,-1,-1,2)
    inside_fov2 = torch.logical_not(inside_fov1)
    valid_fov1 = torch.logical_or(valid_fov1,
        torch.logical_and(overlap, inside_fov1))
    valid_fov2 = torch.logical_or(valid_fov2,
        torch.logical_and(overlap, inside_fov2))

    grid[valid_fov1] = grids1[valid_fov1]
    grid[valid_fov2] = grids2[valid_fov2]

    # grid1 ranges from -1 to 1 (0 ~ w1 - 1)
    # interleaved grid should range from -1 to 1 (0 ~ (w1 + w2 - 1))
    valid_fov1[...,1] = False
    valid_fov2[...,1] = False
    grid[valid_fov1] = ((grid[valid_fov1] + 1) / 2 * (w1 - 1))
    grid[valid_fov2] = ((grid[valid_fov2] + 1) / 2 * (w2 - 1) + w1)
    grid[...,0] = grid[...,0] / (w - 1) * 2 - 1
    return grid


class OmniMVS(torch.utils.data.Dataset):
    def __init__(self,
            dbname: str | List[str],
            db_root: str = '/data/multipleye', db_path: None | str = None,
            opts: None | Edict = None, build_lt = True, train = False,
            db_config_idxs: List[int] | None = None,
            device_id: int = 0):
        super(torch.utils.data.Dataset, self).__init__()
        self.load_multiple_datasets = type(dbname) == list
        if self.load_multiple_datasets:
            self.dbnames = dbname
            dbname = dbname[0]
        if db_path is None:
            db_path = osp.join(db_root, dbname)
        if opts is None:
            opts = Edict()
        self.dbname = dbname
        self.db_root = db_root
        self.db_path = db_path
        self.build_lt = build_lt
        self.train = train
        self.device_id = device_id

        # define default arguments
        _opts = Edict()
        #_opts.img_fmt = 'cam%d/%05d.png' # [cam_idx, fidx]
        _opts.img_fmt = 'cam%d_square_832/%05d.png' # [cam_idx, fidx]
        _opts.gt_depth_fmt = 'cam_center_square_%d_depth/%05d.tiff' # [equi_w, fidx]
        _opts.gt_img_fmt = 'cam_center_square_%d/%05d.png' #[equi_w, fidx]
        _opts.read_input_image = True # for evaluation, False if read only GT
        _opts.start, _opts.step, _opts.end = 0, 1, 4061 # frame read indices
        _opts.use_frame_idxs_for_test = False
        _opts.train_idx, _opts.test_idx = [], []
        _opts.gt_phi = 0.0
        _opts.dtype = 'nogt' #default 'nogt'
        _opts.remove_gt_noise = True
        _opts.morph_win_size = 5
        _opts.input_size = None
        _opts.use_rgb = True
        _opts.read_input_image = True

        _opts.equirect_size = (160, 640)
        _opts.num_invdepth = 192
        _opts.phi_deg, _opts.phi_max_deg = 45, -1.0
        _opts.read_input_image = True # for evaluation, False if read only GT
        _opts.use_rgb = True
        _opts.lt_equirect_sample_ratio = 1
        _opts.lt_depth_sample_ratio = 2
        # _opts.out_cost = False

        _opts.use_rounded_cuboid_sweep = False
        _opts.use_mid_depth_sweep = False
        _opts.camera_idxs = [1, 2, 3, 4] # List[int]
        _opts.interleave_idxs = [] # List[Tuple[int, int]]
        _opts.for_car = False
        _opts.limit_fov_deg = -1.0

        # first update opts using pre-defined config
        # also load ocam parameters
        _opts, self.ocams = src.omni_utils.dbhelper.loadDBConfigs(
            self.dbname, self.db_path, _opts)

        self.baseline = np.mean(
            [sqrt(sum(getTr(ocam.cam2rig)**2)) for ocam in self.ocams])
        # _opts.min_depth = self.baseline / tan(
        #     self.ocams[0].max_theta - (np.pi / 2))
        if not 'min_depth' in _opts and not 'min_depth' in opts:
            _opts.min_depth = self.baseline * 1.5
            _opts.mid_depth = self.baseline * 10

        cam2rigs = concat(
                [ocam.cam2rig for ocam in self.ocams], 1)
        _opts.cuboid_size = (
            np.max(cam2rigs[5,:]),
            self.baseline / sqrt(2),
            np.max(cam2rigs[3,:]))

        # these arguments must follow DB's config
        img_fmt = _opts.img_fmt
        gt_depth_fmt = _opts.gt_depth_fmt
        gt_phi = _opts.gt_phi
        dtype = _opts.dtype
        train_idx = _opts.train_idx
        test_idx = _opts.test_idx

        if 'min_depth' in _opts:
            min_depth = _opts.min_depth
            mid_depth = _opts.mid_depth

        # update opts from the argument
        opts = argparse(_opts, opts)

        # these arguments must follow config files
        opts.img_fmt = img_fmt
        opts.gt_depth_fmt = gt_depth_fmt
        opts.gt_phi = gt_phi
        opts.dtype = dtype
        opts.train_idx = train_idx
        opts.test_idx = test_idx

        try:
            opts.min_depth = min_depth
            opts.mid_depth = mid_depth
        except: pass

        self.opts = opts

        ##added
        if self.dbname == '201029_coex_s9':
            opts.start = 0
            opts.end = 4061
            # opts.step = 1
            opts.train_idx = list(range(0, 4061))
            opts.test_idx = list(range(0, 4061))
            #opts.train_idx = list(range(0, 3928)) #3928
            #opts.test_idx = list(range(9, 3928, 10))
            #opts.train_idx = list(set(opts.train_idx) - set(opts.test_idx))
            opts.min_depth = 0.3
            #opts.dtype = 'gt'
            opts.gt_phi = 45
        elif self.dbname == '210624_xingxing_block6':
            opts.start = 0
            opts.end = 25985
            # opts.step = 5
            opts.train_idx = list(range(0, 25985))
            opts.test_idx = list(range(0, 25985))
            opts.train_idx = list(set(opts.train_idx) - set(opts.test_idx))
            #opts.train_idx = list(range(0, 25985, 5))
            #opts.test_idx = list(range(0, 25985, 55))
            #opts.train_idx = list(set(opts.train_idx) - set(opts.test_idx))
            opts.min_depth = 0.3
            #opts.dtype = 'gt'
            opts.gt_phi = 45
        elif self.dbname == 'omnithings4':
            opts.start = 0
            opts.end = 10239
            # opts.step = 1
            opts.train_idx = list(range(0, 10240 ,1))
            opts.test_idx = list(range(0, 10240, 1))
            opts.min_depth = 0.3
        elif self.dbname == 'scenecity_drone_randomsky4':
            opts.train_idx = list(range(1, 2345))
            opts.test_idx = list(range(1, 2345, 8))
            opts.train_idx = list(set(opts.train_idx) - set(opts.test_idx))
            #opts.dtype = 'gt'
            gt_phi = 90
        elif self.dbname == '211021_seerslab':
            opts.start = 1403
            opts.end = 5597
            # opts.step = 1
            opts.dtype = 'nogt'
            opts.train_idx = list(range(1403, 5597))
            opts.test_idx = list(range(1403, 5597))
            opts.min_depth = 0.3
            opts.gt_phi = 45
            opts.img_fmt = 'cam%d/%06d.png' # [cam_idx, fidx]
        elif self.dbname == 'under_parking_seq':
            opts.gt_depth_fmt = 'cam_center_square_%d_depth/%05d.tiff' # [equi_w, fidx]
            opts.gt_img_fmt = 'cam_center_square_%d/%05d.png' #[equi_w, fidx]
            opts.start = 1
            opts.end = 2049
            opts.train_idx = list(range(1, 2049))
            opts.test_idx = list(range(1, 2049))
            opts.min_depth = 0.3
            opts.gt_phi = 90
            opts.dtype = 'gt'
            # opts.dist_threshold = 10
        elif self.dbname == 'garage_nerf':
            opts.gt_depth_fmt = 'cam_center_square_%d_depth/%05d.tiff' # [equi_w, fidx]
            opts.gt_img_fmt = 'cam_center_square_%d/%05d.png' #[equi_w, fidx]
            opts.min_depth = 0.3
            opts.gt_phi = 90
            opts.dtype = 'gt'
            # opts.dist_threshold = 10
        """else:
            raise KeyError(f'Dataset name error {dbname}')"""
   
        fidxs, poses = self.splitTrajectoryResult(np.loadtxt(osp.join(self.db_path, self.opts.poses_file)).T)
        self.poses = {fidxs[i]: poses[i] for i in range(len(fidxs))}
        
        opts.start, opts.end = fidxs[0], fidxs[-1]
        # opts.train_idx = list(range(opts.start, opts.end, opts.step))
        # opts.test_idx = list(range(opts.start, opts.end, opts.step))
        
        opts.train_idx = fidxs[::opts.step]
        opts.test_idx = fidxs[::opts.step]
        self.frame_idx = fidxs[::opts.step]
        
        # self.frame_idx = list(range(
        #     opts.start, opts.end, opts.step))
        self.train_idx, self.test_idx = opts.train_idx, opts.test_idx
        #print(self.train_idx, self.test_idx)

        self.data_size = len(self.frame_idx)
        self.train_size = len(self.train_idx)
        #print(self.data_size, self.train_size)

        if opts.use_frame_idxs_for_test:
            self.test_idx = self.frame_idx
            self.opts.test_idx = self.frame_idx
        self.test_size = len(self.test_idx)
        

        # initialize capture data reader
        if opts.dtype == 'capture':
            filepath_prefix = osp.join(self.db_path, opts.cam_node_name)
            self.sensor_data_reader = SensorDataReader(0,
                SensorDataType.CAMERA_FRAME)
            self.sensor_data_reader.open(filepath_prefix)
            self.frame_idx = list(range(opts.start,
                min(opts.end, self.sensor_data_reader.end_idx), opts.step))
            self.test_idx = self.frame_idx
            self.data_size = len(self.frame_idx)
            self.test_size = len(self.test_idx)
            
        # initialize member variables
        self.ocams = [self.ocams[i - 1] for i in self.opts.camera_idxs]
        self.equirect_size = opts.equirect_size
        self.num_invdepth = opts.num_invdepth
        self.max_invdepth = 1.0 / opts.min_depth
        self.min_invdepth = EPS
        self.step_invdepth = (self.max_invdepth - self.min_invdepth) / \
            (self.num_invdepth - 1)
        self.phi_deg = opts.phi_deg
        self.phi_max_deg = opts.phi_max_deg
        if opts.use_mid_depth_sweep:
            self.mid_invdepth = 1.0 / opts.mid_depth
            self.step_invdepth1 = (self.mid_invdepth - self.min_invdepth) / \
                (self.num_invdepth // 2 - 1)
            self.step_invdepth2 = (self.max_invdepth - self.mid_invdepth) / \
                (self.num_invdepth // 2)
            self.step_invdepths = np.cumsum(
                [self.step_invdepth1] * (self.num_invdepth // 2 - 1) + \
                [self.step_invdepth2] * (self.num_invdepth // 2))

        self.equi_rays, self.equi_thetas, self.equi_phis = \
            makeSphericalRays(self.equirect_size, self.phi_deg,
                self.phi_max_deg, True)

        self.cuboid_size = opts.cuboid_size
        # self.out_cost = opts.out_cost

        if build_lt:
            lt_equirect_w = self.equirect_size[1] // \
                opts.lt_equirect_sample_ratio
            lt_equirect_h = self.equirect_size[0] // \
                opts.lt_equirect_sample_ratio
            lt_num_invdepth = self.num_invdepth // opts.lt_depth_sample_ratio
            if opts.use_mid_depth_sweep:
                lt_step_invdepth = \
                    self.step_invdepths[
                        opts.lt_depth_sample_ratio - 1::
                        opts.lt_depth_sample_ratio]
            else:
                lt_step_invdepth = \
                    self.step_invdepth * opts.lt_depth_sample_ratio

            LOG_INFO('Build lookup table for "%s"...' % (self.dbname))
            if opts.use_rounded_cuboid_sweep:
                self.grids = buildRoundedCuboidLookUpTable(
                    self.cuboid_size, (lt_equirect_h, lt_equirect_w),
                    lt_num_invdepth, self.min_invdepth, lt_step_invdepth,
                    self.phi_deg, self.phi_max_deg, self.ocams,
                    output_gpu_tensor=True, device_id=device_id)
            else:
                self.grids = buildSphereLookupTable(
                    (lt_equirect_h, lt_equirect_w),
                    lt_num_invdepth, self.min_invdepth, lt_step_invdepth,
                    self.phi_deg, self.phi_max_deg, self.ocams,
                    output_gpu_tensor=True, device_id=device_id)
            if opts.interleave_idxs:
                all_camera_idxs = set(opts.camera_idxs)
                interleaves = []
                for (i, j) in opts.interleave_idxs:
                    interleaves += [
                        opts.camera_idxs[i - 1], opts.camera_idxs[j - 1]]
                not_interleave_idxs = list(
                    all_camera_idxs - set(interleaves))
                self.grids = [self.grids[i - 1] for i in not_interleave_idxs] +\
                     [makeInterleavingGrids(
                        self.ocams[i - 1].width, self.ocams[j - 1].width,
                        self.grids[i - 1], self.grids[j - 1]) for (i, j) in \
                            opts.interleave_idxs]
            # if opts.for_car:
            #     self.grids = self.grids[:4] + [
            #         makeInterleavingGrids(
            #             self.ocams[4].width, self.ocams[6].width,
            #             self.grids[4], self.grids[6]),
            #         makeInterleavingGrids(
            #             self.ocams[5].width, self.ocams[7].width,
            #             self.grids[5], self.grids[7])]

            self.lt_equirect_size = (lt_equirect_h, lt_equirect_w)
            self.lt_num_invdepth = lt_num_invdepth
            self.lt_step_invdepth = lt_step_invdepth
            self.invdepth_indices = torch.arange(0, self.num_invdepth,
                requires_grad=False).view((1, -1, 1, 1)).float().to(device_id)
            self.valid_depth_mask = getValidGridMask(self.grids).cpu().numpy()

            if self.valid_depth_mask.shape != self.equirect_size:
                self.valid_depth_mask = imresize(
                    self.valid_depth_mask, self.equirect_size)
        else:
            self.grids = None
            self.invdepth_indices = None
            self.valid_depth_mask = None

        if opts.limit_fov_deg > 0:
            for ocam in self.ocams:
                ocam.max_theta = np.deg2rad(opts.limit_fov_deg / 2.0)
                new_mask = ocam.makeFoVMask()
                ocam.invalid_mask = np.logical_or(
                    ocam.invalid_mask, new_mask)

        if opts.input_size is not None:
            for ocam in self.ocams:
                if opts.input_size == (ocam.height, ocam.width): continue
                ocam.invalid_mask = imresize(ocam.invalid_mask,
                    opts.input_size)

        log_str = 'OmniMVS initialized \n' + \
            '# equirect size: (h: %d, w: %d)\n' % (
                self.equirect_size[0], self.equirect_size[1]) + \
            '# equirect phi (max): %.1f (%.1f)\n' % (
                self.phi_deg, self.phi_max_deg) + \
            '# num_invdepths: %d\n' % (self.num_invdepth) + \
            '# min_sweep_depth: %.2f' % (opts.min_depth)
        if self.opts.use_rounded_cuboid_sweep:
            log_str += '\n# use rounded cuboid sweeping' + \
                ', cuboid size: (l: %.2f, h: %.2f, w: %.2f)' % (
                self.cuboid_size[0], self.cuboid_size[1], self.cuboid_size[2])
        if self.opts.use_mid_depth_sweep:
            log_str += '\n# use mid-depth sweeping' + \
                ', mid-depth: %.2f' % (1 / self.mid_invdepth)
        LOG_INFO(log_str)
        # handling multiple datasets
        if self.load_multiple_datasets:
            self.__initMultipleDatasets(db_config_idxs, device_id)

    # __init__

    def __initMultipleDatasets(self, db_config_idxs: List[int] | None, device_id: int = 0):
        self.num_datasets = len(self.dbnames)
        build_lt = [self.build_lt] + [False] * (self.num_datasets - 1)
        if db_config_idxs is None:
            db_config_idxs = [0] * self.num_datasets
        else:
            db_config_idxs_set = list(set(db_config_idxs))
            for idx in db_config_idxs_set:
                build_lt[idx] = self.build_lt
        self.db_config_idxs = db_config_idxs
        self.sub_datasets = [self]
        for i in range(1, self.num_datasets):
            self.sub_datasets.append(
                OmniMVS(self.dbnames[i], db_root = self.db_root,
                    opts=self.opts, build_lt = build_lt[i], train = self.train, device_id=device_id))

        self.frame_idx_offsets = [d.data_size for d in self.sub_datasets]
        self.train_idx_offsets = [d.train_size for d in self.sub_datasets]
        self.test_idx_offsets = [d.test_size for d in self.sub_datasets]
        for i in range(1, self.num_datasets):
            self.frame_idx += self.sub_datasets[i].frame_idx
            self.train_idx += self.sub_datasets[i].train_idx
            self.test_idx += self.sub_datasets[i].test_idx
            self.train_size += self.sub_datasets[i].train_size
            self.test_size += self.sub_datasets[i].test_size
            self.data_size += self.sub_datasets[i].data_size
            if self.opts.input_size is not None:
                for ocam in self.sub_datasets[i].ocams:
                    if self.opts.input_size == (ocam.height, ocam.width): continue
                    ocam.invalid_mask = imresize(ocam.invalid_mask,
                        self.opts.input_size)

        LOG_INFO('OmniMVS loads multiple datasets')

    def forwardNetwork(self,
            net: torch.nn.Module, imgs: List[torch.Tensor],
            sub_dataset_idxs: List[int] | None = None,
            output_numpy: bool = False,
            grid_widxs: List[int] | None = None,
            off_indices=[]) \
                 -> Tuple[np.ndarray | torch.Tensor]:
        if sub_dataset_idxs is not None:
            if self.load_multiple_datasets:
                grids = []
                for i in sub_dataset_idxs:
                    grid_idx = self.db_config_idxs[i]
                    if grid_widxs is not None:
                        grids.append([g[..., grid_widxs, :] for g in \
                            self.sub_datasets[grid_idx].grids])
                    else:
                        grids.append(self.sub_datasets[grid_idx].grids)
            else:
                if grid_widxs is not None:
                    grids = [[g[..., grid_widxs, :] for g in self.grids]] * \
                        len(sub_dataset_idxs)
                else:
                    grids = [self.grids] * len(sub_dataset_idxs)
        else:
            if grid_widxs is not None:
                grids = [g[..., grid_widxs, :] for g in self.grids]
            else:
                grids = self.grids
        ##modified
        #index, prob, _ = net(imgs, grids, self.invdepth_indices, off_indices)
        # if self.out_cost:
        index, prob, resp, raw_feat, geo_feat = net(imgs, grids, self.invdepth_indices, off_indices)
        feats = []
        feats.append(raw_feat)
        # feats.append(geo_feat)

        entropy = torch.sum(-torch.log(prob + EPS) * prob, 1)
        if output_numpy:
            index = toNumpy(index.detach()) #toNumpy(index)
            entropy = scipy.ndimage.gaussian_filter(
                toNumpy(exp(entropy).detach()), sigma=1) #toNumpy(exp(entropy)), sigma=1) 
        
        return index, entropy, prob, resp, feats
        
        # if self.out_cost:
        #     return index, entropy, resp
            
    def forwardNetwork_for_integrate(self,
            net: torch.nn.Module, imgs: List[torch.Tensor],
            sub_dataset_idxs: List[int] | None = None,
            output_numpy: bool = False,
            grid_widxs: List[int] | None = None,
            off_indices=[],
            resp_volume_freeze=True) \
                 -> Tuple[np.ndarray | torch.Tensor]:
        with torch.no_grad():
            if sub_dataset_idxs is not None:
                if self.load_multiple_datasets:
                    grids = []
                    for i in sub_dataset_idxs:
                        grid_idx = self.db_config_idxs[i]
                        if grid_widxs is not None:
                            grids.append([g[..., grid_widxs, :] for g in \
                                self.sub_datasets[grid_idx].grids])
                        else:
                            grids.append(self.sub_datasets[grid_idx].grids)
                else:
                    if grid_widxs is not None:
                        grids = [[g[..., grid_widxs, :] for g in self.grids]] * \
                            len(sub_dataset_idxs)
                    else:
                        grids = [self.grids] * len(sub_dataset_idxs)
            else:
                if grid_widxs is not None:
                    grids = [g[..., grid_widxs, :] for g in self.grids]
                else:
                    grids = self.grids
            resp, raw_feat, geo_feat = net(imgs, grids, self.invdepth_indices, off_indices, integrate=True)
        
        # pdb.set_trace()
        if False and not resp_volume_freeze:
            resp = Variable(resp, requires_grad=True)
        
        return resp, raw_feat, geo_feat

    def __len__(self) -> int:
        if self.train:
            return self.train_size
        else:
            return self.test_size
        """if self.opts.dtype == 'nogt':
            return self.data_size
        elif self.train:
            return self.train_size
        else:
            return self.test_size"""

    def __getitem__(self, i: int):
        if self.opts.dtype == 'nogt':
            return self.loadSample(i)
        else:
            opts = Edict()
            opts.remove_gt_noise = self.opts.remove_gt_noise
            opts.morph_win_size = self.opts.morph_win_size
            opts.input_size = self.opts.input_size
            if self.train:
                return self.loadTrainSample(i, read_input_image=True, args=opts)
            else:
                return self.loadTestSample(i,
                    read_input_image=self.opts.read_input_image, args=opts)

    def splitTrajectoryResult(self, trajectory: np.ndarray) \
          -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = trajectory.shape[0]
        if rows != 8:
            sys.exit(
                'Trajectory must has 8 rows '
                '(fidx, rx, ry, rz, tx ty, tz, timestamp)')
        fidxs = trajectory[0, :].astype(np.int32).T.tolist()
        poses = trajectory[1:7, :].astype(np.float32).T
        poses = np.split(poses, len(fidxs))
        return fidxs, poses

    def indexToInvdepth(self,
            idx: float | np.ndarray | torch.Tensor,
            start_index: int = 0) -> float | np.ndarray | torch.Tensor:
        idx = idx - start_index
        if self.opts.use_mid_depth_sweep:
            if type(idx) == float:
                if idx <= self.num_invdepth//2 - 1:
                    invdepth = self.min_invdepth + idx * self.step_invdepth1
                else:
                    invdepth = self.mid_invdepth + \
                        (idx - self.num_invdepth//2 + 1) * self.step_invdepth2
            else:
                invdepth = zeros_like(idx)
                step1 = idx <= self.num_invdepth//2 - 1
                step2 = logical_not(step1)
                invdepth[step1] = self.min_invdepth + idx[step1] * \
                    self.step_invdepth1
                invdepth[step2] = self.mid_invdepth + \
                        (idx[step2] - self.num_invdepth//2 + 1) * \
                            self.step_invdepth2
        else:
            invdepth = self.min_invdepth + \
                idx * self.step_invdepth
        if type(idx) == float or not self.opts.use_rounded_cuboid_sweep:
            return invdepth
        else:
            return inverseCuboidRadiusToDepth(invdepth, self.cuboid_size,
                self.equi_thetas, self.equi_phis)

    def invdepthToIndex(self,
            invdepth: float | np.ndarray | torch.Tensor,
            start_index: int = 0) -> float | np.ndarray | torch.Tensor:
        if not type(invdepth) == float and self.opts.use_rounded_cuboid_sweep:
            invdepth = inverseDepthToCuboidRadius(invdepth, self.cuboid_size,
                self.equi_thetas, self.equi_phis)
        if self.opts.use_mid_depth_sweep:
            if type(invdepth) == float:
                if invdepth <= self.mid_invdepth:
                    index = (invdepth - self.min_invdepth) / self.step_invdepth1
                else:
                    index = (invdepth - self.mid_invdepth) / \
                        self.step_invdepth2 + (self.num_invdepth // 2 - 1)
            else:
                index = zeros_like(invdepth)
                step1 = invdepth <= self.mid_invdepth
                step2 = logical_not(step1)
                index[step1] = (invdepth[step1] - self.min_invdepth) / \
                    self.step_invdepth1
                index[step2] = (invdepth[step2] - self.mid_invdepth) / \
                        self.step_invdepth2 + (self.num_invdepth // 2 - 1)
        else:
            index = (invdepth - self.min_invdepth) / \
                self.step_invdepth
        return index + start_index

    def invdepthToCuboidRadius(self,
            invdepth: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        invradius = inverseDepthToCuboidRadius(invdepth, self.cuboid_size,
            self.equi_thetas, self.equi_phis)
        return invradius

    def respToCameraInvdepth(self,
            resp: ArrayType, rig_roll: int = 0,
            output_index: bool = True, output_numpy: bool = False,
            compute_idxs = None, grids_cam = None) \
                -> List[ArrayType]:
        if not isTorchArray(resp):
            resp = torch.tensor(resp).to(self.device_id)
        D, H, W = resp.shape[-3:]
        resp = resp.reshape((-1, D, H, W)) # B, D H W
        if rig_roll > 0: resp = torch.roll(resp, -rig_roll, -1)
        B = resp.shape[0]
        # add circular pad 1
        resp = concat((resp, resp[:, :, :, 0].unsqueeze(-1)), -1)
        out = []
        if grids_cam is None: grids_cam = self.grids_cam
        if compute_idxs is None: compute_idxs = range(len(grids_cam))
        for i in compute_idxs:
            grid = grids_cam[i]
            if B > 1:
                resp_cams = [F.grid_sample(
                    resp[b, ...].unsqueeze(0).unsqueeze(0),
                    grid, align_corners=True) \
                        for b in range(B)]
                resp_cam = concat(resp_cams, 0).squeeze(1)
            else:
                resp_cam = F.grid_sample(resp.unsqueeze(0),
                    grid, align_corners=True).squeeze(1)
            prob_cam = F.softmax(resp_cam, 1)
            index_cam = torch.mul(prob_cam, self.invdepth_indices)
            index_cam = torch.sum(index_cam, 1)
            index_cam = self.indexToInvdepth(index_cam)
            if output_index:
                out.append(index_cam)
            else:
                invdepth_cam = self.min_invdepth + \
                    index_cam * self.step_invdepth_cam
                out.append(invdepth_cam)
        if output_numpy:
            out = [toNumpy(o) for o in out]
        return out
    
    def getSubDataset(self, dbname: str) -> OmniMVS:
        if not self.load_multiple_datasets:
            return self
        return self.sub_datasets[self.dbnames.index(dbname)]

    def get3Dpoint(self, invdepth, transform=None):
        invdepth = toNumpy(invdepth)
        depth = 1 / invdepth.reshape((1, -1))
        P = depth * self.equi_rays
        if transform is not None:
            P = applyTransform(transform, P)
        return P

    def getPanorama(self, imgs, invdepth, transform=None):
        invdepth = toNumpy(invdepth)
        depth = 1 / invdepth.reshape((1, -1))
        P = depth * self.equi_rays
        if transform is not None:
            P = applyTransform(transform, P)
        pano_sum = np.zeros(self.equirect_size)
        valid_count = np.zeros(self.equirect_size, dtype=np.uint8)
        for i in range(len(imgs)):
            P2 = applyTransform(self.ocams[i].rig2cam, P)
            p = self.ocams[i].rayToPixel(P2)
            #p, theta = self.ocams[i].rayToPixel(P2, out_theta=True)
            grid = pixelToGrid(p, self.equirect_size,
                (self.ocams[i].height,self.ocams[i].width))
            equi_im = toNumpy(interp2D(imgs[i], grid,1))
            valid = (p[0,:] >= 0).reshape(
                self.equirect_size)
            pano_sum[valid] += equi_im[valid]
            valid_count[valid] += 1
        valid = valid_count > 0
        pano_sum[valid] = np.round(pano_sum[valid] / valid_count[valid])
        return pano_sum.astype(np.uint8)

    def getRGBViewPanorama(self, imgs: List[np.ndarray], invdepth: torch.Tensor | np.ndarray, 
                           valid=None, transform=None) -> List[torch.Tensor] | List[np.ndarray]:
        equirect_size = self.equirect_size
        depth = 1 / invdepth.reshape((1, -1))

        P = depth * torch.Tensor(self.equi_rays).to(self.device_id) if isTorchArray(depth) else self.equi_rays
        if transform is not None:
            P = applyTransform(transform, P)

        equi_ims = []
        valid_masks = []
        for i in range(4):
            img = imgs[i]
            if not isTorchArray(depth):
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            P2 = applyTransform(self.ocams[i].rig2cam, P)
            p = self.ocams[i].rayToPixel(P2)

            grid = pixelToGrid(p, equirect_size,
                (self.ocams[i].height,self.ocams[i].width))
            # equi_im = toNumpy(interp2D(img, grid, 0))
            equi_im = interp2D(torch.Tensor(img).to(self.device_id) \
                                if isTorchArray(depth) else img, \
                                grid, 0)
            valid = (p[0,:] >= 0).reshape(self.equirect_size)
            valid_masks.append(valid)
            if isTorchArray(equi_im):
                equi_im = equi_im.permute(2, 0, 1)
            equi_ims.append(equi_im)

        return equi_ims, valid_masks

    def getRefPanorama(self, invdepth: torch.Tensor | np.ndarray, transform=None) -> \
                        List[torch.Tensor] | List[np.ndarray]:
        ref_depth = []
        depth = 1 / invdepth.reshape((1, -1))

        P = depth * torch.Tensor(self.equi_rays).to(self.device_id) if isTorchArray(depth) else self.equi_rays
        if transform is not None:
            P = applyTransform(transform, P)
        pano_sum = np.zeros(self.equirect_size)
        
        for i in range(4):
            P2 = applyTransform(self.ocams[i].rig2cam, P)
            
            depth_cam = sqrt((P2**2).sum(0))
            invdepth_cam = 1.0 / depth_cam
            refidx_cam = self.invdepthToIndex(invdepth_cam)

            refidx_cam = refidx_cam.reshape(self.equirect_size)

            ref_depth.append(refidx_cam)

        return ref_depth

    def getViewPanorama(self, imgs: torch.Tensor | np.ndarray, invdepth: torch.Tensor | np.ndarray, transform=None) -> \
                         List[torch.Tensor] | List[np.ndarray]:
        equirect_size = self.equirect_size
        depth = 1 / invdepth.reshape((1, -1))

        equi_ims = []
        P = depth * torch.Tensor(self.equi_rays).to(self.device_id) if isTorchArray(depth) else self.equi_rays
        if transform is not None:
            P = applyTransform(transform, P)

        for i in range(len(imgs)):
            P2 = applyTransform(self.ocams[i].rig2cam, P)
            p = self.ocams[i].rayToPixel(P2)

            grid = pixelToGrid(p, equirect_size,
                (self.ocams[i].height,self.ocams[i].width))
            equi_im = interp2D(imgs[i], grid, 1)
            equi_ims.append(equi_im)
      
        return equi_ims

    def makeVisImage(self, imgs, invdepth: np.ndarray,
                     entropy=None, gt=None, valid=None, transform=None,
                     vis_depthmap=False, max_depth=100.0, pano=None,
                     vis_errormap=True, depth_colormap='magma',
                     black_invalid=True) -> \
                         np.ndarray:
        def addColorDepthBar(invdepth, invdepth_rgb, vmin, vmax):
            h, w =  invdepth.shape
            bar_w = int(w * 0.03)
            bar_h = int(h * 0.95)
            vstep = (vmax - vmin) / (bar_h - 1)
            invdepths = np.arange(0, bar_h, 1, dtype=np.float32) / \
                (bar_h - 1) * (vmax - vmin) + vmin
            bar = colorMapOliver(
                np.tile(invdepths.reshape(-1, 1), (1, bar_w)), vmin, vmax)
            black = np.zeros_like(bar)
            bar = concat([black, bar], 1)
            ref_depths = (1 / vmax) * np.array([1.0, 1.5, 3.0, 15.0])
            ref_idxs = ((1 / ref_depths) - vmin) / vstep
            ref_idxs = np.round(ref_idxs).astype(np.int32)
            line = np.array([255, 255, 255], dtype=np.uint8).reshape((1, 3))
            line = np.tile(line, (bar_w, 1))
            pos_x = w - (2 * bar_w) - 10
            pos_y = (h - bar_h) // 2
            invdepth_rgb[pos_y:pos_y+bar_h, pos_x:pos_x+bar_w,:] = \
                ((invdepth_rgb[pos_y:pos_y+bar_h, pos_x:pos_x+bar_w,:].astype(
                    np.float32) + \
                bar[:,:bar_w].astype(np.float32)) / 2.0).astype(np.uint8)
            invdepth_rgb[pos_y:pos_y+bar_h, pos_x+bar_w:pos_x+2*bar_w,:] = \
                bar[:,bar_w:]
            for n, idx in enumerate(ref_idxs):
                if idx >= bar_h: continue
                invdepth_rgb[pos_y+idx, pos_x:pos_x+bar_w, :] = line
                text = '%.1f' % ref_depths[n]
                textsize, b = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.putText(invdepth_rgb, '%.1f' % (ref_depths[n]),
                    (pos_x + (bar_w - textsize[0]) // 2,
                     pos_y + idx - textsize[1]//2 + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        for i in range(len(imgs)):
            imgs[i] = toNumpy(imgs[i]).squeeze()
            imgs[i][self.ocams[i].invalid_mask] = 0
        if self.opts.for_car and len(imgs) == 8:
            inputs = concat(
                [concat([imgs[0], imgs[1]], axis=1),
                 concat([imgs[3], imgs[2]], axis=1),
                 concat([imgs[4], imgs[5]], axis=1),
                 concat([imgs[7], imgs[6]], axis=1)], axis=0)
        elif self.opts.for_car and len(imgs) == 6:
            inputs = concat(
                [concat([imgs[0], imgs[1]], axis=1),
                 concat([imgs[3], imgs[2]], axis=1),
                 concat([imgs[4], imgs[5]], axis=1)], axis=0)
        else:
            inputs = concat(
                [concat([imgs[0], imgs[1]], axis=1),
                concat([imgs[3], imgs[2]], axis=1)], axis=0)

        if pano is None:
            pano = self.getPanorama(imgs, invdepth, transform)
            pano_rgb = np.tile(pano[..., np.newaxis], (1, 1, 3))
        else:
            if len(pano.shape) == 2:
                pano_rgb = np.tile(pano[..., np.newaxis], (1, 1, 3))
            else:
                pano_rgb = pano
        if not pano_rgb.dtype == np.uint8:
            pano_rgb = np.round(pano_rgb * 255).astype(np.uint8)

        # make color bar
        if self.opts.for_car:
            inv_r = self.invdepthToCuboidRadius(invdepth)

        vis_max_invdepth = max(0.3, self.max_invdepth / 2)
        if vis_depthmap:
            if self.opts.for_car:
                depth = 1 / inv_r
            else:
                depth = 1 / invdepth
            invdepth_rgb = colorMap(depth_colormap, np.log(depth),
                np.log(1 / self.max_invdepth), np.log(max_depth))
        else:
            if self.opts.for_car:
                invdepth_rgb = colorMap(depth_colormap, inv_r,
                    self.min_invdepth, vis_max_invdepth)
            else:
                invdepth_rgb = colorMap(depth_colormap, invdepth,
                    self.min_invdepth, vis_max_invdepth)

        vis = np.concatenate((pano_rgb, invdepth_rgb), axis=0)
        if entropy is not None:
            entropy_rgb = colorMap('inferno', entropy, 1, 10)
            vis = np.concatenate((vis, entropy_rgb), axis=0)
        if valid is not None:
            valid = toNumpy(valid).squeeze()
            invalid = logical_not(valid)
        else:
            invalid = None
        if gt is not None and len(gt) > 0:
            gt = toNumpy(gt).squeeze()
            if vis_errormap:
                err = np.abs(self.invdepthToIndex(invdepth) - gt).squeeze()
                if invalid is not None:
                    invalid = logical_or(invalid,
                        gt < 0, gt >= self.num_invdepth)
                    err[invalid] = np.nan
                err_rgb = colorMap('jet', err, 0, 10)
                vis = np.concatenate((vis, err_rgb), axis=0)
            else:
                gt_rgb = colorMap(depth_colormap, gt, self.min_invdepth,
                    vis_max_invdepth)
                vis = np.concatenate((vis, gt_rgb), axis=0)
        ratio = vis.shape[0] / float(inputs.shape[0])
        inputs_rgb = np.tile(
            imrescale(inputs, ratio)[..., np.newaxis], (1, 1, 3))
        if black_invalid and invalid is not None:
            num_vis = vis.shape[0] // invdepth_rgb.shape[0]
            invalids = np.tile(invalid[..., np.newaxis], (num_vis, 1, 3))
            vis[invalids] = 0
        if self.opts.for_car:
            equi_h = invdepth.shape[0]
            addColorDepthBar(inv_r, vis[equi_h:2*equi_h, :, :],
                self.min_invdepth, vis_max_invdepth)
        vis = np.concatenate((inputs_rgb, vis), axis=1)
        return vis
    
    def makeInvdepthVis(self, invdepth: np.ndarray,
                     vis_depthmap=False, max_depth=100.0,
                     depth_colormap='oliver') -> \
                         np.ndarray:
        # make color bar
        if self.opts.for_car:
            inv_r = self.invdepthToCuboidRadius(invdepth)

        vis_max_invdepth = max(0.3, self.max_invdepth / 2)
        if vis_depthmap:
            if self.opts.for_car:
                depth = 1 / inv_r
            else:
                depth = 1 / invdepth
            invdepth_rgb = colorMap(depth_colormap, np.log(depth),
                np.log(1 / self.max_invdepth), np.log(max_depth))
        else:
            if self.opts.for_car:
                invdepth_rgb = colorMap(depth_colormap, inv_r,
                    self.min_invdepth, vis_max_invdepth)
            else:
                invdepth_rgb = colorMap(depth_colormap, invdepth,
                    self.min_invdepth, vis_max_invdepth)
        return invdepth_rgb
    
    ## Load Sample ===================================================
    def findDatasetIndex(self,
            sample_idx: int, offsets: List[int]) -> Tuple[int, int]:
        for i in range(self.num_datasets - 1):
            if sample_idx < offsets[i]:
                return i, sample_idx
            sample_idx -= offsets[i]
        return self.num_datasets - 1, sample_idx

    def loadSample(self,
            i: int, read_input_image = True,
            sub_dataset_idx: int | None = 0,
            recursive_level: int = 1,
            args=None):
        fidx = self.frame_idx[i]
        if self.load_multiple_datasets and recursive_level > 0:
            didx, i = self.findDatasetIndex(i, self.frame_idx_offsets)
            return self.sub_datasets[didx].loadSample(
                i, read_input_image, didx, recursive_level - 1, args)
        else:
            return \
                self.__loadSample(fidx, read_input_image, args), sub_dataset_idx

    def loadTrainSample(self,
            i: int, read_input_image = True,
            sub_dataset_idx:int | None = None,
            recursive_level: int = 1,
            args=None):
        fidx = self.train_idx[i]
        if self.load_multiple_datasets and recursive_level > 0:
            didx, i = self.findDatasetIndex(i, self.train_idx_offsets)
            return self.sub_datasets[didx].loadTrainSample(
                i, read_input_image, didx, recursive_level - 1, args)
        else:
            return \
                self.__loadSample(fidx, read_input_image, args), sub_dataset_idx

    def loadTestSample(self,
            i: int, read_input_image = True,
            sub_dataset_idx: int | None = None,
            recursive_level: int = 1,
            args=None):
        fidx = self.test_idx[i]
        if self.load_multiple_datasets and recursive_level > 0:
            didx, i = self.findDatasetIndex(i, self.test_idx_offsets)
            return self.sub_datasets[didx].loadTestSample(
                i, read_input_image, didx, recursive_level - 1, args)
        else:
            return \
                self.__loadSample(fidx, read_input_image, args), sub_dataset_idx

    def __loadSample(self,
            fidx: int, read_input_image: bool, args: Edict | None = None) \
                -> Tuple[
                    List[torch.Tensor], torch.Tensor, torch.Tensor,
                    List[np.ndarray]]:
        opts = Edict()
        opts.remove_gt_noise = True
        opts.morph_win_size = 5
        opts.input_size = None
        opts = argparse(opts, args)
        imgs, raw_imgs, gt, valid = None, None, None, None
        gt_img = None
        gt_invdepth = None
        if read_input_image:
            input_images = self.loadImages(
                fidx, out_raw_imgs=True, use_rgb=self.opts.use_rgb,
                resize=opts.input_size)
            if input_images is None:
                LOG_ERROR('failed to load images fidx: %d' % fidx)
                return None
            imgs, raw_imgs = input_images
            imgs = [torch.tensor(I).float() for I in imgs]

        if self.opts.dtype == 'gt':
            gt = self.loadGTInvdepthIndex(fidx,
                opts.remove_gt_noise, opts.morph_win_size)
            gt_img = self.loadGTImage(fidx)
            gt_img = torch.tensor(gt_img).float()
            if gt is not None:
                gt, gt_invdepth = gt
                valid = np.logical_and(
                    gt >= 0, gt < self.num_invdepth).astype(np.bool)
                gt = torch.tensor(gt).float()
                gt_invdepth = torch.tensor(gt_invdepth).float()
                if self.valid_depth_mask is not None:
                    valid = np.logical_and(valid, self.valid_depth_mask)
                # check gt depth is visible
                gt_pts = (1 / gt_invdepth).reshape((1, -1)) * self.equi_rays
                visible_mask = np.zeros((self.equirect_size)).astype(np.bool)
                for i in range(len(self.ocams)):
                    P = applyTransform(self.ocams[i].rig2cam, gt_pts)
                    p = self.ocams[i].rayToPixel(P)
                    vis = (p[0,:] >= 0).reshape(self.equirect_size)
                    visible_mask[vis] = True
                valid = np.logical_and(valid, visible_mask)
                valid = torch.tensor(valid).bool()

        pose = self.poses[fidx]
        R, tr = getRot(pose), getTr(pose)
        c2w = np.eye(4)
        c2w[:3, :] = np.concatenate((R, tr), axis=1)
        c2w[:3, 1] *= -1
        c2w[:3, 2] *= -1
        c2w = torch.from_numpy(c2w).float()
        
        return (imgs, gt, valid, raw_imgs, c2w, gt_img, gt_invdepth)

    ## File I/O ======================================================
    def loadImages(self, fidx: int, out_raw_imgs=False, use_rgb=False,
                   resize: Tuple[int, int] | None = None) \
                    -> List[np.ndarray]:
        # import pdb
        # pdb.set_trace()
        imgs = []
        raw_imgs = []
        #print("dataset: " + self.dbname + " / fidx:" + '%05d.png'%fidx)
        if self.opts.dtype == 'capture':
            data = self.sensor_data_reader.read(fidx)
        for i, camera_idx in enumerate(self.opts.camera_idxs):
            if self.opts.dtype == 'capture':
                I = data.getImage(i)
            else:
                file_path = osp.join(
                    self.db_path, self.opts.img_fmt % (camera_idx, fidx))
                I = readImage(file_path, read_or_die=False)
            if I is None:
                return None, None
            if resize is not None:
                I = imresize(I, resize)
            if not use_rgb and len(I.shape) == 3 and I.shape[2] == 3:
                I = rgb2gray(I, channel_wise_mean=True)
            if out_raw_imgs: raw_imgs.append(I)
            I = normalizeImage(I, self.ocams[i].invalid_mask)
            if len(I.shape) == 2:
                I = np.expand_dims(I, axis=0) # make 1 x H x W
                if use_rgb: I = np.tile(I, (3, 1, 1)) # make 3 x H x W
            else:
                I = np.transpose(I, (2, 0, 1)) # make C x H x W
            imgs.append(I)
        if out_raw_imgs: return (imgs, raw_imgs)
        else: return imgs

    def getPanoramaViewdepthIndex(self, view_depth: torch.Tensor | np.ndarray, invdepth: torch.Tensor | np.ndarray, 
                                  remove_gt_noise = True, morph_win_size:int = 5) -> \
                                Tuple[torch.Tensor, torch.Tensor] | Tuple[np.ndarray, np.ndarray] | None:
        h, w = self.equirect_size

        # import pdb
        # pdb.set_trace()
        depth_pano = self.getViewPanorama(view_depth, invdepth, transform=None)
        
        view_depth_idx =[]
        for i in range(len(depth_pano)):
            gt = depth_pano[i]
            gt_idx = self.invdepthToIndex(gt)
            gt_idx[gt_idx > (self.num_invdepth - 1)] = self.num_invdepth - 1
            if isTorchArray(gt_idx):
                gt_idx[torch.isnan(gt_idx)] = -1
            else:
                gt_idx[np.isnan(gt_idx)] = -1
            if not remove_gt_noise:
                return gt_idx, gt

            small_than_zero = gt_idx < 0
            gt_idx[small_than_zero] = -1
            
            view_depth_idx.append(gt_idx)

        return view_depth_idx, depth_pano

    def loadGTInvdepthIndex(self, fidx: int, remove_gt_noise = True,
                            morph_win_size:int = 5) -> \
                                Tuple[np.ndarray, np.ndarray] | None:
        h, w = self.equirect_size
        gt_depth_file = osp.join(
            self.db_path, self.opts.gt_depth_fmt % (w, fidx))
        gt = self.readInvdepth(gt_depth_file)
        if gt is None: return None

        gt_h, gt_w = gt.shape
        if gt_w != w:
            LOG_ERROR('width of GT depth map is not equal to equirect size')
            return None

        # crop height
        gt_phi_min_deg, gt_phi_max_deg = -self.opts.gt_phi, self.opts.gt_phi
        if self.phi_max_deg > 0.0:
            phi_min_deg, phi_max_deg = self.phi_deg, self.phi_max_deg
        else:
            phi_min_deg, phi_max_deg = -self.phi_deg, self.phi_deg

        if phi_min_deg < gt_phi_min_deg or phi_max_deg > gt_phi_max_deg:
            print(phi_min_deg,gt_phi_min_deg,phi_max_deg,gt_phi_max_deg )
            LOG_ERROR('vertical FoV is out of GT depth map')
            return None

        gt_px_per_deg = gt_h / (gt_phi_max_deg - gt_phi_min_deg)
        px_per_deg = h / (phi_max_deg - phi_min_deg)
        if abs(gt_px_per_deg - px_per_deg) > 1e-6:
            LOG_ERROR('number of pixels/vertical FoV is not equal to GTs')
            return None

        start_h = int(round(gt_px_per_deg * (gt_phi_max_deg - phi_max_deg)))
        gt = gt[start_h:start_h + h, :]

        gt_idx = self.invdepthToIndex(gt)
        gt_idx[gt_idx > (self.num_invdepth - 1)] = self.num_invdepth - 1
        gt_idx[np.isnan(gt_idx)] = -1
        if not remove_gt_noise:
            return gt_idx, gt
        # make valid mask
        morph_filter = np.ones(
            (morph_win_size, morph_win_size), dtype=np.uint8)
        finite_depth = gt >= 1e-4 # <= 10000 m
        closed_depth = scipy.ndimage.binary_closing(
            finite_depth, morph_filter)
        infinite_depth = np.logical_not(finite_depth)
        infinite_hole = np.logical_and(infinite_depth, closed_depth)
        gt_idx[infinite_hole] = -1
        return gt_idx, gt

    def loadGTImage(self, fidx: int) -> np.ndarray | None:
        h, w = self.equirect_size
        gt_img_file = osp.join(
            self.db_path, self.opts.gt_img_fmt % (w, fidx))
        gt_img = np.array(Image.open(gt_img_file), dtype=np.float32) / 255.
        
        gt_h, gt_w, _ = gt_img.shape
        if gt_w != w:
            LOG_ERROR('width of GT Image map is not equal to equirect size')
            return None

        # crop height
        gt_phi_min_deg, gt_phi_max_deg = -self.opts.gt_phi, self.opts.gt_phi
        if self.phi_max_deg > 0.0:
            phi_min_deg, phi_max_deg = self.phi_deg, self.phi_max_deg
        else:
            phi_min_deg, phi_max_deg = -self.phi_deg, self.phi_deg

        if phi_min_deg < gt_phi_min_deg or phi_max_deg > gt_phi_max_deg:
            print(phi_min_deg,gt_phi_min_deg,phi_max_deg,gt_phi_max_deg )
            LOG_ERROR('vertical FoV is out of GT image map')
            return None

        gt_px_per_deg = gt_h / (gt_phi_max_deg - gt_phi_min_deg)
        px_per_deg = h / (phi_max_deg - phi_min_deg)
        if abs(gt_px_per_deg - px_per_deg) > 1e-6:
            LOG_ERROR('number of pixels/vertical FoV is not equal to GTs')
            return None

        start_h = int(round(gt_px_per_deg * (gt_phi_max_deg - phi_max_deg)))
        gt_img = gt_img[start_h:start_h + h, :, :]
        
        return gt_img

    def readInvdepth(self, path: str) -> np.ndarray:
        if not osp.exists(path):
            LOG_ERROR('Failed to load invdepth: %s' % path)
            return None
        _, ext = osp.splitext(path)
        if ext == '.png':
            step_invdepth = (self.max_invdepth - self.min_invdepth) / 65500.0
            quantized_inv_index = readImage(
                path, read_or_die=False).astype(np.float32)
            if quantized_inv_index is None:
                return None
            invdepth = self.min_invdepth + quantized_inv_index * step_invdepth
            return invdepth
        elif ext == '.tif' or ext == '.tiff':
            return readImageFloat(path, read_or_die=False)
        else:
            return np.fromfile(path, dtype=np.float32)

    def writeInvdepth(self, invdepth: np.ndarray, path: str) -> None:
        _, ext = osp.splitext(path)
        if ext == '.png':
            step_invdepth = (self.max_invdepth - self.min_invdepth) / 65500.0
            quantized_inv_index = (invdepth - self.min_invdepth) / step_invdepth
            writeImage(quantized_inv_index.round().astype(np.uint16), path)
        elif ext == '.tif' or ext == '.tiff':
            thumbnail = colorMap('oliver', invdepth,
                self.min_invdepth, self.max_invdepth)
            thumbnail = imrescale(thumbnail, 1)
            writeImageFloat(invdepth.astype(np.float32), path, thumbnail)
        else:
            invdepth.astype(np.float32).tofile(path)

    def readEntropy(self, path: str) -> np.ndarray:
        _, ext = osp.splitext(path)
        if ext == '.tif' or ext == '.tiff':
            return readImageFloat(path)
        else:
            return np.fromfile(path, dtype=np.float32)

    def writeEntropy(self, entropy: np.ndarray, path: str) -> None:
        _, ext = osp.splitext(path)
        if ext == '.tif' or ext == '.tiff':
            thumbnail = colorMap('inferno', entropy, 1, 10)
            thumbnail = imrescale(thumbnail, 1)
            writeImageFloat(entropy.astype(np.float32), path, thumbnail)
        else:
            entropy.astype(np.float32).tofile(path)

class BatchCollator:
    def __init__(self, data):
        datas, sub_db_idxs = list(zip(*data))
        data, self.sub_db_idxs = [], []

        for i in range(len(datas)):
            if datas[i] is not None:
                data.append(datas[i])
                self.sub_db_idxs.append(sub_db_idxs[i])

        data = list(zip(*data))
        if data[0][0]:
            self.imgs = [torch.stack(I, 0) for I in list(zip(*data[0]))]
        all_batch_have_gt = all([gt is not None for gt in data[1]])
        all_batch_have_gt_img = all([gt_img is not None for gt_img in data[5]])
        all_batch_have_gt_invdepth = all([gt_invdepth is not None for gt_invdepth in data[6]])

        if all_batch_have_gt:
            self.gt = torch.stack(data[1], 0)
            self.valid = torch.stack(data[2], 0)
            if all_batch_have_gt_img:
                self.gt_img = torch.stack(data[5], 0)
            else:
                self.gt_img = None
            if all_batch_have_gt_invdepth:
                self.gt_invdepth = torch.stack(data[6], 0)
            else:
                self.gt_invdepth = None
        else:
            self.gt = None
            self.valid = None
            self.gt_img = None
            self.gt_invdepth = None
        self.raw_imgs = data[3]
        if len(self.raw_imgs) == 1:
            self.raw_imgs = self.raw_imgs[0]

        self.c2w = torch.stack(data[4], 0)

    def pin_memory(self):
        self.imgs = [I.cuda() for I in self.imgs]
        self.c2w = self.c2w.cuda()  # change to device
        if self.gt is not None:
            self.gt = self.gt.cuda()
            self.valid = self.valid.cuda()
        if self.gt_img is not None:
            self.gt_img = self.gt_img.cuda()
        if self.gt_invdepth is not None:
            self.gt_invdepth = self.gt_invdepth.cuda()
        return self

    @staticmethod
    def collate(data):
        return BatchCollator(data)