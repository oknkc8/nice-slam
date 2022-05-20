# utils.camera.py
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
from src.omni_utils.geometry import *
from src.omni_utils.image import *

from enum import Enum

import yaml
from scipy import ndimage

class CameraType(Enum):
    PINHOLE = "pinhole"
    OPENCV_FISHEYE = "opencv_fisheye"
    UWFISHEYE = "uwfisheye"
    OCAM = "ocam"
    OTHER = "other"

class CameraModel:
    def __init__(self):
        self.id = 0
        self.type = CameraType.OTHER
        self.width, self.height = 0, 0
        self.cam2rig = concat(
            (np.identity(3), np.zeros((3, 1))), 1).astype(np.float64)
        self.rig2cam = self.cam2rig.copy()
        self.invalid_mask = None

    def LoadFromDict(self, dict: Dict) -> bool:
        try:
            if 'cam_id' in dict.keys():
                self.id = dict['cam_id']
            if self.type == CameraType.OCAM:
                self.height, self.width = dict['image_size']
            else:
                self.width, self.height = dict['image_size']
            if 'pose' in dict.keys():
                self.cam2rig = np.array(dict['pose']).reshape((6, 1))
            elif  'cam2rig_pose' in dict.keys():
                self.cam2rig = np.array(dict['cam2rig_pose']).reshape((6, 1))
            self.rig2cam = inverseTransform(self.cam2rig)
            return True
        except KeyError as e:
            LOG_ERROR(e)
            return False

    @property
    def image_size(self):
        return self.height, self.width

    @staticmethod
    def CreateFromDict(dict: Dict) -> CameraModel | None:
        cam_type = CameraType(dict['model'])
        if cam_type is CameraType.OCAM:
            cam = OcamModel()
        elif cam_type is CameraType.UWFISHEYE:
            cam = UWFisheyeModel()
        elif cam_type is CameraType.OPENCV_FISHEYE:
            cam = OpenCVFisheyeModel()
        else:
            LOG_ERROR('unknown camera model "%s"' % cam_type)
            return None
        if not cam.LoadFromDict(dict):
            return None
        return cam

    def pixelToRay(self, pts2d: np.ndarray | torch.Tensor) \
            -> np.ndarray | torch.Tensor:
        raise NotImplementedError
    def rayToPixel(self, pts3d: np.ndarray | torch.Tensor) \
            -> np.ndarray | torch.Tensor:
        raise NotImplementedError

    def getPixelGrid(self) -> np.ndarray:
        xs, ys = np.meshgrid(range(self.width), range(self.height))
        pts2d = concat((xs.reshape((1, -1)), ys.reshape((1, -1))), 0)
        return pts2d

    def makeFoVMask(self, mask: np.ndarray | None = None) -> np.ndarray:
        pts2d = self.getPixelGrid()
        pts3d = self.pixelToRay(pts2d)
        pt = self.rayToPixel(pts3d)
        valid_mask = logical_or(pt[0,:] == -1, pt[1,:] == -1)
        valid_mask = valid_mask.reshape((self.height, self.width))
        if mask is not None :
            valid_mask = logical_or(valid_mask, mask.squeeze())
        return valid_mask

    def getRThetas(self, pix2ray=True, num_sample=500) \
            -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class PinholeModel(CameraModel):
    def __init__(self):
        super().__init__()
        self.xc, self.yc, self.fx, self.fy = 0, 0, 1, 1
        self.k = np.zeros(0)
        self.max_theta = -1.0
        self.alpha = 0
        self.type = CameraType.PINHOLE
        self.flag = 0 #non-trans 0 trans 1

    @property
    def K(self):
        return np.array(
            [[self.fx, self.alpha, self.cx],
             [0, self.fy, self.cy],
             [0, 0, 1]])

    def loadFromDict(self, dict: Dict) -> bool:
        if not super().loadFromDict(dict): return False
        try:
            self.fx, self.fy = dict['focal_length']
            self.xc, self.yc = dict['center']
            if 'lens_distort' in dict.keys():
                self.k = np.zeros(5)
                for i, d in enumerate(dict['lens_distort']):
                    self.k[i] = d
            if 'alpha' in dict.keys():
                self.alpha = dict['alpha']
            self.max_theta = np.deg2rad(dict['max_fov'] / 2.0)
            if 'invalid_mask' in dict.keys():
                self.invalid_mask_file = dict['invalid_mask']
            else:
                self.invalid_mask_file = None
            return True
        except KeyError as e:
            LOG_ERROR(e)
            return False

    def __applyLensDistortion(self, xs, ys):
        xs2 = xs**2
        ys2 = ys**2
        xy = xs * ys
        r = xs2 + ys2
        rad_dist = r * (self.k[0] + self.k[1] * r)
        xs_out = xs * rad_dist + 2 * self.k[2] * xy + self.k[3] * (r + 2 * xs2)
        ys_out = ys * rad_dist + 2 * self.k[3] * xy + self.k[2] * (r + 2 * ys2)
        return xs_out, ys_out

    def pixelToRay(self, pts2d: np.ndarray | torch.Tensor) \
            -> np.ndarray | torch.Tensor:
        xs = (pts2d[0, :] - self.xc) / self.fx
        ys = (pts2d[1, :] - self.yc) / self.fy
        zs = ones_like(xs)
        if len(self.k) > 0:
            dx, dy = xs, ys
            for i in range(8):
                xs2, ys2 = self.__applyLensDistortion(dx, dy)
                dx, dy = xs - xs2, ys - ys2
            xs, ys = dx, dy
        ray = concat(
            (xs.reshape((1, -1)), ys.reshape((1, -1)), zs.reshape((1, -1))), 0)
        return ray

    def rayToPixel(self, pts3d: np.ndarray | torch.Tensor) \
            -> np.ndarray | torch.Tensor:
        zs = pts3d[2, :]
        xs = pts3d[0, :] / zs
        ys = pts3d[1, :] / zs
        if len(self.k) > 0:
            dx, dy = self.__applyLensDistortion(xs, ys)
            xs += dx
            ys += dy
        if self.alpha != 0:
            xs = self.fx * (xs + self.alpha * ys) + self.xc
        else:
            xs = self.fx * xs + self.xc
        ys = self.fy * ys + self.yc
        if self.flag == 0:
            return concat((xs.reshape((1, -1)), ys.reshape((1, -1))), 0)
        pix = concat((xs.reshape((1, -1)), ys.reshape((1, -1))), 0)
        pix[:, zs < 0] = -1.0
        if self.flag == 1:
            return pix

    @staticmethod
    def getPerspectiveModel(
            width: int, height: int, wfov_deg: float) -> PinholeModel:
        aspect = width / float(height)
        hfov_deg = wfov_deg / aspect
        model = PinholeModel()
        model.xc, model.yc = (width - 1) / 2.0, (height - 1) / 2.0
        model.fx = model.xc / tan(np.deg2rad(wfov_deg / 2.0))
        model.fy = model.yc / tan(np.deg2rad(hfov_deg / 2.0))
        model.max_theta = np.deg2rad(max(wfov_deg, hfov_deg) / 2.0)
        model.width, model.height = int(width), int(height)
        #print(model.xc, model.yc, model.fx, model.fy, model.max_theta, model.width, model.height)
        return model

class OpenCVFisheyeModel(CameraModel):
    def __init__(self):
        super().__init__()
        self.xc, self.yc, self.fx, self.fy = 0, 0, 1, 1
        self.k = np.zeros(0)
        self.max_theta = -1.0
        self.alpha = 0
        self.type = CameraType.OPENCV_FISHEYE

    @property
    def K(self):
        return np.array(
            [[self.fx, self.alpha, self.cx],
             [0, self.fy, self.cy],
             [0, 0, 1]])

    def LoadFromDict(self, dict: Dict) -> bool:
        if not super().LoadFromDict(dict): return False
        try:
            self.fx, self.fy = dict['focal_length']
            self.xc, self.yc = dict['center']
            self.k = dict['distortion']
            if 'alpha' in dict.keys():
                self.alpha = dict['alpha']
            self.max_theta = np.deg2rad(dict['max_fov'] / 2.0)
            if 'invalid_mask' in dict.keys():
                self.invalid_mask_file = dict['invalid_mask']
            else:
                self.invalid_mask_file = None
            return True
        except KeyError as e:
            LOG_ERROR(e)
            return False

    def pixelToRay(self, pts2d: np.ndarray | torch.Tensor) \
            -> np.ndarray | torch.Tensor:
        xs = (pts2d[0, :] - self.xc) / self.fx
        ys = (pts2d[1, :] - self.yc) / self.fy
        zs = ones_like(xs)
        k = self.k
        r = sqrt(xs**2 + ys**2)
        r[r < -np.pi/2] = -np.pi/2
        r[r > np.pi/2] = np.pi/2
        if isTorchArray(r):
            theta = r.clone()
        else:
            theta = r.copy()
        converged = zeros_like(r)
        converged[abs(r) <= 1e-8] = True
        for i in range(20):
            targets = converged == 0
            if targets.sum() == 0:
                break
            r_ = r[targets]
            th = theta[targets]
            th2 = th**2
            th_fix = \
                (th * (1 + th2 * (
                    k[0] + th2 * (k[1] + th2 * (k[2] + th2 * k[3])))) - r_) / \
                (1 + 3 * th2 * (
                    k[0] + 5 * th2 * (k[1] + 7 * th2 * (
                        k[2] + 9 * th2 * k[3]))))
            th = th - th_fix
            theta[targets] = th
            conv = abs(th_fix) <= 1e-8
            if isTorchArray(targets):
                targets[targets.clone()] = conv
            else:
                targets[targets] = conv
            converged[targets] = True
        scale = tan(theta) / (r + EPS)
        xs = xs * scale
        ys = ys * scale
        out = concat(
            (xs.reshape((1, -1)), ys.reshape((1, -1)), zs.reshape((1, -1))), 0)
        norm = sqrt((out**2).sum(0)).reshape((1, -1))
        out = out / norm
        invalid = logical_or(logical_not(converged),
            logical_and(r < 0, theta > 0), logical_and(r > 0, theta < 0))
        out[:, invalid] = np.nan
        return out

    def rayToPixel(self, pts3d: np.ndarray | torch.Tensor) \
            -> np.ndarray | torch.Tensor:
        zs = pts3d[2, :]
        xs = pts3d[0, :] / zs
        ys = pts3d[1, :] / zs
        k = self.k
        r = sqrt(xs**2 + ys**2)
        theta = atan(r)
        th2 = theta**2
        new_r = theta * (
            1 + th2 * (k[0] + th2 * (k[1] + th2 * (k[2] + th2 * k[3]))))
        new_r = new_r / (r + EPS)
        xs = new_r * xs
        ys = new_r * ys
        if self.alpha != 0:
            xs = self.fx * (xs + self.alpha * ys) + self.xc
        else:
            xs = self.fx * xs + self.xc
        ys = self.fy * ys + self.yc
        invalid = zs < 0
        xs[invalid] = -1.0
        ys[invalid] = -1.0
        return concat((xs.reshape(1, -1), ys.reshape(1, -1)), 0)

    def getRThetas(self, pix2ray=True, num_sample=500) \
            -> Tuple[np.ndarray, np.ndarray]:
        k = self.k
        if pix2ray:
            max_r = max(self.xc, self.width - 1 - self.xc) / self.fx
            step = max_r / (num_sample - 1)
            radii = np.arange(0, max_r + step, step)
            radii[radii < -np.pi/2] = -np.pi/2
            radii[radii > np.pi/2] = np.pi/2
            thetas = radii.copy()
            converged = zeros_like(radii)
            converged[abs(radii) <= 1e-8] = True
            for _ in range(20):
                targets = converged == 0
                if targets.sum() == 0:
                    break
                r_ = radii[targets]
                th = thetas[targets]
                th2 = th**2
                th_fix = \
                    (th * (1 + th2 * (
                        k[0] + th2 * (k[1] + th2 * (k[2] + th2 * k[3])))) - r_) / \
                    (1 + 3 * th2 * (
                        k[0] + 5 * th2 * (k[1] + 7 * th2 * (
                            k[2] + 9 * th2 * k[3]))))
                th = th - th_fix
                thetas[targets] = th
                conv = abs(th_fix) <= 1e-8
                if isTorchArray(targets):
                    targets[targets.clone()] = conv
                else:
                    targets[targets] = conv
                converged[targets] = True
            valid = thetas <= self.max_theta
            thetas = thetas[valid]
            radii = radii[valid]
            return radii * self.fx, thetas
        else:
            step = self.max_theta / (num_sample - 1)
            thetas = np.arange(0, self.max_theta + step, step)
            th2 = thetas**2
            new_r = thetas * (
                1 + th2 * (k[0] + th2 * (k[1] + th2 * (k[2] + th2 * k[3]))))
            radii = new_r * self.fx
            return radii, thetas


class UWFisheyeModel(CameraModel):

    def __init__(self):
        super().__init__()
        self.xc, self.yc, self.f = 0, 0, 1
        self.pol = np.zeros(0)
        self.invpol = np.zeros(0)
        self.max_theta = -1.0
        self.type = CameraType.UWFISHEYE

    def LoadFromDict(self, dict: Dict) -> bool:
        if not super().LoadFromDict(dict): return False
        try:
            num_pol = dict['poly'][0]
            if len(dict['poly']) - 1 != num_pol :
                LOG_WARNING(
                    'Number of coeffs does not match in ocam\'s poly')
            self.pol = dict['poly'][-1:0:-1] # make reverse
            self.pol.append(0)
            num_invpol = dict['inv_poly'][0]
            if len(dict['inv_poly']) - 1 != num_invpol :
                LOG_WARNING(
                    'Number of coeffs does not match in ocam\'s inv_poly')
            self.invpol = dict['inv_poly'][-1:0:-1] # make reverse
            self.invpol.append(0)
            self.f, self.xc, self.yc = dict['intrinsic']
            self.max_theta = np.deg2rad(dict['max_theta'])
            if 'invalid_mask' in dict.keys():
                self.invalid_mask_file = dict['invalid_mask']
            else:
                self.invalid_mask_file = None
            return True
        except KeyError as e:
            LOG_ERROR(e)
            return False

    def pixelToRay(self, pts2d: np.ndarray | torch.Tensor) \
            -> np.ndarray | torch.Tensor:
        x = (pts2d[0, :] - self.xc) / self.f
        y = (pts2d[1, :] - self.yc) / self.f
        r = sqrt(x**2 + y**2)
        theta = polyval(self.invpol, r)
        new_r = sin(theta) / (r + EPS)
        x = new_r * x.reshape((1, -1))
        y = new_r * y.reshape((1, -1))
        z = cos(theta).reshape((1, -1))
        rays = concat((x, y, z), 0)
        if self.max_theta > 0:
            rays[:, theta > self.max_theta] = np.nan
        return rays

    def rayToPixel(self, pts3d: np.ndarray | torch.Tensor) \
            -> np.ndarray | torch.Tensor:
        x = pts3d[0, :]
        y = pts3d[1, :]
        z = pts3d[2, :]
        r = sqrt(x**2 + y**2)
        theta = atan2(r, z)
        new_r = polyval(self.pol, theta).reshape((1, -1)) / r * self.f
        px = new_r * x.reshape((1, -1)) + self.xc
        py = new_r * y.reshape((1, -1)) + self.yc
        pix = concat((px, py), 0)
        if self.max_theta > 0:
            pix[:, theta > self.max_theta] = -1
        return pix

    def getRThetas(self, pix2ray=True, num_sample=500) \
            -> Tuple[np.ndarray, np.ndarray]:
        if pix2ray:
            max_x = max(self.xc, self.width - 1 - self.xc)
            max_y = max(self.yc, self.height - 1 - self.yc)
            max_r = sqrt(max_x**2 + max_y**2)
            step = max_r / (num_sample - 1)
            radii = np.arange(0, max_r + step, step)
            thetas = polyval(self.invpol, radii / self.f)
            valid = thetas <= self.max_theta
            thetas = thetas[valid]
            radii = radii[valid]
            return radii, thetas
        else:
            step = self.max_theta / (num_sample - 1)
            thetas = np.arange(0, self.max_theta + step, step)
            radii = polyval(self.pol, thetas) * self.f
            return radii, thetas

    @staticmethod
    def getEquiDistantModel(width: int, max_fov_deg: float) -> UWFisheyeModel:
        model = UWFisheyeModel()
        fov_2 = np.deg2rad(max_fov_deg / 2.0)
        xc = (width - 1) / 2.0
        model.f = model.xc = model.yc = xc
        model.max_theta = fov_2
        model.invpol = np.array([fov_2, 0], dtype=np.float64)
        model.pol = np.array([1.0 / fov_2, 0], dtype=np.float64)
        model.width = model.height = width
        return model

class OcamModel(CameraModel):
    def __init__(self):
        super().__init__()
        self.xc, self.yc = 0, 0
        self.c, self.d, self.e = 1, 0, 0
        self.max_theta = np.pi
        self.pol = np.zeros(0)
        self.inv_pol = np.zeros(0)
        self.type = CameraType.OCAM

    def LoadFromDict(self, dict: Dict) -> bool:
        if not super().LoadFromDict(dict): return False
        try:
            num_pol = dict['poly'][0]
            if len(dict['poly']) - 1 != num_pol :
                LOG_WARNING(
                    'Number of coeffs does not match in ocam\'s poly')
            self.pol = dict['poly'][-1:0:-1] # make reverse
            num_invpol = dict['inv_poly'][0]
            if len(dict['inv_poly']) - 1 != num_invpol :
                LOG_WARNING(
                    'Number of coeffs does not match in ocam\'s inv_poly')
            self.inv_pol = dict['inv_poly'][-1:0:-1] # make reverse
            self.xc, self.yc = dict['center'] # x, y fliped
            self.c, self.d, self.e = dict['affine']
            self.max_theta = np.deg2rad(dict['max_fov']) / 2.0
            if 'invalid_mask' in dict.keys():
                self.invalid_mask_file = dict['invalid_mask']
            else:
                self.invalid_mask_file = None
            return True
        except KeyError as e:
            LOG_ERROR(e)
            return False

    def pixelToRay(self, pts2d: np.ndarray | torch.Tensor) \
            -> np.ndarray | torch.Tensor:
        return self.pixelToRay(pts2d, False, self.max_theta)
    def rayToPixel(self, pts3d: np.ndarray | torch.Tensor) \
            -> np.ndarray | torch.Tensor:
        return self.rayToPixel(pts3d, False, self.max_theta, True)

    def pixelToRay(self, pts2d, out_theta = False, max_theta = None):
        if max_theta is None: max_theta = self.max_theta
        # flip axis
        x = pts2d[1,:].reshape((1, -1)) - self.xc
        y = pts2d[0,:].reshape((1, -1)) - self.yc
        p = concat((x, y), axis=0)
        invdet = 1.0 / (self.c - self.d * self.e)
        A_inv = invdet * np.array([
            [      1, -self.d],
            [-self.e,  self.c]])
        p = A_inv.dot(p)
        # flip axis
        x = p[1,:].reshape((1, -1))
        y = p[0,:].reshape((1, -1))
        rho = sqrt(x * x + y * y)
        z = polyval(self.pol, rho).reshape((1, -1))
        # theta is angle from the optical axis.
        theta = atan2(rho, -z)
        out = concat((x, y, -z), axis=0)
        norm = sqrt((out**2).sum(0)).reshape((1, -1))
        out = out / norm
        out[:,theta.squeeze() > max_theta] = np.nan
        if out_theta:
            return out, theta
        else:
            return out
    # end pixelToRay

    def rayToPixel(self, pts3d, out_theta = False, max_theta = None,
            use_invalid_mask = True):
        if max_theta is None: max_theta = self.max_theta
        norm = sqrt(pts3d[0,:]**2 + pts3d[1,:]**2) + EPS
        theta = atan2(-pts3d[2,:], norm)
        rho = polyval(self.inv_pol, theta)
        # max_theta check : theta is the angle from xy-plane in ocam,
        # thus add pi/2 to compute the angle from the optical axis.
        theta = theta + np.pi / 2
        # flip axis
        x = pts3d[1,:] / norm * rho
        y = pts3d[0,:] / norm * rho
        x2 = x * self.c + y * self.d + self.xc
        y2 = x * self.e + y          + self.yc
        x2 = x2.reshape((1, -1))
        y2 = y2.reshape((1, -1))
        out = concat((y2, x2), axis=0)
        out[:, isnan(pts3d[0,:])] = -1.0
        out[:, theta.squeeze() > max_theta] = -1.0
        if use_invalid_mask and self.invalid_mask is not None:
            hv, wv = self.invalid_mask.shape
            invalid_mask = self.invalid_mask.flatten()
            px = (y2 * wv / self.width).round().squeeze()
            py = (x2 * hv / self.height).round().squeeze()
            if isTorchArray(pts3d):
                invalid_mask = torch.tensor(invalid_mask, device=pts3d.device)
                px, py = px.long(), py.long()
            else:
                px, py = px.astype(np.int32), py.astype(np.int32)
            is_in_image = logical_and(
                px >= 0, px < wv, py >= 0, py < hv)
            idxs = px[is_in_image] + py[is_in_image] * wv
            # if type(invalid_mask) == torch.Tensor:
            #     is_in_image[is_in_image] = (invalid_mask[idxs] > 0).clone()
            # else:
            if isTorchArray(is_in_image):
                is_in_image[is_in_image.clone()] = invalid_mask[idxs] > 0
            else:
                is_in_image[is_in_image] = invalid_mask[idxs] > 0
            out[:, is_in_image] = -1.0

        if out_theta:
            return out, theta
        else:
            return out
    # end rayToPixel

    def getRThetas(self, pix2ray=True, num_sample=500) \
            -> Tuple[np.ndarray, np.ndarray]:
        if pix2ray:
            max_x = max(self.xc, self.width - 1 - self.xc)
            max_y = max(self.yc, self.height - 1 - self.yc)
            max_r = sqrt(max_x**2 + max_y**2)
            step = max_r / (num_sample - 1)
            radii = np.arange(0, max_r + step, step)
            zs = polyval(self.pol, radii)
            thetas = atan2(radii, -zs)
            valid = thetas <= self.max_theta
            thetas = thetas[valid]
            radii = radii[valid]
            return radii, thetas
        else:
            step = self.max_theta / (num_sample - 1)
            thetas = np.arange(0, self.max_theta + step, step)
            radii = polyval(self.inv_pol, thetas - (np.pi / 2))
            return radii, thetas

def loadCameraListFromYAML(path: str) -> List[CameraModel]:
    config = yaml.safe_load(open(path))
    cams = []
    is_capture = not 'dataset' in config.keys()
    if is_capture:
        nodes = config['sensor_nodes']
        for n in nodes:
            type_str = config[n]['type']
            if type_str == 'camera_multi_files' or type_str == 'multi_camera':
                cam_node_name = n
                break
        cam_node = config[cam_node_name]
        cam_keys = cam_node['sensor_nodes']
        calib_splitted = 'calib' in config.keys()
        for k in cam_keys:
            if calib_splitted:
                val = config['calib'][cam_node_name][k]
            else:
                val = cam_node[k]['calib']
            cam = CameraModel.CreateFromDict(val)
            cams.append(cam)
    else:
        cameras_cfg = config['cameras']
        for i in range(len(cameras_cfg)):
            cam = CameraModel.CreateFromDict(cameras_cfg[i])
            cams.append(cam)
    for cam in cams:
        invalid_mask = None
        yaml_dir, _ = osp.split(path)
        if cam.invalid_mask_file is not None:
            mask_file = osp.join(yaml_dir, cam.invalid_mask_file)
            if osp.exists(mask_file):
                invalid_mask = readImage(mask_file).astype(np.bool)
        cam.invalid_mask = cam.makeFoVMask(invalid_mask)
        morph_filter = np.ones((5, 5), dtype=np.uint8)
        cam.invalid_mask = ndimage.binary_closing(
            cam.invalid_mask, morph_filter, border_value=1)
    return cams