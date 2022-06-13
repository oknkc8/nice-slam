# utils.dbhelper.py
#
# Author: Changhee Won (chwon@hanyang.ac.kr)
#
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import annotations
import enum
from typing import *

from src.omni_utils.common import *
from src.omni_utils.camera import *

import yaml
from scipy import ndimage

from src.omni_utils.log import LOG_INFO, LOG_ERROR, LOG_WARNING
import src.omni_utils.image as utils_im
# import utils.ocam

def loadDBConfigs(dbname: str, dbpath: str, opts: Edict) \
            -> Tuple[Edict, List[CameraModel]]:
    ocams = []
    config_file = osp.join(dbpath, 'config.yaml')
    capture_file = osp.join(dbpath, 'capture.yaml')
    if osp.exists(config_file):
        config = yaml.safe_load(open(config_file))

        for k in config['config'].keys(): opts[k] = config['config'][k]
        for k in config['dataset'].keys(): opts[k] = config['dataset'][k]

        cameras_cfg = config['cameras']
        for i in range(len(cameras_cfg)):
            # ocam = utils.ocam.OcamModel(cameras_cfg[i])
            ocam = CameraModel.CreateFromDict(cameras_cfg[i])
            ocams.append(ocam)
    elif osp.exists(capture_file):
        config = yaml.safe_load(open(capture_file))
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
            ocam = CameraModel.CreateFromDict(val)
            ocams.append(ocam)
        if type_str == 'multi_camera':
            opts.dtype = 'capture'
            opts.cam_node_name = cam_node_name
    else:
        sys.exit('No yaml files found in: %s' % dbpath)

    for ocam in ocams:
        invalid_mask = None
        if ocam.invalid_mask_file is not None:
            mask_file = osp.join(dbpath, ocam.invalid_mask_file)
            if osp.exists(mask_file):
                invalid_mask = utils_im.readImage(mask_file).astype(np.bool)
        ocam.invalid_mask = ocam.makeFoVMask(invalid_mask)
        morph_filter = np.ones((5, 5), dtype=np.uint8)
        ocam.invalid_mask = ndimage.binary_closing(
            ocam.invalid_mask, morph_filter, border_value=1)

    func = '__load_train_%s(opts, dbpath)' % (dbname)
    try:
        opts = eval(func)
        LOG_INFO('Found "%s" training configs' % (dbname))
        opts.dtype = 'gt'
        opts.mid_depth = opts.min_depth * 15
    except NameError:
        LOG_INFO(
            'Training configs not found "%s". see "utils/dbhelper.py"' % (
                dbname))
    finally:
        return opts, ocams

def __load_train_sunny(opts: Edict, dbpath: str) -> Edict:
    opts.train_idx = list(range(1, 701))
    opts.test_idx = list(range(701, 1001))
    opts.gt_phi = 45
    opts.min_depth = 1.65
    return opts

__load_train_sunset = __load_train_cloudy = __load_train_sunny

def __load_train_omnithings(opts: Edict, dbpath: str) -> Edict:
    opts.train_idx = list(range(1, 4097)) + list(range(5121, 10241))
    opts.test_idx = list(range(4097, 5121))
    opts.gt_phi = 90
    opts.min_depth = 1.65
    return opts

def __load_train_omnihouse(opts: Edict, dbpath: str) -> Edict:
    opts.train_idx = list(range(1, 2049))
    opts.test_idx = list(range(2049, 2561))
    opts.gt_phi = 90
    opts.min_depth = 1.65
    return opts

def __load_train_omnihouse2(opts: Edict, dbpath: str) -> Edict:
    opts.train_idx = list(range(1, 2301))
    opts.test_idx = list(range(2301, 2813))
    opts.gt_phi = 90
    opts.min_depth = 1.65
    return opts

def __load_train_omnithings2(opts: Edict, dbpath: str) -> Edict:
    opts.train_idx = list(range(1, 9217))
    opts.test_idx = list(range(9217, 10241))
    opts.gt_phi = 90
    opts.min_depth = 1.65
    return opts

def __load_train_omnithings3(opts: Edict, dbpath: str) -> Edict:
    opts.gt_depth_fmt = 'cam_center_square_%d_depth/%05d.tiff'
    opts.train_idx = list(range(0, 8000))
    opts.test_idx = list(range(8000, 9000))
    opts.gt_phi = 90
    opts.min_depth = 0.3
    return opts


def __load_train_scenecity_drone_randomsky(opts: Edict, dbpath: str) -> Edict:
    opts.gt_depth_fmt = 'cam_center_square_%d_depth/%05d.tiff'
    opts.train_idx = list(range(1, 2049))
    opts.test_idx = list(range(1, 2049, 8))
    opts.gt_phi = 90
    opts.min_depth = 0.3
    return opts

def __load_train_scenecity_car(opts: Edict, dbpath: str) -> Edict:
    opts.gt_depth_fmt = 'cam_center1_car_%d_depth/%05d.tiff'
    opts.train_idx = list(range(1, 2345))
    opts.test_idx = list(range(1, 2345, 8))
    opts.train_idx = list(set(opts.train_idx) - set(opts.test_idx))
    opts.gt_phi = 90
    opts.use_rounded_cuboid_sweep = True
    return opts


def __load_train_omnithings_car(opts: Edict, dbpath: str) -> Edict:
    opts.gt_depth_fmt = 'cam_center1_car_%d_depth/%05d.tiff'
    opts.train_idx = list(range(0, 9216))
    opts.test_idx = list(range(9216, 10240))
    opts.train_idx = list(set(opts.train_idx) - set(opts.test_idx))
    opts.gt_phi = 90
    opts.use_rounded_cuboid_sweep = True
    return opts

"""def __load_train_scenecity_randomsky_car(opts: Edict, dbpath: str) -> Edict:
    opts.gt_depth_fmt = 'cam_center1_car_%d_depth/%05d.tiff'
    opts.train_idx = list(range(1, 2345))
    opts.test_idx = list(range(1, 2345, 8))
    opts.train_idx = list(set(opts.train_idx) - set(opts.test_idx))
    opts.gt_phi = 90
    opts.use_rounded_cuboid_sweep = True
    return opts"""

def __load_train_scenecity_rotate_randomsky_car(opts: Edict, dbpath: str) -> Edict:
    opts.gt_depth_fmt = 'cam_center1_car_%d_depth/%05d.tiff'
    opts.train_idx = list(range(1, 2345))
    opts.test_idx = list(range(1, 2345, 8))
    opts.train_idx = list(set(opts.train_idx) - set(opts.test_idx))
    opts.gt_phi = 90
    opts.use_rounded_cuboid_sweep = True
    return opts

def __load_train_omnihouse_car(opts: Edict, dbpath: str) -> Edict:
    opts.gt_depth_fmt = 'cam_center1_car_%d_depth/%05d.tiff'
    opts.train_idx = list(range(512, 3512))
    opts.test_idx = list(range(0, 512))
    opts.train_idx = list(set(opts.train_idx) - set(opts.test_idx))
    opts.gt_phi = 90
    opts.use_rounded_cuboid_sweep = True
    return opts

def __load_train_garage2_car(opts: Edict, dbpath: str) -> Edict:
    opts.gt_depth_fmt = 'cam_center1_car_%d_depth/%05d.tiff'
    opts.train_idx = list(range(0, 1920))
    opts.test_idx = list(range(1920, 2048))
    opts.train_idx = list(set(opts.train_idx) - set(opts.test_idx))
    opts.gt_phi = 90
    opts.use_rounded_cuboid_sweep = True
    return opts

def __load_train_under_parking_car(opts: Edict, dbpath: str) -> Edict:
    opts.gt_depth_fmt = 'cam_center1_car_%d_depth/%05d.tiff'
    opts.train_idx = list(range(1, 4481))
    opts.test_idx = list(range(4481, 5601))
    opts.train_idx = list(set(opts.train_idx) - set(opts.test_idx))
    opts.gt_phi = 90
    opts.use_rounded_cuboid_sweep = True
    return opts

def __load_train_outdoor_parking_car(opts: Edict, dbpath: str) -> Edict:
    opts.gt_depth_fmt = 'cam_center1_car_%d_depth/%05d.tiff'
    opts.train_idx = list(range(1, 3001))
    opts.test_idx = list(range(1, 3001, 10))
    opts.train_idx = list(set(opts.train_idx) - set(opts.test_idx))
    opts.gt_phi = 90
    opts.use_rounded_cuboid_sweep = True
    return opts

def __load_train_201014_skt_lobby_day(opts: Edict, dbpath: str) -> Edict:
    opts = __load_lidar_gt_fidxs(opts, dbpath)
    opts.min_depth = 0.35
    return opts

def __load_train_201014_skt_3f_day(opts: Edict, dbpath: str) -> Edict:
    opts = __load_lidar_gt_fidxs(opts, dbpath)
    opts.min_depth = 0.35
    return opts

def __load_train_201014_skt_lobby_night(opts: Edict, dbpath: str) -> Edict:
    opts = __load_lidar_gt_fidxs(opts, dbpath)
    opts.min_depth = 0.35
    return opts

def __load_train_201027_tips_s6(opts: Edict, dbpath: str) -> Edict:
    opts = __load_lidar_gt_fidxs(opts, dbpath)
    return opts

def __load_train_201029_coex_s1(opts: Edict, dbpath: str) -> Edict:
    opts = __load_lidar_gt_fidxs(opts, dbpath)
    opts.min_depth = 0.3
    return opts

def __load_train_201029_coex_s8(opts: Edict, dbpath: str) -> Edict:
    opts = __load_lidar_gt_fidxs(opts, dbpath)
    opts.min_depth = 0.3
    return opts

"""def __load_train_201029_coex_s9(opts: Edict, dbpath: str) -> Edict:
    #opts = __load_lidar_gt_fidxs(opts, dbpath)
    opts.start = 0
    opts.end = 4061
    opts.train_idx = list(range(0, 3654))
    opts.test_idx = list(range(3654, 4061))   
    opts.min_depth = 0.3
    return opts

def __load_train_210624_xingxing_block6(opts: Edict, dbpath: str) -> Edict:
    #opts = __load_lidar_gt_fidxs(opts, dbpath)
    opts.start = 0
    opts.end = 25985
    opts.step = 5   
    opts.train_idx = list(range(0, 23386 ,5))
    opts.test_idx = list(range(23386, 25985, 5))
    opts.min_depth = 0.3
    return opts"""

def __load_train_201118_tips_s6(opts: Edict, dbpath: str) -> Edict:
    opts = __load_lidar_gt_fidxs(opts, dbpath)
    opts.min_depth = 0.3
    return opts

def __load_lidar_gt_fidxs(opts: Edict, dbpath: str) -> Edict:
    opts.gt_depth_fmt = 'omnidepth_lidar_%d_motion3/%05d.tiff'
    opts.gt_fidxs = np.loadtxt(
        osp.join(dbpath, 'lidar_gt_fidxs3.txt'), dtype=int)
    opts.train_idx = list(opts.gt_fidxs)
    opts.test_idx = list(opts.gt_fidxs[::5])
    opts.train_idx = list(set(opts.train_idx) - set(opts.test_idx) - set(list(range(1000))))
    opts.gt_phi = 45
    return opts

# def __load_train_210330_yeoksam(opts: Edict, dbpath: str) -> Edict:
#     opts.train_idx = []
#     opts.test_idx = list(range(0, 22393))
#     opts.dtype = 'capture'
#     return opts

