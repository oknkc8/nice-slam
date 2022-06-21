from curses import raw
import os
import time

import cv2
import numpy as np
import torch
from colorama import Fore, Style
from torch.autograd import Variable
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data.dataloader import DataLoader

from src.omnimvs_src.omnimvs import *
from src.omnimvs_src.module.network import OmniMVSNet
from src.omnimvs_src.module.network_fast import OmniMVSNet as OmniMVSNetFast
from src.omnimvs_src.module.loss_functions import *

from src.common import (get_camera_from_tensor, get_samples, get_samples_omni,
                        get_tensor_from_camera, random_select)
from src.utils.datasets import get_dataset
from src.utils.Visualizer import Visualizer

import matplotlib.pyplot as plt

import pdb


class Integrator(object):
    """
    Mapper thread. Note that coarse mapper also uses this code.

    """

    def __init__(self, cfg, args, slam,
                 ):

        self.cfg = cfg
        self.args = args

        self.idx = slam.idx
        self.nice = slam.nice
        self.c = slam.shared_c
        self.bound = slam.bound
        self.logger = slam.logger
        self.output = slam.output
        self.verbose = slam.verbose
        self.low_gpu_mem = slam.low_gpu_mem
        self.mapping_idx = slam.mapping_idx
        self.mapping_cnt = slam.mapping_cnt
        self.decoders = slam.shared_decoders
        self.estimate_c2w_list = slam.estimate_c2w_list
        self.mapping_first_frame = slam.mapping_first_frame
        # self.summary_writer = slam.summary_writer
        self.logdir = slam.logdir

        self.device = cfg['omnimvs']['device']
        self.scale = cfg['scale']
        self.coarse = cfg['coarse']
        self.occupancy = cfg['occupancy']
        self.sync_method = cfg['sync_method']
        
        # self.frame_reader = get_dataset(
        #     cfg, args, self.scale, device=self.device)
        # self.n_img = len(self.frame_reader)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy
        self.cam_method = slam.cam_method
        self.phi_deg, self.phi_max_deg = slam.phi_deg, slam.phi_max_deg
        
        self.target_depths = slam.target_depths
        self.target_colors = slam.target_colors
        self.target_entropys = slam.target_entropys
        
        opts = Edict(cfg['omnimvs'])
        opts.omnimvs_opts.equirect_size = tuple(opts.omnimvs_opts.equirect_size)
        self.equirect_size = opts.omnimvs_opts.equirect_size
        self.omnimvs = OmniMVS(opts.dbname, opts=opts.omnimvs_opts, train=not(opts.net_freeze), db_config_idxs=opts.db_config_idxs, device_id=self.device)
        self.dbloader = DataLoader(self.omnimvs, shuffle=False, num_workers=0, batch_size=1, 
                                   collate_fn=BatchCollator.collate, pin_memory=True)
        self.mvsnet = OmniMVSNet(opts.net_opts).to(self.device)
        
        if not osp.exists(opts.snapshot_path):
            sys.exit('%s does not exsits' % (opts.snapshot_path))
        snapshot = torch.load(opts.snapshot_path)
        self.mvsnet.load_state_dict(snapshot['net_state_dict'])
        
        _Conv3D = HorizontalCircularConv3D if opts.net_opts.circular_pad else Conv3D
        CH = 32
        self.feature_fusion = _Conv3D(2*CH, CH, 1, 1, 1)
        
        self.grid_cams = buildIndividualLookupTable(self.omnimvs.ocams,
                                self.omnimvs.num_invdepth, self.omnimvs.min_invdepth, self.omnimvs.step_invdepth, self.omnimvs)
        masks = []
        for cam_idx in range(1,5,1):
            mask_img = cv2.imread('./src/omni_utils/mask%d_%d.png'%(cam_idx, self.equirect_size[1]), cv2.IMREAD_GRAYSCALE)
            mask = mask_img==255
            masks.append(torch.tensor(mask).to(self.device))
        self.non_zero_area = torch.stack(masks, dim=0)

    def run_omnimvs(self, data):        
        index, entropy, prob, resp, feats = self.omnimvs.forwardNetwork(
                                            self.mvsnet, data.imgs, data.sub_db_idxs, False)
        
        invdepth = self.omnimvs.indexToInvdepth(index)
        view_invdepth = self.omnimvs.respToCameraInvdepth(resp, grids_cam=self.grid_cams)
        view_idxs, _ = self.omnimvs.getPanoramaViewdepthIndex(view_invdepth, invdepth)
        
        ref_idxs = self.omnimvs.getRefPanorama(invdepth)
        ref_idxs = torch.stack(ref_idxs, dim=0)
        view_idxs = torch.stack(view_idxs, dim=0)
        
        origin_imgs = self.omnimvs.getRGBViewPanorama(data.raw_imgs, invdepth)
        sigma = 3
        
        weight = torch.zeros((4, self.equirect_size[0], self.equirect_size[1])).to(self.device)
        
        origin_imgs = torch.stack(origin_imgs, dim=0)
        
        weight[self.non_zero_area] = torch.exp(-((ref_idxs[self.non_zero_area] - view_idxs[self.non_zero_area]) / sigma) ** 2)
        weight_normed = torch.divide(weight, torch.sum(weight, dim=0))
        weight_normed = weight_normed.unsqueeze(1)
        
        color = origin_imgs * weight_normed
        color = torch.sum(color, dim=0)
        depth = 1.0 / invdepth
        raw_feat, geo_feat = feats
        
        color[isnan(color)] = 0
        
        return color, depth, raw_feat, geo_feat, prob, entropy

    def integrate_features(self, raw_feat, geo_feat, prob):
        
        pdb.set_trace()
        
        raw_feat = F.interpolate(raw_feat, scale_factor=2, mode='trilinear', 
                                 align_corners=True)
        geo_feat = F.interpolate(geo_feat, scale_factor=2, mode='trilinear', 
                                 align_corners=True)
        raw_feat = raw_feat * prob.unsqueeze(1)
        concat_feat = torch.cat([raw_feat, geo_feat], dim=1)
        
        cam_feature = self.fusion()
        
        print('ddd')
        
    

    def run(self):
        """
        OmniMVS에서 해야될 것
            1. 4개의 이미지를 넣어서 pano depth를 생성
            2. pano depth로부터 pano rgb 생성
            3. 
        """
        
        for i, data in enumerate(self.dbloader):
            print('Iter:', i)
            color, depth, raw_feat, geo_feat, prob, entropy = self.run_omnimvs(data)
            
            self.target_colors.append(color)
            self.target_depths.append(depth)
            self.target_entropys.append(entropy)
            
            self.integrate_features(raw_feat, geo_feat, prob)
            
            break
        
        
        self.summary_writer = SummaryWriter(self.logdir)
        cfg = self.cfg
        idx, gt_color, gt_depth, gt_c2w = self.frame_reader[0]

        self.estimate_c2w_list[0] = gt_c2w.cpu()
        init = True
        prev_idx = -1
        self.global_iter = 0
        while (1):
            while True:
                idx = self.idx[0].clone()
                if idx == self.n_img-1:
                    break
                if self.sync_method == 'strict':
                    if idx % self.every_frame == 0 and idx != prev_idx:
                        break

                elif self.sync_method == 'loose':
                    if idx == 0 or idx >= prev_idx+self.every_frame//2:
                        break
                elif self.sync_method == 'free':
                    break
                time.sleep(0.1)
            prev_idx = idx

            if self.verbose:
                print(Fore.GREEN)
                prefix = 'Coarse ' if self.coarse_mapper else ''
                print(prefix+"Mapping Frame ", idx.item())
                print(Style.RESET_ALL)

            _, gt_color, gt_depth, gt_c2w = self.frame_reader[idx]

            if not init:
                lr_factor = cfg['mapping']['lr_factor']
                num_joint_iters = cfg['mapping']['iters']

                # here provides a color refinement postprocess
                if idx == self.n_img-1 and self.color_refine and not self.coarse_mapper:
                    outer_joint_iters = 5
                    self.mapping_window_size *= 2
                    self.middle_iter_ratio = 0.0
                    self.fine_iter_ratio = 0.0
                    num_joint_iters *= 5
                    self.fix_color = True
                    self.frustum_feature_selection = False
                else:
                    if self.nice:
                        outer_joint_iters = 1
                    else:
                        outer_joint_iters = 3

            else:
                outer_joint_iters = 1
                lr_factor = cfg['mapping']['lr_first_factor']
                num_joint_iters = cfg['mapping']['iters_first']

            cur_c2w = self.estimate_c2w_list[idx].to(self.device)
            num_joint_iters = num_joint_iters//outer_joint_iters
            for outer_joint_iter in range(outer_joint_iters):

                self.BA = (len(self.keyframe_list) > 4) and cfg['mapping']['BA'] and (
                    not self.coarse_mapper)

                _ = self.optimize_map(num_joint_iters, lr_factor, idx, gt_color, gt_depth,
                                      gt_c2w, self.keyframe_dict, self.keyframe_list, cur_c2w=cur_c2w)
                if self.BA:
                    cur_c2w = _
                    self.estimate_c2w_list[idx] = cur_c2w

                # add new frame to keyframe set
                if outer_joint_iter == outer_joint_iters-1:
                    if (idx % self.keyframe_every == 0 or (idx == self.n_img-2)) \
                            and (idx not in self.keyframe_list):
                        self.keyframe_list.append(idx)
                        self.keyframe_dict.append({'gt_c2w': gt_c2w.cpu(), 'idx': idx, 'color': gt_color.cpu(
                        ), 'depth': gt_depth.cpu(), 'est_c2w': cur_c2w.clone()})

            if self.low_gpu_mem:
                torch.cuda.empty_cache()

            init = False
            # mapping of first frame is done, can begin tracking
            self.mapping_first_frame[0] = 1

            if not self.coarse_mapper:
                if ((not (idx == 0 and self.no_log_on_first_frame)) and idx % self.ckpt_freq == 0) \
                        or idx == self.n_img-1:
                    self.logger.log(idx, self.keyframe_dict, self.keyframe_list,
                                    selected_keyframes=self.selected_keyframes
                                    if self.save_selected_keyframes_info else None)

                self.mapping_idx[0] = idx
                self.mapping_cnt[0] += 1

                if (idx % self.mesh_freq == 0) and (not (idx == 0 and self.no_mesh_on_first_frame)):
                    mesh_out_file = f'{self.output}/mesh/{idx:05d}_mesh.ply'
                    self.mesher.get_mesh(mesh_out_file, self.c, self.decoders, self.keyframe_dict, self.estimate_c2w_list,
                                         idx,  self.device, show_forecast=self.mesh_coarse_level,
                                         clean_mesh=self.clean_mesh, get_mask_use_all_frames=False)

                if idx == self.n_img-1:
                    mesh_out_file = f'{self.output}/mesh/final_mesh.ply'
                    self.mesher.get_mesh(mesh_out_file, self.c, self.decoders, self.keyframe_dict, self.estimate_c2w_list,
                                         idx,  self.device, show_forecast=self.mesh_coarse_level,
                                         clean_mesh=self.clean_mesh, get_mask_use_all_frames=False)
                    os.system(
                        f"cp {mesh_out_file} {self.output}/mesh/{idx:05d}_mesh.ply")
                    if self.eval_rec:
                        mesh_out_file = f'{self.output}/mesh/final_mesh_eval_rec.ply'
                        self.mesher.get_mesh(mesh_out_file, self.c, self.decoders, self.keyframe_dict,
                                             self.estimate_c2w_list, idx, self.device, show_forecast=False,
                                             clean_mesh=self.clean_mesh, get_mask_use_all_frames=True)
                    break

            if idx == self.n_img-1:
                break
