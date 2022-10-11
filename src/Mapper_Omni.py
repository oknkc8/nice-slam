import os
import time
import gc

import cv2
import numpy as np
import torch
from colorama import Fore, Style
from torch.autograd import Variable
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torchsparse.tensor import PointTensor, SparseTensor
from torch.nn.functional import grid_sample

from src.common import (get_camera_from_tensor, get_samples, get_samples_omni,
                        get_tensor_from_camera, random_select, sparse_to_dense, grid_sample_3d)
from src.utils.datasets import get_dataset
from src.utils.Visualizer import Visualizer
from src.omnimvs_src.module.network import CostFusion, CostFusion_tmp, ConvGRU, FeatureNet_Panorama
from src.omni_utils.common import EPS

import pdb

# torch.autograd.set_detect_anomaly(True)

class Mapper(object):
    """
    Mapper thread. Note that coarse mapper also uses this code.

    """

    def __init__(self, cfg, args, slam, coarse_mapper=False
                 ):

        self.cfg = cfg
        self.args = args
        self.coarse_mapper = coarse_mapper

        self.idx = slam.idx
        self.method = slam.method
        self.c = slam.shared_c
        self.coord = slam.shared_coord
        self.bound = slam.bound
        self.logger = slam.logger
        self.mesher = slam.mesher
        self.output = slam.output
        self.verbose = slam.verbose
        self.renderer = slam.renderer
        self.low_gpu_mem = slam.low_gpu_mem
        self.mapping_idx = slam.mapping_idx
        self.mapping_cnt = slam.mapping_cnt
        self.decoders = slam.shared_decoders
        self.estimate_c2w_list = slam.estimate_c2w_list
        self.mapping_first_frame = slam.mapping_first_frame
        # self.summary_writer = slam.summary_writer
        self.logdir = slam.logdir
        self.n_img = slam.n_img
        
        self.scale = cfg['scale']
        self.coarse = cfg['coarse']
        self.occupancy = cfg['occupancy']
        self.sync_method = cfg['sync_method']

        self.device = cfg['mapping']['device']
        self.fix_fine = cfg['mapping']['fix_fine']
        self.eval_rec = cfg['meshing']['eval_rec']
        self.BA = False  # Even if BA is enabled, it starts only when there are at least 4 keyframes
        self.BA_cam_lr = cfg['mapping']['BA_cam_lr']
        self.mesh_freq = cfg['mapping']['mesh_freq']
        self.ckpt_freq = cfg['mapping']['ckpt_freq']
        self.fix_color = cfg['mapping']['fix_color']
        self.mapping_pixels = cfg['mapping']['pixels']
        self.num_joint_iters = cfg['mapping']['iters']
        self.clean_mesh = cfg['meshing']['clean_mesh']
        self.every_frame = cfg['mapping']['every_frame']
        self.color_refine = cfg['mapping']['color_refine']
        self.w_color_loss = cfg['mapping']['w_color_loss']
        self.w_geo_loss = cfg['mapping']['w_geo_loss']
        self.keyframe_every = cfg['mapping']['keyframe_every']
        self.fine_iter_ratio = cfg['mapping']['fine_iter_ratio']
        self.middle_iter_ratio = cfg['mapping']['middle_iter_ratio']
        self.mesh_coarse_level = cfg['meshing']['mesh_coarse_level']
        self.mapping_window_size = cfg['mapping']['mapping_window_size']
        self.no_vis_on_first_frame = cfg['mapping']['no_vis_on_first_frame']
        self.no_log_on_first_frame = cfg['mapping']['no_log_on_first_frame']
        self.no_mesh_on_first_frame = cfg['mapping']['no_mesh_on_first_frame']
        self.frustum_feature_selection = cfg['mapping']['frustum_feature_selection']
        self.keyframe_selection_method = cfg['mapping']['keyframe_selection_method']
        self.save_selected_keyframes_info = cfg['mapping']['save_selected_keyframes_info']
        self.n_frames = cfg['omnimvs']['n_frames']
        if self.save_selected_keyframes_info:
            self.selected_keyframes = {}

        if self.method == 'nice':
            if coarse_mapper:
                self.keyframe_selection_method = 'global'

        self.keyframe_dict = []
        self.keyframe_list = []
        # if 'Demo' not in self.output:  # disable this visualization in demo
        self.visualizer = Visualizer(freq=cfg['mapping']['vis_freq'], inside_freq=cfg['mapping']['vis_inside_freq'],
                                        vis_dir=os.path.join(self.output, 'mapping_vis'), renderer=self.renderer,
                                        verbose=self.verbose, device=self.device)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy
        self.cam_method = slam.cam_method
        self.phi_deg, self.phi_max_deg = slam.phi_deg, slam.phi_max_deg
        
        self.integrator = slam.integrator
        self.fusion_method = self.cfg['model']['fusion_method']
        self.costfusion = {}
        # self.costfusion['grid_middle'] = SPVCNN(ch_in=64, ch_out=self.cfg['model']['c_dim']).to(self.integrator.device)
        # self.costfusion['grid_fine'] = SPVCNN(ch_in=64, ch_out=self.cfg['model']['c_dim']).to(self.integrator.device)
        # self.costfusion['grid_color'] = SPVCNN(ch_in=64, ch_out=self.cfg['model']['c_dim']).to(self.integrator.device)
        # self.costfusion['grid_middle'] = CostFusion(ch_in=64, ch_out=self.cfg['model']['c_dim']).to(self.device)
        # self.costfusion['grid_fine'] = CostFusion(ch_in=64, ch_out=self.cfg['model']['c_dim']).to(self.device)
        # self.costfusion['grid_color'] = CostFusion(ch_in=64, ch_out=self.cfg['model']['c_dim']).to(self.device)
        ch_in = 64
        if self.fusion_method == 'variance':
            ch_in *= 2
        self.costfusion['grid_middle'] = CostFusion_tmp(ch_in=ch_in, ch_out=self.cfg['model']['c_dim']).to(self.device)
        self.costfusion['grid_fine'] = CostFusion_tmp(ch_in=ch_in, ch_out=self.cfg['model']['c_dim']).to(self.device)
        self.costfusion['grid_color'] = CostFusion_tmp(ch_in=ch_in+3*self.n_frames, ch_out=self.cfg['model']['c_dim']).to(self.device)
        # self.costfusion['grid_color'] = CostFusion_tmp(ch_in=ch_in+3, ch_out=self.cfg['model']['c_dim']).to(self.device)
        
        
        self.mvsnet = slam.mvsnet
        self.grufusion = {}
        self.grufusion['grid_middle'] = ConvGRU(hidden_dim=64, input_dim=64, ks=self.cfg['model']['convgru_ks'],
                                                pres=1, vres=self.cfg['grid_len']['middle']).to(self.integrator.device)
        self.grufusion['grid_fine'] = ConvGRU(hidden_dim=64, input_dim=64, ks=self.cfg['model']['convgru_ks'],
                                              pres=1, vres=self.cfg['grid_len']['fine']).to(self.integrator.device)
        self.grufusion['grid_color'] = ConvGRU(hidden_dim=64, input_dim=64, ks=self.cfg['model']['convgru_ks'],
                                               pres=1, vres=self.cfg['grid_len']['color']).to(self.integrator.device)
        
        self.featurenet_pano = FeatureNet_Panorama().to(self.integrator.device)
        # self.featurenet_pano = FeatureNet_Panorama().to("cuda:3")
        
        self.probs = slam.probs
        self.target_depths = slam.target_depths
        self.target_colors = slam.target_colors
        self.use_depths = slam.use_depths
        self.use_colors = slam.use_colors
        self.target_entropys = slam.target_entropys
        self.target_c2w = slam.target_c2w
        self.backproj_feats = slam.backproj_feats

        self.resp_volume_freeze = slam.resp_volume_freeze
        
        self.global_iter = 0
        self.global_frag_iter = 0

        self.summary_writer = SummaryWriter(self.logdir)

    def init_params(self, slam):
        self.c = slam.shared_c
        self.coord = slam.shared_coord
        self.bound = slam.bound
        
        self.probs = slam.probs
        self.target_depths = slam.target_depths
        self.target_colors = slam.target_colors
        self.use_depths = slam.use_depths
        self.use_colors = slam.use_colors
        self.target_entropys = slam.target_entropys
        self.target_c2w = slam.target_c2w
        self.backproj_feats = slam.backproj_feats

    def del_params(self):
        del self.c
        del self.coord
        del self.bound
        
        del self.probs
        del self.target_depths
        del self.target_colors
        del self.use_depths
        del self.use_colors
        del self.target_entropys
        del self.target_c2w
        # del self.backproj_feats
        # del self.backproj_colors
        # del self.backproj_feats_mask

        gc.collect()
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()

    def optimize_map_omni(self, num_joint_iters, lr_factor, epoch):
        """
        Mapping iterations. Sample pixels from selected keyframes,
        then optimize scene representation and camera poses(if local BA enables).

        Args:
            num_joint_iters (int): number of mapping iterations.
            lr_factor (float): the factor to times on current lr.
            idx (int): the index of current frame
            cur_gt_color (tensor): gt_color image of the current camera.
            cur_gt_depth (tensor): gt_depth image of the current camera.
            gt_cur_c2w (tensor): groundtruth camera to world matrix corresponding to current frame.
            keyframe_dict (list): list of keyframes info dictionary.
            keyframe_list (list): list ofkeyframe index.
            cur_c2w (tensor): the estimated camera to world matrix of current frame. 

        Returns:
            cur_c2w/None (tensor/None): return the updated cur_c2w, return None if no BA
        """
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        phi_deg, phi_max_deg = self.phi_deg, self.phi_max_deg
        cfg = self.cfg
        device = self.device
        bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape(
            [1, 4])).type(torch.float32).to(device)
        
        # optimize_frame = list(np.random.permutation(np.array(range(len(self.target_c2w)))))
        optimize_frame = list(np.array(range(len(self.target_c2w))))
        pixs_per_image = self.mapping_pixels

        if epoch == 0:
            decoders_para_list = []

            if self.method == 'nice':
                decoders_para_list += list(
                        self.decoders.coarse_decoder.parameters())
                decoders_para_list += list(
                        self.decoders.middle_decoder.parameters())
                if not self.fix_fine:
                    decoders_para_list += list(
                        self.decoders.fine_decoder.parameters())
                if not self.fix_color:
                    decoders_para_list += list(
                        self.decoders.color_decoder.parameters())
            else:
                # single MLP
                decoders_para_list += list(self.decoders.parameters())

            if self.BA:
                camera_tensor_list = []
                for frame in optimize_frame:
                    c2w = self.target_c2w[frame]
                    camera_tensor = get_tensor_from_camera(c2w)
                    camera_tensor = Variable(
                        camera_tensor.to(device), requires_grad=True)
                    camera_tensor_list.append(camera_tensor)
            """gru fusion"""
            grufusion_para_list = []
            for key, net in self.grufusion.items():
                grufusion_para_list += list(net.parameters())
            
            
            """cost fusion network"""
            fusion_para_list = []
            for key, net in self.costfusion.items():
                fusion_para_list += list(net.parameters())
                
            """pano feature network"""
            featnet_para_list = []
            featnet_para_list += self.integrator.featurenet_pano.parameters()
                        
            if self.BA:
                # The corresponding lr will be set according to which stage the optimization is in
                self.optimizer = torch.optim.Adam([{'params': decoders_para_list, 'lr': 0},
                                            {'params': fusion_para_list, 'lr': 0},
                                            {'params': grufusion_para_list, 'lr': 0},
                                            {'params': camera_tensor_list, 'lr': 0}])
            else:
                # optimizer = torch.optim.Adam([{'params': decoders_para_list, 'lr': 0},
                #                             ])
                self.optimizer = torch.optim.Adam([{'params': decoders_para_list, 'lr': 0},
                                            {'params': fusion_para_list, 'lr': 0},
                                            {'params': grufusion_para_list, 'lr': 0},
                                            {'params': featnet_para_list, 'lr': 0},
                                            ])
                
            self.optimizer.param_groups[0]['lr'] = cfg['mapping']['stage']['fine']['decoders_lr']*lr_factor
            self.optimizer.param_groups[1]['lr'] = cfg['omnimvs']['lr']*lr_factor
            self.optimizer.param_groups[2]['lr'] = cfg['omnimvs']['lr']*lr_factor
            self.optimizer.param_groups[3]['lr'] = cfg['omnimvs']['lr']*lr_factor
            if self.BA:
                if self.stage == 'color':
                    self.optimizer.param_groups[4]['lr'] = self.BA_cam_lr

            # optimizer.param_groups[0]['lr'] = cfg['mapping']['stage']['fine']['decoders_lr']*lr_factor
            # optimizer.param_groups[1]['lr'] = cfg['omnimvs']['lr']*lr_factor
            from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.8, patience=5000, verbose=True)
            # scheduler = StepLR(optimizer, step_size=200, gamma=0.8)

            

        self.optimizer.zero_grad()
        for joint_iter in tqdm(range(num_joint_iters)):
            # pdb.set_trace()
            # self.global_iter += 1
            if self.method == 'nice':
                if self.coarse_mapper:
                    self.stage = 'coarse'
                elif joint_iter < int(num_joint_iters*self.middle_iter_ratio):
                    self.stage = 'middle'
                elif joint_iter < int(num_joint_iters*self.fine_iter_ratio):
                    self.stage = 'fine'
                else:
                    self.stage = 'color'
            else:
                self.stage = 'color'

            # self.optimizer.param_groups[0]['lr'] = cfg['mapping']['stage'][self.stage]['decoders_lr']*lr_factor
            # self.optimizer.param_groups[1]['lr'] = cfg['omnimvs']['lr']*lr_factor
            # self.optimizer.param_groups[2]['lr'] = cfg['omnimvs']['lr']*lr_factor
            # if self.BA:
            #     if self.stage == 'color':
            #         self.optimizer.param_groups[4]['lr'] = self.BA_cam_lr

            self.target_depths = self.integrator.target_depths
            self.target_colors = self.integrator.target_colors
            self.use_depths = self.integrator.use_depths
            self.use_colors = self.integrator.use_colors
            self.target_entropys = self.integrator.target_entropys
            self.target_c2w = self.integrator.target_c2w
            self.backproj_feats = self.integrator.backproj_feats
            # c = self.integrator.c
            
            camera_tensor_id = 0
            for i, frame in tqdm(enumerate(optimize_frame)):
                use_depth = self.use_depths[frame].to(device).detach()
                use_color = self.use_colors[frame].to(device).detach()
                use_entropy = self.target_entropys[frame].to(device).detach()
                gt_depth = self.target_depths[frame].to(device).detach()
                gt_color = self.target_colors[frame].to(device).detach()
                
                if self.BA:
                    camera_tensor = camera_tensor_list[camera_tensor_id]
                    camera_tensor_id += 1
                    c2w = get_camera_from_tensor(camera_tensor)
                else:
                    c2w = self.target_c2w[frame]
                
                if self.cam_method == 'perspective':
                    batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(
                        0, H, 0, W, pixs_per_image, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device)
                elif self.cam_method == 'panorama':
                    batch_rays_o, batch_rays_d, batch_use_depth, batch_use_color, batch_use_entropy, batch_gt_depth, batch_gt_color = get_samples_omni(
                        0, H, 0, W, pixs_per_image, H, W, phi_deg, phi_max_deg, c2w, use_depth, use_color, self.device, use_entropy, gt_depth, gt_color)
                    
                    batch_use_depth = batch_use_depth.float()
                    batch_use_color = batch_use_color.float()
                    batch_use_entropy = batch_use_entropy.float()
                    
                batch_rays_o = batch_rays_o.float()
                batch_rays_d = batch_rays_d.float()
                batch_gt_depth = batch_gt_depth.float()
                batch_gt_color = batch_gt_color.float()
            
                # should pre-filter those out of bounding box depth value
                with torch.no_grad():
                    det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                    det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                    t = (self.bound.unsqueeze(0).to(
                        device)-det_rays_o)/det_rays_d
                    t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                    inside_mask = t >= batch_gt_depth
                batch_rays_d = batch_rays_d[inside_mask]
                batch_rays_o = batch_rays_o[inside_mask]
                batch_gt_depth = batch_gt_depth[inside_mask]
                batch_gt_color = batch_gt_color[inside_mask]
                batch_use_depth = batch_use_depth[inside_mask]
                batch_use_color = batch_use_color[inside_mask]
                batch_use_entropy = batch_use_entropy[inside_mask]


                """gru fusion / average frame feature volume"""
                vote_grid = {}
                for j, fidx in (enumerate(optimize_frame)):
                    for key, grid in self.integrator.c.items():
                        if not('coarse' in key) and not('color' in key):
                            if self.fusion_method == 'grufusion':
                                feat_c_dim = self.integrator.c[key].shape[1]
                                backproj_feat = self.backproj_feats[key][fidx].to(self.integrator.device)
                                if j == 0:
                                    self.integrator.c[key] = torch.zeros_like(grid).to(self.integrator.device)
                                    mask = ((backproj_feat != 0).sum(dim=1) != 0)
                                else:
                                    mask = torch.logical_or(((backproj_feat != 0).sum(dim=1) != 0),
                                                            ((grid != 0).sum(dim=1) != 0))
                                coord = torch.nonzero(mask).type(torch.int)
                                coord = coord[:, [1, 2, 3, 0]] # [N, 4]: [x,y,z,B]
                                feat = torch.index_select(backproj_feat.reshape(feat_c_dim, -1), 1, 
                                                        torch.nonzero(mask.reshape(1, -1))[:, 1]).T
                                new_sparse_feat = SparseTensor(feats=feat, coords=coord)
                            
                                hidden_feat = torch.index_select(self.integrator.c[key].reshape(feat_c_dim, -1), 1, \
                                                                torch.nonzero(mask.reshape(1, -1))[:, 1]).T
                                hidden_sparse_feat = SparseTensor(feats=hidden_feat, coords=coord)
                                
                                fused_feat = self.grufusion[key](hidden_sparse_feat, new_sparse_feat)
                            
                                self.integrator.c[key] = sparse_to_dense(fused_feat, grid)
                            elif self.fusion_method == 'average':
                                backproj_feat = self.backproj_feats[key][fidx].to(self.integrator.device)
                                if j == 0:
                                    vote_grid[key] = ((backproj_feat != 0).sum(dim=1) != 0).float()
                                    self.integrator.c[key] = backproj_feat
                                else:
                                    vote = ((backproj_feat != 0).sum(dim=1) != 0).float()
                                    vote_grid[key] = vote_grid[key] + vote
                                    self.integrator.c[key] = self.integrator.c[key] + backproj_feat
                                    if j == len(optimize_frame) - 1:
                                        vote_grid[key][vote_grid[key] == 0] = 1
                                        self.integrator.c[key] = self.integrator.c[key] / vote_grid[key]

                """refinement feature volume"""
                c = {}
                for key, grid in self.integrator.c.items():
                    if not('coarse' in key):
                        # c[key] = self.costfusion[key](grid).to(self.device)
                        c[key] = self.costfusion[key](grid.to(self.device))
                
                        
                ret = self.renderer.render_batch_ray(c, self.decoders, batch_rays_d,
                                                    batch_rays_o, device, self.stage,
                                                    # gt_depth=None if self.coarse_mapper else batch_gt_depth,
                                                    gt_depth=None if self.coarse_mapper else batch_use_depth,
                                                    summary_writer=(self.summary_writer, self.global_iter))
                
                depth, uncertainty, color = ret

                self.global_iter += 1

                depth_mask = (batch_gt_depth > 0)
                
                loss = torch.abs(
                    (batch_gt_depth[depth_mask]-depth[depth_mask])).mean()
                self.summary_writer.add_scalar(f'{self.stage}/mapping_depth_loss', loss, global_step=self.global_iter)
                
                if ((self.method != 'nice') or (self.stage == 'color')):
                    color_loss = torch.abs(batch_gt_color - color).mean()
                    weighted_color_loss = self.w_color_loss*color_loss
                    loss += weighted_color_loss
                    self.summary_writer.add_scalar(f'{self.stage}/mapping_color_loss', color_loss, global_step=self.global_iter)
                    color_loss = color_loss.item()
                
                self.summary_writer.add_scalar(f'{self.stage}/mapping_loss', loss, global_step=self.global_iter)
                
                loss.backward(retain_graph=True)
                
                self.optimizer.step()
                
                self.scheduler.step(loss)
                # scheduler.step()
                self.optimizer.zero_grad()
                
                del c
                torch.cuda.empty_cache()

            """visualize log image"""
            with torch.no_grad():
                c = {}
                for key, grid in self.integrator.c.items():
                    if not('coarse' in key):
                        # c[key] = self.costfusion[key](grid).to(self.device)
                        c[key] = self.costfusion[key](grid.to(self.device))
            
            idx = joint_iter % len(self.target_depths)
            self.visualizer.vis_omni(
                self.epoch, joint_iter, num_joint_iters, self.use_depths[idx], self.use_colors[idx], self.target_c2w[idx], 
                c, self.decoders, self.stage, self.summary_writer, self.target_depths[idx], self.target_colors[idx])
            
            del c


        if self.BA:
            # put the updated camera poses back
            camera_tensor_id = 0
            for frame in optimize_frame:
                c2w = get_camera_from_tensor(
                    camera_tensor_list[camera_tensor_id].detach())
                c2w = torch.cat([c2w, bottom], dim=0)
                camera_tensor_id += 1
                self.target_c2w[frame] = c2w.clone()


    def optimize_map_omni_frag(self, lr_factor, frag_idx, num_joint_iters, finetune=False):
        """
        Mapping iterations. Sample pixels from selected keyframes,
        then optimize scene representation and camera poses(if local BA enables).

        Args:
            num_joint_iters (int): number of mapping iterations.
            lr_factor (float): the factor to times on current lr.
            idx (int): the index of current frame
            cur_gt_color (tensor): gt_color image of the current camera.
            cur_gt_depth (tensor): gt_depth image of the current camera.
            gt_cur_c2w (tensor): groundtruth camera to world matrix corresponding to current frame.
            keyframe_dict (list): list of keyframes info dictionary.
            keyframe_list (list): list ofkeyframe index.
            cur_c2w (tensor): the estimated camera to world matrix of current frame. 

        Returns:
            cur_c2w/None (tensor/None): return the updated cur_c2w, return None if no BA
        """
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        phi_deg, phi_max_deg = self.phi_deg, self.phi_max_deg
        cfg = self.cfg
        device = self.device
        bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape(
            [1, 4])).type(torch.float32).to(device)
        
        # optimize_frame = list(np.random.permutation(np.array(range(len(self.target_c2w)))))
        optimize_frame = list(np.array(range(len(self.target_c2w))))
        pixs_per_image = self.mapping_pixels

        if self.global_iter == 0:
            decoders_para_list = []

            if self.method == 'nice':
                decoders_para_list += list(
                        self.decoders.coarse_decoder.parameters())
                decoders_para_list += list(
                        self.decoders.middle_decoder.parameters())
                if not self.fix_fine:
                    decoders_para_list += list(
                        self.decoders.fine_decoder.parameters())
                if not self.fix_color:
                    decoders_para_list += list(
                        self.decoders.color_decoder.parameters())
            else:
                # single MLP
                decoders_para_list += list(self.decoders.parameters())

            if self.BA:
                camera_tensor_list = []
                for frame in optimize_frame:
                    c2w = self.target_c2w[frame]
                    camera_tensor = get_tensor_from_camera(c2w)
                    camera_tensor = Variable(
                        camera_tensor.to(device), requires_grad=True)
                    camera_tensor_list.append(camera_tensor)
            """gru fusion"""
            grufusion_para_list = []
            for key, net in self.grufusion.items():
                grufusion_para_list += list(net.parameters())
            
            
            """cost fusion network"""
            fusion_para_list = []
            for key, net in self.costfusion.items():
                fusion_para_list += list(net.parameters())
                
            """pano feature network"""
            featnet_para_list = []
            featnet_para_list += self.integrator.featurenet_pano.parameters()
                        
            if self.BA:
                # The corresponding lr will be set according to which stage the optimization is in
                self.optimizer = torch.optim.Adam([{'params': decoders_para_list, 'lr': 0},
                                            {'params': fusion_para_list, 'lr': 0},
                                            {'params': grufusion_para_list, 'lr': 0},
                                            {'params': camera_tensor_list, 'lr': 0}])
            else:
                self.optimizer = torch.optim.Adam([{'params': decoders_para_list, 'lr': 0},
                                            {'params': fusion_para_list, 'lr': 0},
                                            {'params': grufusion_para_list, 'lr': 0},
                                            {'params': featnet_para_list, 'lr': 0},
                                            ])
                
            self.optimizer.param_groups[0]['lr'] = cfg['mapping']['stage']['fine']['decoders_lr']*lr_factor
            self.optimizer.param_groups[1]['lr'] = cfg['omnimvs']['lr']*lr_factor
            self.optimizer.param_groups[2]['lr'] = cfg['omnimvs']['lr']*lr_factor
            self.optimizer.param_groups[3]['lr'] = cfg['omnimvs']['lr']*lr_factor
            if self.BA:
                if self.stage == 'color':
                    self.optimizer.param_groups[4]['lr'] = self.BA_cam_lr

            from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.8, patience=100, verbose=True)

        self.optimizer.zero_grad()
        self.stage = 'color'

        self.target_depths = self.integrator.target_depths
        self.target_colors = self.integrator.target_colors
        self.use_depths = self.integrator.use_depths
        self.use_colors = self.integrator.use_colors
        self.target_entropys = self.integrator.target_entropys
        self.target_c2w = self.integrator.target_c2w
        self.backproj_feats = self.integrator.backproj_feats

        self.n_img = len(optimize_frame)
        for joint_iter in tqdm(range(num_joint_iters)):
            self.global_frag_iter += 1
            for batch_idx in tqdm(range(self.n_img)):
                optimize_frame = list(np.array(range(len(self.target_c2w))))
                batch_rays_o = []
                batch_rays_d = []
                batch_use_depth = []
                batch_use_color = []
                batch_use_entropy = []
                batch_gt_depth = []
                batch_gt_color = []
                pdb.set_trace()
                for i, frame in enumerate(optimize_frame):
                    use_depth = self.use_depths[frame].to(device).detach()
                    use_color = self.use_colors[frame].to(device).detach()
                    use_entropy = self.target_entropys[frame].to(device).detach()
                    gt_depth = self.target_depths[frame].to(device).detach()
                    gt_color = self.target_colors[frame].to(device).detach()
                    
                    c2w = self.target_c2w[frame]
                    
                    if self.cam_method == 'perspective':
                        batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(
                            0, H, 0, W, pixs_per_image, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device)
                    elif self.cam_method == 'panorama':
                        batch_rays_o_i, batch_rays_d_i, batch_use_depth_i, batch_use_color_i, batch_use_entropy_i, batch_gt_depth_i, batch_gt_color_i = get_samples_omni(
                            0, H, 0, W, pixs_per_image//self.n_img, H, W, phi_deg, phi_max_deg, c2w, use_depth, use_color, self.device, use_entropy, gt_depth, gt_color)
                        batch_rays_o.append(batch_rays_o_i)
                        batch_rays_d.append(batch_rays_d_i)
                        batch_use_depth.append(batch_use_depth_i)
                        batch_use_color.append(batch_use_color_i)
                        batch_use_entropy.append(batch_use_entropy_i)
                        batch_gt_depth.append(batch_gt_depth_i)
                        batch_gt_color.append(batch_gt_color_i)
                        
                if finetune:
                    batch_use_color = batch_gt_color
                    batch_use_depth = batch_use_depth
                
                batch_use_depth = torch.cat(batch_use_depth, dim=0).float()
                batch_use_color = torch.cat(batch_use_color, dim=0).float()
                batch_use_entropy = torch.cat(batch_use_entropy, dim=0).float()
                batch_rays_o = torch.cat(batch_rays_o, dim=0).float()
                batch_rays_d = torch.cat(batch_rays_d, dim=0).float()
                batch_gt_depth = torch.cat(batch_gt_depth, dim=0).float()
                batch_gt_color = torch.cat(batch_gt_color, dim=0).float()

                # should pre-filter those out of bounding box depth value
                with torch.no_grad():
                    det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                    det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                    t = (self.bound.unsqueeze(0).to(
                        device)-det_rays_o)/det_rays_d
                    t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                    inside_mask = t >= batch_gt_depth
                batch_rays_d = batch_rays_d[inside_mask]
                batch_rays_o = batch_rays_o[inside_mask]
                batch_gt_depth = batch_gt_depth[inside_mask]
                batch_gt_color = batch_gt_color[inside_mask]
                batch_use_depth = batch_use_depth[inside_mask]
                batch_use_color = batch_use_color[inside_mask]
                batch_use_entropy = batch_use_entropy[inside_mask]


                """gru fusion / average frame feature volume"""
                vote_grid = {}
                sq_sum_grid = {}
                for j, fidx in (enumerate(optimize_frame)):
                    for key, grid in self.integrator.c.items():
                        if not('coarse' in key):
                            if self.fusion_method == 'grufusion':
                                feat_c_dim = self.integrator.c[key].shape[1]
                                backproj_feat = self.backproj_feats[key][fidx].to(self.integrator.device)
                                if j == 0:
                                    self.integrator.c[key] = torch.zeros_like(grid).to(self.integrator.device)
                                    mask = ((backproj_feat != 0).sum(dim=1) != 0)
                                else:
                                    mask = torch.logical_or(((backproj_feat != 0).sum(dim=1) != 0),
                                                            ((grid != 0).sum(dim=1) != 0))
                                coord = torch.nonzero(mask).type(torch.int)
                                coord = coord[:, [1, 2, 3, 0]] # [N, 4]: [x,y,z,B]
                                feat = torch.index_select(backproj_feat.reshape(feat_c_dim, -1), 1, 
                                                        torch.nonzero(mask.reshape(1, -1))[:, 1]).T
                                new_sparse_feat = SparseTensor(feats=feat, coords=coord)
                                hidden_feat = torch.index_select(self.integrator.c[key].reshape(feat_c_dim, -1), 1, \
                                                                torch.nonzero(mask.reshape(1, -1))[:, 1]).T
                                hidden_sparse_feat = SparseTensor(feats=hidden_feat, coords=coord)
                                fused_feat = self.grufusion[key](hidden_sparse_feat, new_sparse_feat)
                                self.integrator.c[key] = sparse_to_dense(fused_feat, grid)
                            elif self.fusion_method == 'average':
                                backproj_feat = self.backproj_feats[key][fidx].to(self.integrator.device)
                                if j == 0:
                                    vote_grid[key] = ((backproj_feat != 0).sum(dim=1) != 0).float()
                                    self.integrator.c[key] = backproj_feat
                                else:
                                    vote = ((backproj_feat != 0).sum(dim=1) != 0).float()
                                    vote_grid[key] = vote_grid[key] + vote
                                    self.integrator.c[key] = self.integrator.c[key] + backproj_feat
                                    if j == len(optimize_frame) - 1:
                                        vote_grid[key][vote_grid[key] == 0] = 1
                                        self.integrator.c[key] = self.integrator.c[key] / vote_grid[key]
                            elif self.fusion_method == 'variance':
                                backproj_feat = self.backproj_feats[key][fidx].to(self.integrator.device)
                                if j == 0:
                                    vote_grid[key] = ((backproj_feat != 0).sum(dim=1) != 0).float()
                                    self.integrator.c[key] = backproj_feat
                                    sq_sum_grid[key] = backproj_feat ** 2
                                else:
                                    vote = ((backproj_feat != 0).sum(dim=1) != 0).float()
                                    vote_grid[key] = vote_grid[key] + vote
                                    self.integrator.c[key] = self.integrator.c[key] + backproj_feat
                                    sq_sum_grid[key] = sq_sum_grid[key] + backproj_feat ** 2
                                    if j == len(optimize_frame) - 1:
                                        vote_grid[key][vote_grid[key] == 0] = 1
                                        var_grid = (sq_sum_grid[key] / vote_grid[key]) \
                                                    - (self.integrator.c[key] / vote_grid[key]) ** 2
                                        self.integrator.c[key] = torch.cat([var_grid, (self.integrator.c[key] / vote_grid[key])], dim=1)

                """refinement feature volume"""
                c = {}
                for key, grid in self.integrator.c.items():
                    if not('coarse' in key):
                        # c[key] = self.costfusion[key](grid).to(self.device)
                        c[key] = self.costfusion[key](grid.to(self.device))
                
                        
                ret = self.renderer.render_batch_ray(c, self.decoders, batch_rays_d,
                                                    batch_rays_o, device, self.stage,
                                                    # gt_depth=None if self.coarse_mapper else batch_gt_depth,
                                                    gt_depth=None if self.coarse_mapper else batch_use_depth,
                                                    summary_writer=(self.summary_writer, self.global_iter))
                
                depth, uncertainty, color = ret

                self.global_iter += 1

                depth_mask = (batch_gt_depth > 0)
                
                finetune_str = 'finetune'
                if not finetune:
                    loss = torch.abs(
                        (batch_gt_depth[depth_mask]-depth[depth_mask])).mean()
                else:
                    loss = torch.abs((batch_use_depth[depth_mask]-depth[depth_mask]))
                    self.summary_writer.add_scalar('finetune/mapping_depth_loss_wo_weight', loss.mean(), global_step=self.global_iter)
                    # entropy_weight = 1 / (1 + torch.exp(3 * (batch_use_entropy[depth_mask] - 3)))
                    entropy_weight = 1 / batch_use_entropy[depth_mask]
                    loss = (loss * entropy_weight).mean()
                self.summary_writer.add_scalar(f'{self.stage if not finetune else finetune_str}/mapping_depth_loss', loss, global_step=self.global_iter)
                
                if ((self.method != 'nice') or (self.stage == 'color')):
                    color_loss = torch.abs(batch_gt_color - color).mean()
                    weighted_color_loss = self.w_color_loss*color_loss
                    loss += weighted_color_loss
                    self.summary_writer.add_scalar(f'{self.stage if not finetune else finetune_str}/mapping_color_loss', color_loss, global_step=self.global_iter)
                    color_loss = color_loss.item()
                
                self.summary_writer.add_scalar(f'{self.stage if not finetune else finetune_str}/mapping_loss', loss, global_step=self.global_iter)
                
                loss.backward(retain_graph=True)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                del vote_grid
                del sq_sum_grid
                del c
                torch.cuda.empty_cache()

            self.scheduler.step(loss)

            """visualize log image"""
            with torch.no_grad():
                c = {}
                for key, grid in self.integrator.c.items():
                    if not('coarse' in key):
                        # c[key] = self.costfusion[key](grid).to(self.device)
                        c[key] = self.costfusion[key](grid.to(self.device))
                    # elif 'color' in key:
                    #     c[key] = self.costfusion['grid_fine'](self.integrator.c['grid_fine'].to(self.device))
            
            idx = self.iter % len(self.target_depths)
            self.visualizer.vis_omni_frag(
                self.epoch, self.global_frag_iter, self.iter, frag_idx, self.use_depths[idx], self.use_colors[idx], self.target_c2w[idx], 
                c, self.decoders, self.stage, self.summary_writer, self.target_depths[idx], self.target_colors[idx], entropy=self.target_entropys[idx], finetune=finetune)
            
            del c
            torch.cuda.empty_cache()

    def optimize_map_omni_frag_feat(self, lr_factor, frag_idx, num_joint_iters, finetune=False, save_log=False):
        """
        Mapping iterations. Sample pixels from selected keyframes,
        then optimize scene representation and camera poses(if local BA enables).

        Args:
            num_joint_iters (int): number of mapping iterations.
            lr_factor (float): the factor to times on current lr.
            idx (int): the index of current frame
            cur_gt_color (tensor): gt_color image of the current camera.
            cur_gt_depth (tensor): gt_depth image of the current camera.
            gt_cur_c2w (tensor): groundtruth camera to world matrix corresponding to current frame.
            keyframe_dict (list): list of keyframes info dictionary.
            keyframe_list (list): list ofkeyframe index.
            cur_c2w (tensor): the estimated camera to world matrix of current frame. 

        Returns:
            cur_c2w/None (tensor/None): return the updated cur_c2w, return None if no BA
        """
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        phi_deg, phi_max_deg = self.phi_deg, self.phi_max_deg
        cfg = self.cfg
        device = self.device
        
        # optimize_frame = list(np.random.permutation(np.array(range(len(self.target_c2w)))))
        optimize_frame = list(np.array(range(len(self.target_c2w))))
        pixs_per_image = self.mapping_pixels

        # if self.global_iter == 0:
        # decoders_para_list = []

        # decoders_para_list += list(
        #         self.decoders.middle_decoder.parameters())
        # if not self.fix_fine:
        #     decoders_para_list += list(
        #         self.decoders.fine_decoder.parameters())

        middle_decoder_para_list = []    
        middle_decoder_para_list += list(
            self.decoders.middle_decoder.parameters())

        fine_decoder_para_list = []    
        fine_decoder_para_list += list(
            self.decoders.fine_decoder.parameters())

        color_decoder_para_list = []    
        if not self.fix_color:
            color_decoder_para_list += list(
                self.decoders.color_decoder.parameters())

        """gru fusion"""
        grufusion_para_list = []
        for key, net in self.grufusion.items():
            grufusion_para_list += list(net.parameters())
        
        
        """cost fusion network"""
        # fusion_para_list = []
        # for key, net in self.costfusion.items():
        #     fusion_para_list += list(net.parameters())
        middle_fusion_para_list = []
        middle_fusion_para_list += list(self.costfusion['grid_middle'].parameters())
        fine_fusion_para_list = []
        fine_fusion_para_list += list(self.costfusion['grid_fine'].parameters())
        color_fusion_para_list = []
        color_fusion_para_list += list(self.costfusion['grid_color'].parameters())
            
        """pano feature network"""
        featnet_para_list = []
        featnet_para_list += list(self.featurenet_pano.parameters())
                    
        self.optimizer = torch.optim.Adam([{'params': middle_decoder_para_list, 'lr': 0},
                                           {'params': fine_decoder_para_list, 'lr': 0},
                                           {'params': color_decoder_para_list, 'lr': 0},
                                           {'params': grufusion_para_list, 'lr': 0},
                                           {'params': middle_fusion_para_list, 'lr': 0},
                                           {'params': fine_fusion_para_list, 'lr': 0},
                                           {'params': color_fusion_para_list, 'lr': 0},
                                           {'params': featnet_para_list, 'lr': 0},
                                           ])
            
        self.optimizer.param_groups[0]['lr'] = cfg['mapping']['stage'][self.stage]['middle_decoder_lr']
        self.optimizer.param_groups[1]['lr'] = cfg['mapping']['stage'][self.stage]['fine_decoder_lr']
        self.optimizer.param_groups[2]['lr'] = cfg['mapping']['stage'][self.stage]['color_decoder_lr']
        self.optimizer.param_groups[3]['lr'] = cfg['mapping']['stage'][self.stage]['grufusion_lr']
        self.optimizer.param_groups[4]['lr'] = cfg['mapping']['stage'][self.stage]['costfusion_lr']['middle']
        self.optimizer.param_groups[5]['lr'] = cfg['mapping']['stage'][self.stage]['costfusion_lr']['fine']
        self.optimizer.param_groups[6]['lr'] = cfg['mapping']['stage'][self.stage]['costfusion_lr']['color']
        self.optimizer.param_groups[7]['lr'] = cfg['mapping']['stage'][self.stage]['featnet_lr']

        from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.8, patience=100, verbose=True)
        
        del middle_decoder_para_list
        del fine_decoder_para_list
        del color_decoder_para_list
        del middle_fusion_para_list
        del fine_fusion_para_list
        del color_fusion_para_list
        del grufusion_para_list
        del featnet_para_list
            


        self.optimizer.zero_grad()
        # self.stage = 'color'

        # self.probs = self.integrator.probs
        # self.target_depths = self.integrator.target_depths
        # self.target_colors = self.integrator.target_colors
        # self.use_depths = self.integrator.use_depths
        # self.use_colors = self.integrator.use_colors
        # self.target_entropys = self.integrator.target_entropys
        # self.target_c2w = self.integrator.target_c2w

        self.n_img = len(optimize_frame)
        # with torch.cuda.device(self.device):
        #     torch.cuda.empty_cache()
        # with torch.cuda.device(self.integrator.device):
        #     torch.cuda.empty_cache()
            
        # pdb.set_trace()
        for joint_iter in tqdm(range(num_joint_iters)):
            self.global_frag_iter += 1
            for batch_idx in tqdm(range(self.n_img)):
                # pdb.set_trace()
                optimize_frame = list(np.array(range(len(self.target_c2w))))
                batch_rays_o = []
                batch_rays_d = []
                batch_use_depth = []
                batch_use_color = []
                batch_use_entropy = []
                batch_gt_depth = []
                batch_gt_color = []

                use_colors = torch.stack(self.use_colors, dim=0).detach()
                use_colors = use_colors.permute(0, 3, 1, 2).to(self.integrator.device)
                # use_colors = use_colors.permute(0, 3, 1, 2).to(self.device)
                pano_feats_tensor = self.featurenet_pano(use_colors).unsqueeze(2)
                pano_feats = pano_feats_tensor.chunk(pano_feats_tensor.shape[0], dim=0)
                del use_colors
                del pano_feats_tensor

                backproj_feats = {}
                backproj_feats['grid_middle'] = []
                backproj_feats['grid_fine'] = []
                backproj_feats['grid_color'] = []
                backproj_colors = {}
                backproj_colors['grid_middle'] = []
                backproj_colors['grid_fine'] = []
                backproj_colors['grid_color'] = []
                # self.backproj_colors_mask = {}
                # self.backproj_colors_mask['grid_middle'] = []
                # self.backproj_colors_mask['grid_fine'] = []
                # self.backproj_colors_mask['grid_color'] = []

                # pdb.set_trace()
                for i, frame in enumerate(optimize_frame):
                    use_depth = self.use_depths[frame].to(self.device).detach()
                    use_color = self.use_colors[frame].to(self.device).detach()
                    use_entropy = self.target_entropys[frame].to(self.device).detach()
                    gt_depth = self.target_depths[frame].to(self.device).detach()
                    gt_color = self.target_colors[frame].to(self.device).detach()
                    
                    c2w = self.target_c2w[frame]

                    # input_use_color = use_color.permute(2, 0, 1).unsqueeze(0).to(self.integrator.device)
                    # pano_feat = self.featurenet_pano(input_use_color).unsqueeze(2)[0]
                    
                    # pdb.set_trace()
                    for stage, _ in self.c.items():
                        if 'coarse' in stage:
                            continue
                        
                        # backproj_feat, backproj_color = self.integrator.backproject_features_color(stage, self.probs[frame], c2w, pano_feats[frame], use_color)
                        equi_grid, backproj_prob, backproj_feat_mask = self.integrator.backproject_features_color(stage, self.probs[frame], c2w, self.integrator.device)
                        back_projected_feature = grid_sample(pano_feats[frame].to(self.integrator.device), equi_grid.float(),
                                                            padding_mode='border', align_corners=True)
                        # back_projected_feature = grid_sample_3d(pano_feats[frame].to(self.integrator.device), equi_grid.float())
                        back_projected_feature = back_projected_feature * backproj_prob
                        back_projected_feature[backproj_feat_mask == False] = 0

                        backproj_feats[stage].append(back_projected_feature)
                        if 'color' in stage:
                            back_projected_color = grid_sample(use_color.permute(2, 0, 1).unsqueeze(0).unsqueeze(-3).to(self.integrator.device), equi_grid.float(), 
                                                               padding_mode='zeros', align_corners=True)
                            backproj_colors[stage].append(back_projected_color.detach())
                            del back_projected_color
                        # self.backproj_colors_mask[stage].append(mask_color)
                        del back_projected_feature
                        del equi_grid
                        del backproj_prob
                        del backproj_feat_mask
                        # del mask_feat

                    # del pano_feat
                    with torch.no_grad():
                        if self.cam_method == 'perspective':
                            batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(
                                0, H, 0, W, pixs_per_image, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device)
                        elif self.cam_method == 'panorama':
                            batch_rays_o_i, batch_rays_d_i, batch_use_depth_i, batch_use_color_i, batch_use_entropy_i, batch_gt_depth_i, batch_gt_color_i = get_samples_omni(
                                0, H, 0, W, pixs_per_image//self.n_img, H, W, phi_deg, phi_max_deg, c2w, use_depth, use_color, self.device, use_entropy, gt_depth, gt_color)
                            batch_rays_o.append(batch_rays_o_i)
                            batch_rays_d.append(batch_rays_d_i)
                            batch_use_depth.append(batch_use_depth_i)
                            batch_use_color.append(batch_use_color_i)
                            batch_use_entropy.append(batch_use_entropy_i)
                            batch_gt_depth.append(batch_gt_depth_i)
                            batch_gt_color.append(batch_gt_color_i)
                    
                    del use_depth
                    del use_color
                    del use_entropy
                    del gt_depth
                    del gt_color
                    del c2w
                    
                    del batch_rays_o_i
                    del batch_rays_d_i
                    del batch_use_depth_i
                    del batch_use_color_i
                    del batch_use_entropy_i
                    del batch_gt_depth_i
                    del batch_gt_color_i
                    
                    # gc.collect()
                    # with torch.cuda.device(self.device):
                    #     torch.cuda.empty_cache()
                    # with torch.cuda.device(self.integrator.device):
                    #     torch.cuda.empty_cache()
                # pdb.set_trace()
                if finetune:
                    batch_use_color = batch_gt_color
                    batch_use_depth = batch_use_depth
                
                
                del pano_feats
                
                with torch.no_grad():
                    batch_use_depth = torch.cat(batch_use_depth, dim=0).float()
                    batch_use_color = torch.cat(batch_use_color, dim=0).float()
                    batch_use_entropy = torch.cat(batch_use_entropy, dim=0).float()
                    batch_rays_o = torch.cat(batch_rays_o, dim=0).float()
                    batch_rays_d = torch.cat(batch_rays_d, dim=0).float()
                    batch_gt_depth = torch.cat(batch_gt_depth, dim=0).float()
                    batch_gt_color = torch.cat(batch_gt_color, dim=0).float()

                # should pre-filter those out of bounding box depth value
                with torch.no_grad():
                    det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                    det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                    t = (self.bound.unsqueeze(0).to(
                        self.device)-det_rays_o)/det_rays_d
                    t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                    inside_mask = t >= batch_gt_depth
                    del det_rays_o
                    del det_rays_d
                    del t
                    
                    batch_rays_d = batch_rays_d[inside_mask]
                    batch_rays_o = batch_rays_o[inside_mask]
                    batch_gt_depth = batch_gt_depth[inside_mask]
                    batch_gt_color = batch_gt_color[inside_mask]
                    batch_use_depth = batch_use_depth[inside_mask]
                    batch_use_color = batch_use_color[inside_mask]
                    batch_use_entropy = batch_use_entropy[inside_mask]
                    del inside_mask

                # pdb.set_trace()
                """gru fusion / average frame feature volume"""
                vote_grid = {}
                intg_c_grid = {}
                # vote_grid_color = {}
                sq_sum_grid = {}
                intg_c = {}
                # color_grid = {}
                for j, fidx in (enumerate(optimize_frame)):
                    for key, grid in self.integrator.c.items():
                        if not('coarse' in key):
                            if self.fusion_method == 'grufusion':
                                feat_c_dim = self.integrator.c[key].shape[1]
                                backproj_feat = self.backproj_feats[key][fidx].to(self.integrator.device)
                                if j == 0:
                                    self.integrator.c[key] = torch.zeros_like(grid).to(self.integrator.device)
                                    mask = ((backproj_feat != 0).sum(dim=1) != 0)
                                else:
                                    mask = torch.logical_or(((backproj_feat != 0).sum(dim=1) != 0),
                                                            ((grid != 0).sum(dim=1) != 0))
                                coord = torch.nonzero(mask).type(torch.int)
                                coord = coord[:, [1, 2, 3, 0]] # [N, 4]: [x,y,z,B]
                                feat = torch.index_select(backproj_feat.reshape(feat_c_dim, -1), 1, 
                                                        torch.nonzero(mask.reshape(1, -1))[:, 1]).T
                                new_sparse_feat = SparseTensor(feats=feat, coords=coord)
                                hidden_feat = torch.index_select(self.integrator.c[key].reshape(feat_c_dim, -1), 1, \
                                                                torch.nonzero(mask.reshape(1, -1))[:, 1]).T
                                hidden_sparse_feat = SparseTensor(feats=hidden_feat, coords=coord)
                                fused_feat = self.grufusion[key](hidden_sparse_feat, new_sparse_feat)
                                self.integrator.c[key] = sparse_to_dense(fused_feat, grid)
                                del fused_feat
                                del new_sparse_feat
                                del hidden_sparse_feat
                                del coord
                                del mask
                                del feat
                            elif self.fusion_method == 'average':
                                backproj_feat = self.backproj_feats[key][fidx].to(self.integrator.device)
                                if j == 0:
                                    vote_grid[key] = ((backproj_feat != 0).sum(dim=1) != 0).float()
                                    self.integrator.c[key] = backproj_feat
                                else:
                                    vote = ((backproj_feat != 0).sum(dim=1) != 0).float()
                                    vote_grid[key] = vote_grid[key] + vote
                                    del vote
                                    self.integrator.c[key] = self.integrator.c[key] + backproj_feat
                                    if j == len(optimize_frame) - 1:
                                        vote_grid[key][vote_grid[key] == 0] = 1
                                        self.integrator.c[key] = self.integrator.c[key] / vote_grid[key]
                            elif self.fusion_method == 'variance':
                                backproj_feat = backproj_feats[key][fidx].to(self.device)
                                if j == 0:
                                    vote_grid[key] = ((backproj_feat != 0).sum(dim=1) != 0).float().detach()
                                    intg_c_grid[key] = backproj_feat
                                    sq_sum_grid[key] = backproj_feat ** 2
                                else:
                                    vote = ((backproj_feat != 0).sum(dim=1) != 0).float().detach()
                                    del vote
                                    intg_c_grid[key] = intg_c_grid[key] + backproj_feat
                                    sq_sum_grid[key] = sq_sum_grid[key] + backproj_feat ** 2

                                    if j == len(optimize_frame) - 1:
                                        if not ('color' in key):
                                            vote_grid[key][vote_grid[key] == 0] = 1
                                            var_grid = (sq_sum_grid[key] / vote_grid[key]) \
                                                        - (intg_c_grid[key] / vote_grid[key]) ** 2
                                            intg_c[key] = torch.cat([var_grid, (intg_c_grid[key] / vote_grid[key])], dim=1)
                                            del var_grid
                                        else:
                                            vote_grid[key][vote_grid[key] == 0] = 1
                                            var_grid = (sq_sum_grid[key] / vote_grid[key]) \
                                                        - (intg_c_grid[key] / vote_grid[key]) ** 2
                                            color_grid = torch.cat(backproj_colors[key], dim=1).to(self.device).detach()
                                            # color_grid /= vote_grid[key]
                                            intg_c[key] = torch.cat([var_grid, (intg_c_grid[key] / vote_grid[key])], dim=1)
                                            del var_grid
                                del backproj_feat
                
                
                """refinement feature volume"""
                c = {}
                for key, grid in intg_c.items():
                    if not('coarse' in key):
                        if not ('color' in key):
                            c[key] = self.costfusion[key](grid.to(self.device))
                        else:
                            c[key] = self.costfusion[key](grid.to(self.device), color_grid.to(self.device))
                        
                
                # pdb.set_trace()
                ret = self.renderer.render_batch_ray(c, color_grid.to(self.device), self.decoders, batch_rays_d,
                                                    batch_rays_o, self.device, self.stage,
                                                    # gt_depth=None if self.coarse_mapper else batch_gt_depth,
                                                    gt_depth=None if self.coarse_mapper else batch_use_depth,
                                                    summary_writer=(self.summary_writer, self.global_iter))
                
                depth, _, color = ret

                self.global_iter += 1

                depth_mask = (batch_gt_depth > 0)
                
                
                # pdb.set_trace()
                finetune_str = 'finetune'
                if not finetune:
                    loss = torch.abs(
                        (batch_gt_depth[depth_mask]-depth[depth_mask])).mean()
                else:
                    loss = torch.abs((batch_use_depth[depth_mask]-depth[depth_mask]))
                    self.summary_writer.add_scalar('finetune/mapping_depth_loss_wo_weight', loss.mean().item(), global_step=self.global_iter)
                    # entropy_weight = 1 / (1 + torch.exp(3 * (batch_use_entropy[depth_mask] - 3)))
                    entropy_weight = 1 / batch_use_entropy[depth_mask]
                    loss = (loss * entropy_weight).mean()
                self.summary_writer.add_scalar(f'{self.stage if not finetune else finetune_str}/mapping_depth_loss', loss.item(), global_step=self.global_iter)
                
                
                # pdb.set_trace()
                if ((self.method != 'nice') or (self.stage == 'color')):
                    color_loss = torch.abs(batch_gt_color - color).mean()
                    # color_loss = ((batch_gt_color - color) ** 2).mean()
                    weighted_color_loss = self.w_color_loss*color_loss
                    # loss += weighted_color_loss
                    loss = loss * self.w_geo_loss + weighted_color_loss
                    self.summary_writer.add_scalar(f'{self.stage if not finetune else finetune_str}/mapping_color_loss', color_loss.item(), global_step=self.global_iter)
                    
                    del color_loss
                    del weighted_color_loss
                
                # for densit / sdf
                regulation = (not self.occupancy)
                if regulation:
                    """dentiy / sdf"""
                    point_sigma = self.renderer.regulation(
                        c, color_grid.to(self.device), self.decoders, batch_rays_d, batch_rays_o, batch_use_depth, device, self.stage)
                    regulation_loss = torch.abs(point_sigma).mean()
                    self.summary_writer.add_scalar(f'{self.stage if not finetune else finetune_str}/regulation_loss', regulation_loss.item(), global_step=self.global_iter)
                    loss += 0.0005*regulation_loss
        
                self.summary_writer.add_scalar(f'{self.stage if not finetune else finetune_str}/mapping_loss', loss.item(), global_step=self.global_iter)
                
                # pdb.set_trace()
                
                # sched_loss = loss.item()
                # pdb.set_trace()
                loss.backward()
                # pdb.set_trace()
                self.optimizer.step()
                self.optimizer.zero_grad()
                # pdb.set_trace()
                
                # del loss

                # del ret
                # del depth
                # # del uncertainty
                # del color
                # # del pano_feats
                # # del c
                # # del self.backproj_feats
                # # del self.backproj_colors
                
                # for key in list(c.keys()):
                #     del c[key]
                # for key in list(backproj_feats.keys()):
                #     del backproj_feats[key]
                # for key in list(backproj_colors.keys()):
                #     del backproj_colors[key]
                # for key in list(vote_grid.keys()):
                #     del vote_grid[key]
                # for key in list(sq_sum_grid.keys()):
                #     del sq_sum_grid[key]
                # for key in list(intg_c_grid.keys()):
                #     del intg_c_grid[key]
                # # del self.backproj_feats_mask
                
                # del batch_use_depth
                # del batch_use_color
                # del batch_use_entropy
                # del batch_rays_o
                # del batch_rays_d
                # del batch_gt_depth
                # del batch_gt_color  
                # # del color_grid             
                
                
                
                # # pdb.set_trace()
                # gc.collect()
                # with torch.cuda.device(self.device):
                #     torch.cuda.empty_cache()
                # with torch.cuda.device(self.integrator.device):
                #     torch.cuda.empty_cache()
                
                # pdb.set_trace()
                # print_gc_objs()

            self.scheduler.step(float(loss))

        """visualize log image"""
        with torch.no_grad():
            c = {}
            for key, grid in intg_c.items():
                if not('coarse' in key):
                    if not ('color' in key):
                        c[key] = self.costfusion[key](grid.to(self.device))
                    else:
                        c[key] = self.costfusion[key](grid.to(self.device), color_grid.to(self.device))
        
            idx = self.iter % len(self.target_depths)
            self.visualizer.vis_omni_frag(
                self.epoch, self.global_frag_iter, self.iter, frag_idx, self.use_depths[idx].to(self.device), self.use_colors[idx].to(self.device), self.target_c2w[idx].to(self.device),
                c, color_grid.to(self.device), self.decoders, self.stage, self.summary_writer, self.target_depths[idx].to(self.device), self.target_colors[idx].to(self.device), entropy=self.target_entropys[idx].to(self.device), finetune=finetune)
        

            if save_log:
                mesh_out_file = f'{self.output}/mesh/{(self.epoch+1):05d}_{(frag_idx+1):03d}_mesh.ply'
                self.mesher.get_mesh_omni(mesh_out_file, c, color_grid.to(self.device), self.decoders,
                                    self.device, clean_mesh=self.clean_mesh)
            # del c
            for key in list(c.keys()):
                del c[key]
        

    def run_omni(self, epoch):
        cfg = self.cfg
        
        lr_factor = cfg['mapping']['lr_factor']
        num_joint_iters = cfg['mapping']['iters']
        
        if self.verbose:
            print(Fore.GREEN)
            prefix = 'Coarse ' if self.coarse_mapper else ''
            print('Epoch ', epoch, prefix+"Mapping Frame ")
            print(Style.RESET_ALL)

        self.epoch = epoch
        self.BA = cfg['mapping']['BA'] and (not self.coarse_mapper)
        self.optimize_map_omni(num_joint_iters, lr_factor, epoch)
            
        if self.low_gpu_mem:
            torch.cuda.empty_cache()

        if not self.coarse_mapper:
            if (epoch+1) % self.ckpt_freq == 0:
                self.logger.log_omni(epoch, self.decoders, self.costfusion, self.grufusion, self.optimizer)
            
            if (epoch+1) % self.mesh_freq == 0:
                with torch.no_grad():
                    c = {}
                    for key, grid in self.integrator.c.items():
                        if not('coarse' in key):
                            c[key] = self.costfusion[key](grid.to(self.device))
                    # c = {}
                    # for key, grid in self.integrator.c.items():
                    #     if not('coarse' in key):
                    #         feat_c_dim = grid.shape[1]
                    #         mask = ((grid != 0).sum(dim=1) != 0)
                    #         coord = torch.nonzero(mask).type(torch.int)
                    #         coord = coord[:, [1, 2, 3, 0]] # [N, 4]: [x,y,z,B]
                    #         feat = torch.index_select(grid.reshape(feat_c_dim, -1), 1, 
                    #                                 torch.nonzero(mask.reshape(1, -1))[:, 1]).T
                            
                    #         sparse_feat = SparseTensor(feats=feat, coords=coord)
                    #         fused_feat = self.costfusion[key](sparse_feat).to(self.device)
                            
                    #         c_dim = self.cfg['model']['c_dim']
                    #         dense_zero_tensor = torch.zeros(grid.shape[0], c_dim, *grid.shape[2:])
                    #         c[key] = sparse_to_dense(fused_feat, dense_zero_tensor)

            
                mesh_out_file = f'{self.output}/mesh/{(epoch+1):05d}_mesh.ply'
                self.mesher.get_mesh_omni(mesh_out_file, c, self.decoders,
                                    self.device, clean_mesh=self.clean_mesh)
                del c

    def run_omni_frag(self, epoch, iter, frag_idx, i, save_log=False, finetune=False):
        cfg = self.cfg
        num_joint_iters = cfg['mapping']['joint_iters']
        lr_factor = cfg['mapping']['lr_factor']
        
        if self.verbose:
            print(Fore.GREEN)
            prefix = 'Coarse ' if self.coarse_mapper else ''
            print('Epoch', epoch, 'Iter', iter, 'Frag', f'{i}({frag_idx})', prefix+"Mapping Frame ")
            print(Style.RESET_ALL)

        self.epoch = epoch
        self.iter = iter
        self.BA = cfg['mapping']['BA'] and (not self.coarse_mapper)
        self.optimize_map_omni_frag_feat(lr_factor, frag_idx, num_joint_iters, finetune=finetune, save_log=save_log)
    
        if save_log:
            if (epoch+1) % self.ckpt_freq == 0:
                self.logger.log_omni(epoch, self.decoders, self.costfusion, self.grufusion, self.featurenet_pano, self.optimizer)
            


# occ / sdf / density에 대해 나눠서 구현하기


def print_gc_objs():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.requires_grad == False:
                s = f'{type(obj)}, {obj.size()}, {obj.device}'
                print(s)
        except:
            pass