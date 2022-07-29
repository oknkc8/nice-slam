import os
import time

import cv2
import numpy as np
import torch
from colorama import Fore, Style
from torch.autograd import Variable
from tqdm import tqdm
from tensorboardX import SummaryWriter

from src.common import (get_camera_from_tensor, get_samples, get_samples_omni,
                        get_tensor_from_camera, random_select)
from src.utils.datasets import get_dataset
from src.utils.Visualizer import Visualizer
from src.omnimvs_src.module.network import CostFusion
from src.omni_utils.common import EPS

import pdb

torch.autograd.set_detect_anomaly(True)

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
        self.costfusion = CostFusion(ch_in=64).to(self.integrator.device)
        
        self.mvsnet = slam.mvsnet
        # self.feature_fusion = slam.feature_fusion
        
        self.target_depths = slam.target_depths
        self.target_colors = slam.target_colors
        self.use_depths = slam.use_depths
        self.use_colors = slam.use_colors
        self.target_entropys = slam.target_entropys
        self.target_c2w = slam.target_c2w

        self.resp_volume_freeze = slam.resp_volume_freeze
        
        self.global_iter = 0

    def optimize_map_omni(self, num_joint_iters, lr_factor):
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
        # c = self.c
        cfg = self.cfg
        device = self.device
        bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape(
            [1, 4])).type(torch.float32).to(device)
        
        # pdb.set_trace()
        optimize_frame = list(np.random.permutation(np.array(range(len(self.target_c2w)))))
        pixs_per_image = self.mapping_pixels

        # pdb.set_trace()
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
                
        """cost fusion network"""
        fusion_para_list = []
        fusion_para_list += list(self.costfusion.parameters())
                    

        if self.BA:
            # The corresponding lr will be set according to which stage the optimization is in
            optimizer = torch.optim.Adam([{'params': decoders_para_list, 'lr': 0},
                                          {'params': fusion_para_list, 'lr': 0},
                                          {'params': camera_tensor_list, 'lr': 0}])
        else:
            # optimizer = torch.optim.Adam([{'params': decoders_para_list, 'lr': 0},
            #                             ])
            optimizer = torch.optim.Adam([{'params': decoders_para_list, 'lr': 0},
                                          {'params': fusion_para_list, 'lr': 0},
                                        ])
            
        # optimizer.param_groups[0]['lr'] = cfg['mapping']['stage']['fine']['decoders_lr']*lr_factor
        # optimizer.param_groups[1]['lr'] = cfg['omnimvs']['lr']*lr_factor
        # from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=len(optimize_frame), verbose=True)
        # scheduler = StepLR(optimizer, step_size=200, gamma=0.8)

        optimizer.zero_grad()
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

            optimizer.param_groups[0]['lr'] = cfg['mapping']['stage'][self.stage]['decoders_lr']*lr_factor
            optimizer.param_groups[1]['lr'] = cfg['omnimvs']['lr']*lr_factor
            if self.BA:
                if self.stage == 'color':
                    optimizer.param_groups[2]['lr'] = self.BA_cam_lr

            self.target_depths = self.integrator.target_depths
            self.target_colors = self.integrator.target_colors
            self.use_depths = self.integrator.use_depths
            self.use_colors = self.integrator.use_colors
            self.target_entropys = self.integrator.target_entropys
            self.target_c2w = self.integrator.target_c2w
            # c = self.integrator.c
            
            camera_tensor_id = 0
            for i, frame in enumerate(optimize_frame):
                use_depth = self.use_depths[frame].to(device).detach()
                use_color = self.use_colors[frame].to(device).detach()
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
                    batch_rays_o_1, batch_rays_d_1, batch_use_depth_1, batch_use_color_1, batch_gt_depth_1, batch_gt_color_1 = get_samples_omni(
                        0, H//3, 0, W, pixs_per_image//6, H, W, phi_deg, phi_max_deg, c2w, use_depth, use_color, self.device, gt_depth, gt_color)
                    batch_rays_o_2, batch_rays_d_2, batch_use_depth_2, batch_use_color_2, batch_gt_depth_2, batch_gt_color_2 = get_samples_omni(
                        H//3, H//3 * 2, 0, W, pixs_per_image//3 * 2, H, W, phi_deg, phi_max_deg, c2w, use_depth, use_color, self.device, gt_depth, gt_color)
                    batch_rays_o_3, batch_rays_d_3, batch_use_depth_3, batch_use_color_3, batch_gt_depth_3, batch_gt_color_3 = get_samples_omni(
                        H//3 * 2, H, 0, W, pixs_per_image//6, H, W, phi_deg, phi_max_deg, c2w, use_depth, use_color, self.device, gt_depth, gt_color)
                    
                    batch_rays_o = torch.cat([batch_rays_o_1, batch_rays_o_2, batch_rays_o_3], dim=0)
                    batch_rays_d = torch.cat([batch_rays_d_1, batch_rays_d_2, batch_rays_d_3], dim=0)
                    batch_use_depth = torch.cat([batch_use_depth_1, batch_use_depth_2, batch_use_depth_3], dim=0)
                    batch_use_color = torch.cat([batch_use_color_1, batch_use_color_2, batch_use_color_3], dim=0)
                    batch_gt_depth = torch.cat([batch_gt_depth_1, batch_gt_depth_2, batch_gt_depth_3], dim=0)
                    batch_gt_color = torch.cat([batch_gt_color_1, batch_gt_color_2, batch_gt_color_3], dim=0)
                    
                    batch_use_depth = batch_use_depth.float()
                    batch_use_color = batch_use_color.float()
                    
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

                c = {}
                for key, grid in self.integrator.c.items():
                    if not('coarse' in key) and not('color' in key):
                        c[key] = self.costfusion(grid).to(self.device)
                        # print(key, c[key].isnan().sum(), (c[key]==0).sum())

                ret = self.renderer.render_batch_ray(c, self.decoders, batch_rays_d,
                                                    batch_rays_o, device, self.stage,
                                                    # gt_depth=None if self.coarse_mapper else batch_gt_depth,
                                                    gt_depth=None if self.coarse_mapper else batch_use_depth,
                                                    summary_writer=(self.summary_writer, self.global_iter))
                
                depth, uncertainty, color = ret
                
                """print('depth', depth.isnan().sum())
                print('-'*10)
                for name, param in self.costfusion.named_parameters():
                    print(name, '   |  ', (param.numel() - param.count_nonzero()).item(), '  |  ', param.isnan().sum().item())
                print('-'*10)"""
                    
                # pdb.set_trace()

                # pdb.set_trace()

                self.global_iter += 1

                depth_mask = (batch_gt_depth > 0)
                
                # pdb.set_trace()
                
                loss = torch.abs(
                    (batch_gt_depth[depth_mask]-depth[depth_mask])).mean()
                # print('loss', loss)
                self.summary_writer.add_scalar(f'{self.stage}/mapping_depth_loss', loss, global_step=self.global_iter)
                
                # depth[depth_mask][depth[depth_mask] == 0] = EPS
                # loss = torch.abs(
                #     (1.0 / batch_gt_depth[depth_mask]) - (1.0 / depth[depth_mask])).mean()
                # self.summary_writer.add_scalar(f'{self.stage}/mapping_invdepth_loss', loss, global_step=self.global_iter)
                
                # depth_loss = loss.item()
                if ((self.method != 'nice') or (self.stage == 'color')):
                    color_loss = torch.abs(batch_gt_color - color).mean()
                    weighted_color_loss = self.w_color_loss*color_loss
                    loss += weighted_color_loss
                    self.summary_writer.add_scalar(f'{self.stage}/mapping_color_loss', color_loss, global_step=self.global_iter)
                    color_loss = color_loss.item()
                    
                
                # for imap* and omni_feat, it use volume density
                regulation = (not self.occupancy)
                if regulation:
                    point_sigma = self.renderer.regulation(
                        c, self.decoders, batch_rays_d, batch_rays_o, batch_gt_depth, device, self.stage)
                    regulation_loss = torch.abs(point_sigma).mean()
                    self.summary_writer.add_scalar(f'{self.stage}/mapping_regulation_loss', regulation_loss, global_step=self.global_iter)
                    loss += 0.0005*regulation_loss
                
                self.summary_writer.add_scalar(f'{self.stage}/mapping_loss', loss, global_step=self.global_iter)
                # pdb.set_trace()
                loss.backward(retain_graph=True)
                
                """print('-'*10)
                
                for name, param in self.costfusion.named_parameters():
                    print(name,  '   |  ', (param.grad.numel() - param.grad.count_nonzero()).item())
                
                print('='*30)
                print()"""
                # pdb.set_trace()
                
                # print((self.costfusion.get_parameter('enc_conv0.conv.weight').grad==0).sum())
                # torch.nn.utils.clip_grad.clip_grad_norm(decoders_para_list + fusion_para_list, max_norm=10)
                optimizer.step()
                
                # scheduler.step(loss)
                # scheduler.step()
                optimizer.zero_grad()
                
                del c
                # for name, param in self.costfusion.named_parameters(): print(name, param.shape)
                # self.costfusion.get_parameter('enc_conv0.conv.weight').grad
                # torch.load('output/new_underparking_tmp/29_debug_nice_occ_small_only_prob_use_pretrain_no_feat_fusion_add_3dconv/ckpts/00019.tar')
                torch.cuda.empty_cache()

            """visualize log image"""
            with torch.no_grad():
                c = {}
                for key, grid in self.integrator.c.items():
                    if not('coarse' in key):
                        c[key] = self.costfusion(grid).to(self.device)
            
            idx = joint_iter % len(self.target_depths)
            # self.visualizer.vis_omni(
            #     self.epoch, joint_iter, num_joint_iters, self.target_depths[idx], self.target_colors[idx], self.target_c2w[idx], 
            #     self.c, self.decoders, self.stage, self.summary_writer).grad
            self.visualizer.vis_omni(
                self.epoch, joint_iter, num_joint_iters, self.use_depths[idx], self.use_colors[idx], self.target_c2w[idx], 
                c, self.decoders, self.stage, self.summary_writer, self.target_depths[idx], self.target_colors[idx],)
            
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

    def run_omni(self, epoch):
        self.summary_writer = SummaryWriter(self.logdir)
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
        self.optimize_map_omni(num_joint_iters, lr_factor)
            
        if self.low_gpu_mem:
            torch.cuda.empty_cache()

        if not self.coarse_mapper:
            if (epoch+1) % self.ckpt_freq == 0:
                self.logger.log_omni(epoch, self.costfusion)
            
            if (epoch+1) % self.mesh_freq == 0:
                with torch.no_grad():
                    c = {}
                    for key, grid in self.integrator.c.items():
                        if not('coarse' in key):
                            c[key] = self.costfusion(grid).to(self.device)

            
                mesh_out_file = f'{self.output}/mesh/{(epoch+1):05d}_mesh.ply'
                self.mesher.get_mesh_omni(mesh_out_file, c, self.decoders,
                                    self.device, clean_mesh=self.clean_mesh)
                del c