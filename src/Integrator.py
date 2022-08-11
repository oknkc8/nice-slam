from curses import raw
from math import trunc
import os
import time
from tkinter import N

import cv2
from matplotlib import projections
import numpy as np
import torch
from colorama import Fore, Style
from torch.autograd import Variable
from torch.nn.functional import grid_sample
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from torchsparse.tensor import SparseTensor

from src.omnimvs_src.omnimvs import *
from src.omnimvs_src.module.network import OmniMVSNet, GRUFusion
from src.omnimvs_src.module.network_fast import OmniMVSNet as OmniMVSNetFast
from src.omnimvs_src.module.loss_functions import *
from src.omnimvs_src.module.basic_sparse import FilteringConv

from src.common import (get_camera_from_tensor, get_samples, get_samples_omni,
                        get_tensor_from_camera, random_select, sparse_to_dense)
from src.utils.datasets import get_dataset
from src.utils.Visualizer import Visualizer

import matplotlib.pyplot as plt

import pdb

torch.autograd.set_detect_anomaly(True)

class Integrator(object):
    """
    Mapper thread. Note that coarse mapper also uses this code.

    """

    def __init__(self, cfg, args, slam,
                 ):

        self.cfg = cfg
        self.args = args

        self.idx = slam.idx
        self.method = slam.method
        self.c = slam.shared_c
        self.coord = slam.shared_coord
        self.grid_coord = slam.shared_grid_coord
        self.vote = slam.shared_vote
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
        self.coarse_bound_enlarge = slam.coarse_bound_enlarge
        # self.summary_writer = slam.summary_writer
        self.logdir = slam.logdir
        self.n_img = slam.n_img

        self.device = cfg['omnimvs']['device']
        self.mapper_device = cfg['mapping']['device']
        self.scale = cfg['scale']
        self.coarse = cfg['coarse']
        self.occupancy = cfg['occupancy']
        self.sync_method = cfg['sync_method']
        self.c_dim = cfg['model']['c_dim']
        
        # self.frame_reader = get_dataset(
        #     cfg, args, self.scale, device=self.device)
        # self.n_img = len(self.frame_reader)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy
        self.cam_method = slam.cam_method
        self.phi_deg, self.phi_max_deg = slam.phi_deg, slam.phi_max_deg
        
        self.raw_imgs = []
        self.target_depths = slam.target_depths
        self.target_colors = slam.target_colors
        self.use_depths = slam.use_depths
        self.use_colors = slam.use_colors
        self.target_entropys = slam.target_entropys
        self.resps = slam.resps
        self.backproj_feats = slam.backproj_feats
        self.raw_feats = slam.raw_feats
        self.probs = slam.probs
        self.target_c2w = slam.target_c2w
        
        self.resp_volume_freeze = slam.resp_volume_freeze
        
        opts = Edict(cfg['omnimvs'])
        opts.omnimvs_opts.equirect_size = tuple(opts.omnimvs_opts.equirect_size)
        self.equirect_size = opts.omnimvs_opts.equirect_size
        self.omnimvs = OmniMVS(opts.dbname, opts.db_root, opts=opts.omnimvs_opts, train=False,
                               db_config_idxs=opts.db_config_idxs, device_id=self.device)
        self.dbloader = DataLoader(self.omnimvs, shuffle=False, num_workers=0, batch_size=1, 
                                   collate_fn=BatchCollator.collate, pin_memory=False)
        self.mvsnet = OmniMVSNet(opts.net_opts).to(self.device)
        self.n_img = len(self.dbloader)
        
        # self.grufusion = {}
        # self.grufusion['middle'] = GRUFusion(ch_in=64, ch_out=64).to(self.device)
        # self.grufusion['fine'] = GRUFusion(ch_in=64, ch_out=64).to(self.device)
        # self.grufusion['color'] = GRUFusion(ch_in=64, ch_out=64).to(self.device)
        
        # self.grufusion_h = {}
        # self.grufusion_h['middle'] = False
        # self.grufusion_h['fine'] = False
        # self.grufusion_h['color'] = False
        
        if not osp.exists(opts.snapshot_path):
            sys.exit('%s does not exsits' % (opts.snapshot_path))
        snapshot = torch.load(opts.snapshot_path)
        self.mvsnet.load_state_dict(snapshot['net_state_dict'])
        
        self.grid_cams = buildIndividualLookupTable(self.omnimvs.ocams,
                                self.omnimvs.num_invdepth, self.omnimvs.min_invdepth, 
                                self.omnimvs.step_invdepth, self.omnimvs, device_id=self.device)
        self.num_invdepth = self.omnimvs.num_invdepth

    def run_omnimvs(self, data):
        """pin memory"""
        data.imgs = [I.to(self.device) for I in data.imgs]
        data.c2w = data.c2w.to(self.device)
        if data.gt is not None:
            data.gt = data.gt.to(self.device)
            data.valid = data.valid.to(self.device)
        if data.gt_img is not None:
            data.gt_img = data.gt_img.to(self.device)
        c2w = data.c2w[0].to(self.mapper_device)
        
        """run omnimvs"""
        resp, raw_feat, geo_feat = self.omnimvs.forwardNetwork_for_integrate(
                                            self.mvsnet, data.imgs, data.sub_db_idxs, 
                                            resp_volume_freeze=self.resp_volume_freeze)
        
        return resp, raw_feat, geo_feat, c2w, data

    def generate_prob_depth_color(self, resp, data):
        raw_imgs = data.raw_imgs
        prob = F.softmax(resp, 1)
        entropy = torch.sum(-torch.log(prob + EPS) * prob, 1)
        entropy = entropy.squeeze(0)
        
        gt_depth = None
        gt_color = None
        if data.gt is not None and data.gt_img is not None:
            gt_depth = 1.0 / self.omnimvs.indexToInvdepth(data.gt)
            gt_color = data.gt_img
            gt_depth = gt_depth.to(self.mapper_device).squeeze(0)
            gt_color = gt_color.to(self.mapper_device).squeeze(0)
        
        index = torch.mul(prob, self.omnimvs.invdepth_indices)
        index = torch.sum(index, 1)
        
        invdepth = self.omnimvs.indexToInvdepth(index)
        view_invdepth = self.omnimvs.respToCameraInvdepth(resp, grids_cam=self.grid_cams)
        view_idxs, _ = self.omnimvs.getPanoramaViewdepthIndex(view_invdepth, invdepth)
        
        ref_idxs = self.omnimvs.getRefPanorama(invdepth)
        ref_idxs = torch.stack(ref_idxs, dim=0)
        view_idxs = torch.stack(view_idxs, dim=0)
        
        origin_imgs, valid_masks = self.omnimvs.getRGBViewPanorama(raw_imgs, invdepth)
        sigma = 3
        
        weight = torch.zeros((4, self.equirect_size[0], self.equirect_size[1])).to(self.device)
        
        origin_imgs = torch.stack(origin_imgs, dim=0)
        non_zero_area = torch.stack(valid_masks, dim=0)
        
        weight[non_zero_area] = torch.exp(-((ref_idxs[non_zero_area] - view_idxs[non_zero_area]) / sigma) ** 2)
        weight_normed = torch.divide(weight, torch.sum(weight, dim=0))
        weight_normed = weight_normed.unsqueeze(1)
        
        color = origin_imgs * weight_normed
        color = torch.sum(color, dim=0)
        depth = 1.0 / invdepth
        
        color[isnan(color)] = 0
        
        depth = depth.squeeze(0)
        color = color.permute(1, 2, 0) / 255.
        
        return prob, depth.to(self.mapper_device), color.to(self.mapper_device), entropy.to(self.mapper_device), gt_depth, gt_color
    
    def integrate_features(self, stage, raw_feat, geo_feat, prob, c2w):
        dist_threshold = 10
        
        """fusion features"""
        raw_feat = raw_feat.to(self.device)
        # geo_feat = geo_feat.to(self.mapper_device)
        prob = prob.to(self.device)
        
        if False and 'fine' in stage:
            pdb.set_trace()
            grid_range = [torch.arange(0, n_vox) for n_vox in [192,320,640]]
            grid = torch.meshgrid(*grid_range)
            
            max_indices = torch.max(prob.squeeze(1), dim=1)[1].reshape(1, -1)
            p_shape = prob.shape
            prob = prob.reshape(self.num_invdepth, -1)
            for i in tqdm(range(max_indices.shape[-1])):
                mask = torch.zeros_like(prob[:, i])
                mask[max_indices[:, i]] = 1
                prob[mask==1, i] = 1
                prob[mask == 0, i] = 0
            prob = prob.reshape(*p_shape)
            
            index = max_indices.reshape(prob.shape[-2:])
                    
            pdb.set_trace()
            invdepth = self.omnimvs.indexToInvdepth(index)
            self.omnimvs.writeInvdepth(invdepth.cpu().numpy(), 'debug/prob_invdepth.tiff')
        
        # prob = F.interpolate(prob.unsqueeze(1), scale_factor=0.5, mode='trilinear', align_corners=True)
        # prob = F.interpolate(prob.unsqueeze(1), scale_factor=0.5, mode='nearest')
        geo_feat = F.interpolate(geo_feat, scale_factor=2, mode='trilinear', align_corners=True)
        
        trunc_dist = self.cfg['grid_len'][stage[5:]]
        # trunc_dist = 0.04
        
        index = torch.mul(prob, self.omnimvs.invdepth_indices)
        index = torch.sum(index, 1)
        depth = 1.0 / self.omnimvs.indexToInvdepth(index)
        
        front_trunc_depth = depth - trunc_dist
        back_trunc_depth = depth + trunc_dist
        
        # cam_feature = self.feature_fusion(raw_feat * prob)
        # cam_feature = raw_feat * prob # grid channel: 64
        # cam_feature = prob.unsqueeze(1)   # grid_channel: 1
        cam_feature = geo_feat # grid channel: 64
        
        grid_range = [torch.arange(0, n_vox) for n_vox in prob.shape[-3:]]
        index_grid, _, _ = torch.meshgrid(*grid_range)
        depth_grid = 1.0 / self.omnimvs.indexToInvdepth(index_grid).to(self.device)
        
        dist_valid_mask = depth_grid <= dist_threshold
        trunc_in_mask = (depth_grid >= front_trunc_depth) & (depth_grid <= back_trunc_depth) & dist_valid_mask
        back_mask = (depth_grid > back_trunc_depth) & dist_valid_mask
        trunc_in_mask = trunc_in_mask[None, None, ...]
        valid_vote_mask = (back_mask[None, None, ...] == False).float()
        
        trunc_in_mask = trunc_in_mask.expand(-1, self.c[stage].shape[1], -1, -1, -1)
        valid_vote_mask = valid_vote_mask.expand(-1, self.c[stage].shape[1], -1, -1, -1)
        
        # cam_feature[trunc_in_mask == False] = 0
        cam_feature[back_mask == True] = 0
        
        w2c = torch.tensor(inverseTransform(c2w.cpu().numpy())).to(self.device)
        
        """back-project feature"""
        pts_world = self.coord[stage].contiguous().reshape(3, -1)
        pts_world = pts_world[[2,1,0]]
        pts_cam = applyTransform(w2c.cpu().numpy(), pts_world)
        pts_cam[1, :] *= -1
        pts_cam[2, :] *= -1
        invdepth_cam = 1.0 / sqrt(torch.sum(pts_cam ** 2, 0))
        
        equi_pix = getEquirectCoordinate(
                pts_cam, self.equirect_size, self.phi_deg, self.phi_max_deg)
        equi_didx = self.omnimvs.invdepthToIndex(invdepth_cam)
        
        height, width = self.equirect_size
        
        equi_grid_x = equi_pix[0, :] / (width - 1) * 2 - 1
        equi_grid_y = equi_pix[1, :] / (height - 1) * 2 - 1                
        equi_grid_d = (equi_didx / (self.num_invdepth - 1)) * 2 - 1
        
        equi_grid_x = equi_grid_x.reshape(*self.coord[stage].shape[-3:])
        equi_grid_y = equi_grid_y.reshape(*self.coord[stage].shape[-3:])
        equi_grid_d = equi_grid_d.reshape(*self.coord[stage].shape[-3:])
        
        equi_grid = torch.stack([equi_grid_x, equi_grid_y, equi_grid_d], dim=-1).unsqueeze(0)
        
        mask = ((equi_grid.abs() <= 1.0).sum(dim=-1) == 3).unsqueeze(0)
        equi_grid = equi_grid.float().to(self.device)
        
        mask = mask.expand(-1, self.c[stage].shape[1], -1, -1, -1)
        
        back_projected_feature = grid_sample(cam_feature, equi_grid.float(),
                                    padding_mode='border', align_corners=True)
        """
        # filtering convolution using sparse conv
        feat = back_projected_feature.reshape(self.c[stage].shape[1], -1).T
        coord = self.grid_coord[stage].reshape(3, -1).T
        coord = torch.cat([coord, torch.zeros(coord.shape[0], 1).to(self.device)], dim=-1).type(torch.int)
        back_projected_feature_sparse = SparseTensor(feat, coord)
        filterd_back_projected_feature_sparse = self.filtering_conv[stage[5:]](back_projected_feature_sparse)
        back_projected_feature = sparse_to_dense(filterd_back_projected_feature_sparse, back_projected_feature, 0)
        """
        back_projected_valid_vote_mask = grid_sample(valid_vote_mask, equi_grid.float(),
                                    padding_mode='border', align_corners=True)
        mask = mask & (back_projected_valid_vote_mask != 0)
        back_projected_feature[mask == False] = 0
        mask = mask.float()
        
        """
        # averaging
        self.c[stage] = self.c[stage] * self.vote[stage] + back_projected_feature
        self.vote[stage] = self.vote[stage] + mask
        invalid_mask = self.vote[stage] == 0
        mask_vote = self.vote[stage].clone()
        mask_vote[invalid_mask] = 1
        self.c[stage] = self.c[stage] / mask_vote
        """
        pdb.set_trace()
        # gru fusion
        if self.grufusion_h[stage[5:]] == False:
            self.c[stage] = self.grufusion[stage[5:]](back_projected_feature)
            self.grufusion_h[stage[5:]] = True
        else:
            self.c[stage] = self.grufusion[stage[5:]](back_projected_feature, self.c[stage])
        
        if False and self.frame_i == self.n_img - 1:
            if 'middle' in stage:
                from mpl_toolkits.mplot3d import Axes3D
                import plotly.graph_objects as go
                import plotly.io as po
                with torch.no_grad():                
                    pdb.set_trace()
                    
                    # index = max_indices.reshape(prob.shape[-2:])
                    
                    # invdepth = self.omnimvs.indexToInvdepth(index)
                    # self.omnimvs.writeInvdepth(invdepth.cpu().numpy(), 'debug/prob_invdepth.png')
                    
                    sshape=self.c[stage][0][0].shape
                    grid_range = [torch.arange(0, n_vox) for n_vox in sshape]
                    x, y, z = torch.meshgrid(*grid_range)
                    x = x.cpu().numpy().astype(np.float) * 0.32 + self.bound[2][0].cpu().numpy()
                    y = y.cpu().numpy().astype(np.float) * 0.32 + self.bound[1][0].cpu().numpy()
                    z = z.cpu().numpy().astype(np.float) * 0.32 + self.bound[0][0].cpu().numpy()
                    
                    values = self.vote[stage][0][0].cpu().numpy().reshape(-1)
                    fig = go.Figure(data=go.Volume(
                        x=z.reshape(-1),
                        y=y.reshape(-1),
                        z=x.reshape(-1),
                        value=values.reshape(-1),
                        isomin=values.min().item(),
                        isomax=values.max().item(),
                        opacity=0.05,
                        surface_count=25,
                    ))
                    fig.update_layout(scene_aspectmode='data')
                    po.write_html(fig, file='debug/html/vote_volume.html', auto_open=False)
                    
                    values = back_projected_valid_vote_mask[0][0].cpu().numpy().reshape(-1)
                    fig = go.Figure(data=go.Volume(
                        x=z.reshape(-1),
                        y=y.reshape(-1),
                        z=x.reshape(-1),
                        value=values.reshape(-1),
                        isomin=values.min().item(),
                        isomax=values.max().item(),
                        opacity=0.05,
                        surface_count=25,
                    ))
                    fig.update_layout(scene_aspectmode='data')
                    po.write_html(fig, file='debug/html/valid_vote_volume.html', auto_open=False)
                    
                    values = mask[0][0].cpu().numpy().reshape(-1)
                    fig = go.Figure(data=go.Volume(
                        x=z.reshape(-1),
                        y=y.reshape(-1),
                        z=x.reshape(-1),
                        value=values.reshape(-1),
                        isomin=values.min().item(),
                        isomax=values.max().item(),
                        opacity=0.05,
                        surface_count=25,
                    ))
                    fig.update_layout(scene_aspectmode='data')
                    po.write_html(fig, file='debug/html/mask_bound_volume.html', auto_open=False)
                    
                    values = back_projected_feature[0][0].cpu().numpy().reshape(-1)
                    values[values<=0.01] = 0
                    fig = go.Figure(data=go.Volume(
                        x=z.reshape(-1),
                        y=y.reshape(-1),
                        z=x.reshape(-1),
                        value=values.reshape(-1),
                        isomin=values.min().item(),
                        isomax=values.max().item(),
                        opacity=0.05,
                        surface_count=25,
                    ))
                    fig.update_layout(scene_aspectmode='data')
                    po.write_html(fig, file='debug/html/backfeat_volume.html', auto_open=False)
                    
                    values = self.c[stage][0][0].cpu().numpy().reshape(-1)
                    fig = go.Figure(data=go.Volume(
                        x=z.reshape(-1),
                        y=y.reshape(-1),
                        z=x.reshape(-1),
                        value=values.reshape(-1),
                        isomin=values.min().item(),
                        isomax=values.max().item(),
                        opacity=0.05,
                        surface_count=25,
                    ))
                    fig.update_layout(scene_aspectmode='data')
                    po.write_html(fig, file='debug/html/intg_volume.html', auto_open=False)
                    
                    
                    sshape=self.coord[stage][0][0].shape
                    grid_range = [torch.arange(0, n_vox) for n_vox in sshape]
                    x, y, z = torch.meshgrid(*grid_range)
                    x = x.cpu().numpy().astype(np.float) * 0.32 + self.bound[2][0].cpu().numpy()
                    y = y.cpu().numpy().astype(np.float) * 0.32 + self.bound[1][0].cpu().numpy()
                    z = z.cpu().numpy().astype(np.float) * 0.32 + self.bound[0][0].cpu().numpy()
                    
                    
                    values = ((equi_didx / (self.num_invdepth - 1)) * 2 - 1).cpu().numpy().reshape(-1)
                    fig = go.Figure(data=go.Volume(
                        x=z.reshape(-1),
                        y=y.reshape(-1),
                        z=x.reshape(-1),
                        value=values.reshape(-1),
                        isomin=values.min().item(),
                        isomax=values.max().item(),
                        opacity=0.05,
                        surface_count=25,
                    ))
                    fig.update_layout(scene_aspectmode='data')
                    po.write_html(fig, file='debug/html/didx_volume.html', auto_open=False)
                    
                    values = equi_pix[0,:].cpu().numpy().reshape(-1)
                    fig = go.Figure(data=go.Volume(
                        x=z.reshape(-1),
                        y=y.reshape(-1),
                        z=x.reshape(-1),
                        value=values.reshape(-1),
                        isomin=values.min().item(),
                        isomax=values.max().item(),
                        opacity=0.05,
                        surface_count=25,
                    ))
                    fig.update_layout(scene_aspectmode='data')
                    po.write_html(fig, file='debug/html/equi_x_volume.html', auto_open=False)
                    
                    values = equi_pix[1,:].cpu().numpy().reshape(-1)
                    fig = go.Figure(data=go.Volume(
                        x=z.reshape(-1),
                        y=y.reshape(-1),
                        z=x.reshape(-1),
                        value=values.reshape(-1),
                        isomin=values.min().item(),
                        isomax=values.max().item(),
                        opacity=0.05,
                        surface_count=25,
                    ))
                    fig.update_layout(scene_aspectmode='data')
                    po.write_html(fig, file='debug/html/equi_y_volume.html', auto_open=False)
                
                    values = invdepth_cam.cpu().numpy().reshape(-1)
                    fig = go.Figure(data=go.Volume(
                        x=z.reshape(-1),
                        y=y.reshape(-1),
                        z=x.reshape(-1),
                        value=values.reshape(-1),
                        isomin=values.min().item(),
                        isomax=values.max().item(),
                        opacity=0.05,
                        surface_count=25,
                    ))
                    fig.update_layout(scene_aspectmode='data')
                    # po.write_html(fig, file='intg_volume.html', auto_open=False)
                    po.write_html(fig, file='debug/html/invdepth_volume.html', auto_open=False)
                    
                    values = pts_cam[0, :].cpu().numpy().reshape(-1)
                    fig = go.Figure(data=go.Volume(
                        x=z.reshape(-1),
                        y=y.reshape(-1),
                        z=x.reshape(-1),
                        value=values.reshape(-1),
                        isomin=values.min().item(),
                        isomax=values.max().item(),
                        opacity=0.05,
                        surface_count=25,
                    ))
                    fig.update_layout(scene_aspectmode='data')
                    # po.write_html(fig, file='intg_volume.html', auto_open=False)
                    po.write_html(fig, file='debug/html/pts_cam_x_volume.html', auto_open=False)
                                        
                    values = pts_cam[1, :].cpu().numpy().reshape(-1)
                    fig = go.Figure(data=go.Volume(
                        x=z.reshape(-1),
                        y=y.reshape(-1),
                        z=x.reshape(-1),
                        value=values.reshape(-1),
                        isomin=values.min().item(),
                        isomax=values.max().item(),
                        opacity=0.05,
                        surface_count=25,
                    ))
                    fig.update_layout(scene_aspectmode='data')
                    # po.write_html(fig, file='intg_volume.html', auto_open=False)
                    po.write_html(fig, file='debug/html/pts_cam_y_volume.html', auto_open=False)
                    
                    values = pts_cam[2, :].cpu().numpy().reshape(-1)
                    fig = go.Figure(data=go.Volume(
                        x=z.reshape(-1),
                        y=y.reshape(-1),
                        z=x.reshape(-1),
                        value=values.reshape(-1),
                        isomin=values.min().item(),
                        isomax=values.max().item(),
                        opacity=0.05,
                        surface_count=25,
                    ))
                    fig.update_layout(scene_aspectmode='data')
                    # po.write_html(fig, file='intg_volume.html', auto_open=False)
                    po.write_html(fig, file='debug/html/pts_cam_z_volume.html', auto_open=False)
                
        del back_projected_feature
        del cam_feature    
        # del mask_vote
        del geo_feat
        del raw_feat
        del prob
        torch.cuda.empty_cache()
        
        
    def backproject_features(self, stage, raw_feat, geo_feat, prob, c2w):
        dist_threshold = 10
        
        """fusion features"""
        raw_feat = raw_feat.to(self.device)
        # geo_feat = geo_feat.to(self.mapper_device)
        prob = prob.to(self.device)
        
        geo_feat = F.interpolate(geo_feat, scale_factor=2, mode='trilinear', align_corners=True)
        
        trunc_dist = self.cfg['grid_len'][stage[5:]]
        
        index = torch.mul(prob, self.omnimvs.invdepth_indices)
        index = torch.sum(index, 1)
        depth = 1.0 / self.omnimvs.indexToInvdepth(index)
        
        front_trunc_depth = depth - trunc_dist
        back_trunc_depth = depth + trunc_dist
        
        cam_feature = geo_feat # grid channel: 64
        
        grid_range = [torch.arange(0, n_vox) for n_vox in prob.shape[-3:]]
        index_grid, _, _ = torch.meshgrid(*grid_range)
        depth_grid = 1.0 / self.omnimvs.indexToInvdepth(index_grid).to(self.device)
        
        dist_valid_mask = depth_grid <= dist_threshold
        trunc_in_mask = (depth_grid >= front_trunc_depth) & (depth_grid <= back_trunc_depth) & dist_valid_mask
        back_mask = (depth_grid > back_trunc_depth) & dist_valid_mask
        trunc_in_mask = trunc_in_mask[None, None, ...]
        valid_vote_mask = (back_mask[None, None, ...] == False).float()
        
        trunc_in_mask = trunc_in_mask.expand(-1, self.c[stage].shape[1], -1, -1, -1)
        valid_vote_mask = valid_vote_mask.expand(-1, self.c[stage].shape[1], -1, -1, -1)
        
        cam_feature[trunc_in_mask == False] = 0
        
        w2c = torch.tensor(inverseTransform(c2w.cpu().numpy())).to(self.device)
        
        """back-project feature"""
        pts_world = self.coord[stage].contiguous().reshape(3, -1)
        pts_world = pts_world[[2,1,0]]
        pts_cam = applyTransform(w2c.cpu().numpy(), pts_world)
        pts_cam[1, :] *= -1
        pts_cam[2, :] *= -1
        invdepth_cam = 1.0 / sqrt(torch.sum(pts_cam ** 2, 0))
        
        equi_pix = getEquirectCoordinate(
                pts_cam, self.equirect_size, self.phi_deg, self.phi_max_deg)
        equi_didx = self.omnimvs.invdepthToIndex(invdepth_cam)
        
        height, width = self.equirect_size
        
        equi_grid_x = equi_pix[0, :] / (width - 1) * 2 - 1
        equi_grid_y = equi_pix[1, :] / (height - 1) * 2 - 1                
        equi_grid_d = (equi_didx / (self.num_invdepth - 1)) * 2 - 1
        
        equi_grid_x = equi_grid_x.reshape(*self.coord[stage].shape[-3:])
        equi_grid_y = equi_grid_y.reshape(*self.coord[stage].shape[-3:])
        equi_grid_d = equi_grid_d.reshape(*self.coord[stage].shape[-3:])
        
        equi_grid = torch.stack([equi_grid_x, equi_grid_y, equi_grid_d], dim=-1).unsqueeze(0)
        
        mask = ((equi_grid.abs() <= 1.0).sum(dim=-1) == 3).unsqueeze(0)
        equi_grid = equi_grid.float().to(self.device)
        
        mask = mask.expand(-1, self.c[stage].shape[1], -1, -1, -1)
        
        back_projected_feature = grid_sample(cam_feature, equi_grid.float(),
                                    padding_mode='border', align_corners=True)
        back_projected_valid_vote_mask = grid_sample(valid_vote_mask, equi_grid.float(),
                                    padding_mode='border', align_corners=True)
        mask = mask & (back_projected_valid_vote_mask != 0)
        back_projected_feature[mask == False] = 0
        
        del cam_feature
        del geo_feat
        del raw_feat
        del prob
        torch.cuda.empty_cache()
        
        return back_projected_feature

    def run_omni(self, data, frame_i):
        with torch.no_grad():
            resp, raw_feat, geo_feat, c2w, data = self.run_omnimvs(data)
            prob, depth, color, entropy, gt_depth, gt_color = self.generate_prob_depth_color(resp, data)
            del resp
            
        self.frame_i = frame_i
        
        """
        for stage, _ in self.c.items():
            if 'coarse' in stage:
                continue
            self.integrate_features(stage, raw_feat, geo_feat, prob, c2w)
        """
        for stage, _ in self.c.items():
            if 'coarse' in stage:
                continue
            self.backproj_feats[stage].append(self.backproject_features(stage, raw_feat, geo_feat, prob, c2w))
        
        self.target_depths.append(gt_depth)
        self.target_colors.append(gt_color)
        self.use_depths.append(depth)
        self.use_colors.append(color)
        self.target_entropys.append(entropy)
        self.target_c2w.append(c2w)
        
        del raw_feat
        del geo_feat
        del prob
        torch.cuda.empty_cache()