import os
import time
import shutil
import random

import numpy as np
import torch
import torch.multiprocessing
import torch.multiprocessing as mp
from tqdm import tqdm
from colorama import Fore, Style

from src import config
from src.Mapper_Omni import Mapper
from src.Tracker import Tracker
from src.Integrator import Integrator
from src.omnimvs_src.omnimvs import BatchCollator
from src.utils.datasets import get_dataset
from src.utils.Logger import Logger
from src.utils.Mesher import Mesher
from src.utils.Renderer import Renderer

torch.multiprocessing.set_sharing_strategy('file_system')

import pdb
torch.autograd.set_detect_anomaly(True)

class NICE_SLAM_Omni():
    """
    NICE_SLAM main class.
    Mainly allocate shared resources, and dispatch mapping and tracking process.
    """

    def __init__(self, cfg, args):

        self.cfg = cfg
        self.args = args
        self.method = cfg['method']

        self.coarse = cfg['coarse']
        self.occupancy = cfg['occupancy']
        self.low_gpu_mem = cfg['low_gpu_mem']
        self.verbose = cfg['verbose']
        self.dataset = cfg['dataset']
        self.coarse_bound_enlarge = cfg['model']['coarse_bound_enlarge']
        if args.output is None:
            self.output = cfg['data']['output']
        else:
            self.output = args.output
        self.ckptsdir = os.path.join(self.output, 'ckpts')
        self.logdir = f'{self.output}/log'
        os.makedirs(self.output, exist_ok=True)
        os.makedirs(self.ckptsdir, exist_ok=True)
        os.makedirs(f'{self.output}/log', exist_ok=True)
        os.makedirs(f'{self.output}/mesh', exist_ok=True)
        shutil.copy(args.config, os.path.join(self.output, 'config.yaml'))
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
        self.cam_method = cfg['cam']['method']
        self.phi_deg, self.phi_max_deg = cfg['cam']['phi_deg'], cfg['cam']['phi_max_deg']
        self.update_cam()

        model = config.get_model(cfg,  model=self.method)
        self.shared_decoders = model

        self.scale = cfg['scale']
        self.n_img = 0

        self.load_bound(cfg)
        if self.method != 'imap':
            if self.method == 'nice' and cfg['pretrained_decoders']['use']:
                self.load_pretrain(cfg)
            self.grid_init(cfg)
        else:
            self.shared_c = {}
            self.shared_coord = {}
            self.shared_grid_coord = {}
            self.shared_vote = {}

        # need to use spawn
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        # self.summary_writer = SummaryWriter(f'{self.output}/log')
        # self.frame_reader = get_dataset(cfg, args, self.scale)
        # self.n_img = len(self.frame_reader)
        self.estimate_c2w_list = torch.zeros((self.n_img, 4, 4))
        self.estimate_c2w_list.share_memory_()

        self.gt_c2w_list = torch.zeros((self.n_img, 4, 4))
        self.gt_c2w_list.share_memory_()
        self.idx = torch.zeros((1)).int()
        self.idx.share_memory_()
        self.mapping_first_frame = torch.zeros((1)).int()
        self.mapping_first_frame.share_memory_()
        # the id of the newest frame Mapper is processing
        self.mapping_idx = torch.zeros((1)).int()
        self.mapping_idx.share_memory_()
        self.mapping_cnt = torch.zeros((1)).int()  # counter for mapping
        self.mapping_cnt.share_memory_()
        for key, val in self.shared_c.items():
            val = val.to(self.cfg['omnimvs']['device'])
            val.share_memory_()
            self.shared_c[key] = val
        for key, val in self.shared_coord.items():
            val = val.to(self.cfg['omnimvs']['device'])
            val.share_memory_()
            self.shared_coord[key] = val
        for key, val in self.shared_grid_coord.items():
            val = val.to(self.cfg['omnimvs']['device'])
            val.share_memory_()
            self.shared_grid_coord[key] = val
        for key, val in self.shared_vote.items():
            val = val.to(self.cfg['omnimvs']['device'])
            val.share_memory_()
            self.shared_vote[key] = val
        self.shared_decoders = self.shared_decoders.to(
            self.cfg['mapping']['device'])
        self.shared_decoders.share_memory()
        self.renderer = Renderer(cfg, args, self)
        self.mesher = Mesher(cfg, args, self)
        self.logger = Logger(cfg, args, self)
        """ OmniMVS """
        self.target_depths = []
        self.target_colors = []
        self.use_depths = []
        self.use_colors = []
        self.target_entropys = []
        self.resps = []
        self.backproj_feats = {}
        self.backproj_feats['grid_middle'] = []
        self.backproj_feats['grid_fine'] = []
        self.backproj_feats['grid_color'] = []
        self.raw_feats = []
        self.probs = []
        self.target_c2w = []
        self.resp_volume_freeze = self.cfg['omnimvs']['resp_volume_freeze']
        self.integrator = Integrator(cfg, args, self)
        self.dbloader = self.integrator.dbloader
        self.mvsnet = self.integrator.mvsnet
        # self.feature_fusion = self.integrator.feature_fusion
        """"""
        self.mapper = Mapper(cfg, args, self, coarse_mapper=False)
        if self.coarse:
            self.coarse_mapper = Mapper(cfg, args, self, coarse_mapper=True)
            
        self.costfusion = self.mapper.costfusion

        # self.load_pretrain_frag(cfg, '/data5/changho/nice-slam/output/underparking_sparse/41_debug_nice_occ_65_geo_feat_multiplyProb_use_pretrain_avg_nobias_48_16_fix_backmask_trunc1_newcostfusion_noshuffle_change_integration_localfrag_debugging/ckpts/00027.tar')
        
        self.print_output_desc()

    def print_output_desc(self):
        print(f"INFO: The output folder is {self.output}")
        if 'Demo' in self.output:
            print(
                f"INFO: The GT, generated and residual depth/color images can be found under " +
                f"{self.output}/vis/")
        else:
            print(
                f"INFO: The GT, generated and residual depth/color images can be found under " +
                f"{self.output}/tracking_vis/ and {self.output}/mapping_vis/")
        print(f"INFO: The mesh can be found under {self.output}/mesh/")
        print(f"INFO: The checkpoint can be found under {self.output}/ckpt/")

    def update_cam(self):
        """
        Update the camera intrinsics according to pre-processing config, 
        such as resize or edge crop.
        """
        # resize the input images to crop_size (variable name used in lietorch)
        if 'crop_size' in self.cfg['cam']:
            crop_size = self.cfg['cam']['crop_size']
            sx = crop_size[1] / self.W
            sy = crop_size[0] / self.H
            self.fx = sx*self.fx
            self.fy = sy*self.fy
            self.cx = sx*self.cx
            self.cy = sy*self.cy
            self.W = crop_size[1]
            self.H = crop_size[0]

        # croping will change H, W, cx, cy, so need to change here
        if self.cfg['cam']['crop_edge'] > 0:
            self.H -= self.cfg['cam']['crop_edge']*2
            self.W -= self.cfg['cam']['crop_edge']*2
            self.cx -= self.cfg['cam']['crop_edge']
            self.cy -= self.cfg['cam']['crop_edge']

    def load_bound(self, cfg):
        """
        Pass the scene bound parameters to different decoders and self.

        Args:
            cfg (dict): parsed config dict.
        """
        # scale the bound if there is a global scaling factor
        self.bound = torch.from_numpy(
            np.array(cfg['mapping']['bound'])*self.scale)
        bound_divisable = cfg['grid_len']['bound_divisable']
        # enlarge the bound a bit to allow it divisable by bound_divisable
        self.bound[:, 1] = (((self.bound[:, 1]-self.bound[:, 0]) /
                            bound_divisable).int()+1)*bound_divisable+self.bound[:, 0]
        if self.method == 'nice':
            self.shared_decoders.bound = self.bound
            self.shared_decoders.middle_decoder.bound = self.bound
            self.shared_decoders.fine_decoder.bound = self.bound
            self.shared_decoders.color_decoder.bound = self.bound
            if self.coarse:
                self.shared_decoders.coarse_decoder.bound = self.bound*self.coarse_bound_enlarge
        elif self.method == 'omni_feat':
            self.shared_decoders.bound = self.bound
    
    def set_bound(self, cfg, bound):
        """
        Pass the scene bound parameters to different decoders and self.

        Args:
            cfg (dict): parsed config dict.
        """
        # scale the bound if there is a global scaling factor
        self.bound = torch.from_numpy((bound)*self.scale)
        bound_divisable = cfg['grid_len']['bound_divisable']
        # enlarge the bound a bit to allow it divisable by bound_divisable
        self.bound[:, 1] = (((self.bound[:, 1]-self.bound[:, 0]) /
                            bound_divisable).int()+1)*bound_divisable+self.bound[:, 0]
        if self.method == 'nice':
            self.shared_decoders.bound = self.bound
            self.shared_decoders.middle_decoder.bound = self.bound
            self.shared_decoders.fine_decoder.bound = self.bound
            self.shared_decoders.color_decoder.bound = self.bound
            if self.coarse:
                self.shared_decoders.coarse_decoder.bound = self.bound*self.coarse_bound_enlarge
        elif self.method == 'omni_feat':
            self.shared_decoders.bound = self.bound

    def load_pretrain(self, cfg):
        """
        Load parameters of pretrained ConvOnet checkpoints to the decoders.

        Args:
            cfg (dict): parsed config dict
        """

        if self.coarse:
            ckpt = torch.load(cfg['pretrained_decoders']['coarse'],
                              map_location=cfg['mapping']['device'])
            coarse_dict = {}
            for key, val in ckpt['model'].items():
                if ('decoder' in key) and ('encoder' not in key):
                    key = key[8:]
                    coarse_dict[key] = val
            self.shared_decoders.coarse_decoder.load_state_dict(coarse_dict, strict=False)

        ckpt = torch.load(cfg['pretrained_decoders']['middle_fine'],
                          map_location=cfg['mapping']['device'])
        middle_dict = {}
        fine_dict = {}
        for key, val in ckpt['model'].items():
            if ('decoder' in key) and ('encoder' not in key):
                if 'coarse' in key:
                    key = key[8+7:]
                    middle_dict[key] = val
                elif 'fine' in key:
                    key = key[8+5:]
                    fine_dict[key] = val
        self.shared_decoders.middle_decoder.load_state_dict(middle_dict, strict=False)
        self.shared_decoders.fine_decoder.load_state_dict(fine_dict, strict=False)
        
    def load_pretrain_full(self, cfg, ckpt_path):
        """
        Load parameters of pretrained checkpoints to the decoders and variables.

        Args:
            cfg (dict): parsed config dict
        """
        
        print('Get ckpt : ', ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=cfg['mapping']['device'])
        self.estimate_c2w_list = ckpt['estimate_c2w_list']
        self.gt_c2w_list = ckpt['gt_c2w_list']
        self.shared_c = ckpt['c']
        self.shared_decoders.load_state_dict(ckpt['decoder_state_dict'])

    def load_pretrain_frag(self, cfg, ckpt_path):
        print('Get ckpt : ', ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=cfg['mapping']['device'])
        self.shared_c = ckpt['c']
        self.shared_decoders.load_state_dict(ckpt['decoder_state_dict'])
        
        self.mapper.grufusion['grid_middle'].load_state_dict(ckpt['grufusion_state_dict']['grid_middle'])
        self.mapper.grufusion['grid_fine'].load_state_dict(ckpt['grufusion_state_dict']['grid_fine'])
        self.mapper.grufusion['grid_color'].load_state_dict(ckpt['grufusion_state_dict']['grid_color'])

        self.mapper.costfusion['grid_middle'].load_state_dict(ckpt['costfusion_state_dict']['grid_middle'])
        self.mapper.costfusion['grid_fine'].load_state_dict(ckpt['costfusion_state_dict']['grid_fine'])
        self.mapper.costfusion['grid_color'].load_state_dict(ckpt['costfusion_state_dict']['grid_color'])

    def grid_init(self, cfg):
        """
        Initialize the hierarchical feature grids.

        Args:
            cfg (dict): parsed config dict.
        """
        if self.coarse:
            coarse_grid_len = cfg['grid_len']['coarse']
            self.coarse_grid_len = coarse_grid_len
        middle_grid_len = cfg['grid_len']['middle']
        self.middle_grid_len = middle_grid_len
        fine_grid_len = cfg['grid_len']['fine']
        self.fine_grid_len = fine_grid_len
        color_grid_len = cfg['grid_len']['color']
        self.color_grid_len = color_grid_len

        c = {}
        # c_dim = cfg['model']['c_dim']
        c_dim = 64
        # c_dim = 1
        xyz_len = self.bound[:, 1]-self.bound[:, 0]
        bound = self.bound.clone()
        bound = bound[[2,1,0]]
        
        coord = {}
        grid_coord = {}
        vote = {}

        if self.coarse:
            coarse_key = 'grid_coarse'
            coarse_val_shape = list(
                map(int, (xyz_len*self.coarse_bound_enlarge/coarse_grid_len).tolist()))
            coarse_val_shape[0], coarse_val_shape[2] = coarse_val_shape[2], coarse_val_shape[0]
            self.coarse_val_shape = coarse_val_shape
            val_shape = [1, c_dim, *coarse_val_shape]
            coarse_val = torch.zeros(val_shape)
            c[coarse_key] = coarse_val
            
            # coarse_val_shape[0], coarse_val_shape[2] = coarse_val_shape[2], coarse_val_shape[0]
            grid_range = [torch.arange(0, n_vox) + 0.5 for n_vox in coarse_val_shape]
            coarse_coord_grid = torch.stack(torch.meshgrid(*grid_range)) * self.coarse_grid_len
            coarse_coord_grid = coarse_coord_grid.reshape(3, -1) + bound[:, 0].unsqueeze(-1) * self.coarse_bound_enlarge
            coord[coarse_key] = coarse_coord_grid.reshape(1, 3, *coarse_val_shape)
            
            grid_range = [torch.arange(0, n_vox) for n_vox in coarse_val_shape]
            coarse_coord_grid = torch.stack(torch.meshgrid(*grid_range))
            grid_coord[coarse_key] = coarse_coord_grid.reshape(1, 3, *coarse_val_shape).type(torch.int)
            
            vote[coarse_key] = torch.zeros(val_shape)

        middle_key = 'grid_middle'
        middle_val_shape = list(map(int, (xyz_len/middle_grid_len).tolist()))
        middle_val_shape[0], middle_val_shape[2] = middle_val_shape[2], middle_val_shape[0]
        self.middle_val_shape = middle_val_shape
        val_shape = [1, c_dim, *middle_val_shape]
        middle_val = torch.zeros(val_shape)
        c[middle_key] = middle_val
        
        # middle_val_shape[0], middle_val_shape[2] = middle_val_shape[2], middle_val_shape[0]
        grid_range = [torch.arange(0, n_vox) + 0.5 for n_vox in middle_val_shape]
        middle_coord_grid = torch.stack(torch.meshgrid(*grid_range)) * self.middle_grid_len
        middle_coord_grid = middle_coord_grid.reshape(3, -1) + bound[:, 0].unsqueeze(-1)
        coord[middle_key] = middle_coord_grid.reshape(1, 3, *middle_val_shape)
        
        grid_range = [torch.arange(0, n_vox) for n_vox in middle_val_shape]
        middle_coord_grid = torch.stack(torch.meshgrid(*grid_range))
        grid_coord[middle_key] = middle_coord_grid.reshape(1, 3, *middle_val_shape).type(torch.int)
        
        vote[middle_key] = torch.zeros(val_shape)

        fine_key = 'grid_fine'
        fine_val_shape = list(map(int, (xyz_len/fine_grid_len).tolist()))
        fine_val_shape[0], fine_val_shape[2] = fine_val_shape[2], fine_val_shape[0]
        self.fine_val_shape = fine_val_shape
        val_shape = [1, c_dim, *fine_val_shape]
        fine_val = torch.zeros(val_shape)
        c[fine_key] = fine_val
        
        # fine_val_shape[0], fine_val_shape[2] = fine_val_shape[2], fine_val_shape[0]
        grid_range = [torch.arange(0, n_vox) + 0.5 for n_vox in fine_val_shape]
        fine_coord_grid = torch.stack(torch.meshgrid(*grid_range)) * self.fine_grid_len
        fine_coord_grid = fine_coord_grid.reshape(3, -1) + bound[:, 0].unsqueeze(-1)
        coord[fine_key] = fine_coord_grid.reshape(1, 3, *fine_val_shape)
        
        grid_range = [torch.arange(0, n_vox) for n_vox in fine_val_shape]
        fine_coord_grid = torch.stack(torch.meshgrid(*grid_range))
        grid_coord[fine_key] = fine_coord_grid.reshape(1, 3, *fine_val_shape).type(torch.int)
        
        vote[fine_key] = torch.zeros(val_shape)

        color_key = 'grid_color'
        color_val_shape = list(map(int, (xyz_len/color_grid_len).tolist()))
        color_val_shape[0], color_val_shape[2] = color_val_shape[2], color_val_shape[0]
        self.color_val_shape = color_val_shape
        val_shape = [1, c_dim, *color_val_shape]
        color_val = torch.ones(val_shape)
        c[color_key] = color_val
        
        # color_val_shape[0], color_val_shape[2] = color_val_shape[2], color_val_shape[0]
        grid_range = [torch.arange(0, n_vox) + 0.5 for n_vox in color_val_shape]
        color_coord_grid = torch.stack(torch.meshgrid(*grid_range)) * self.color_grid_len
        color_coord_grid = color_coord_grid.reshape(3, -1) + bound[:, 0].unsqueeze(-1)
        coord[color_key] = color_coord_grid.reshape(1, 3, *color_val_shape)
        
        grid_range = [torch.arange(0, n_vox) for n_vox in color_val_shape]
        color_coord_grid = torch.stack(torch.meshgrid(*grid_range))
        grid_coord[color_key] = color_coord_grid.reshape(1, 3, *color_val_shape).type(torch.int)
        
        vote[color_key] = torch.zeros(val_shape)
        
        self.shared_c = c
        self.shared_coord = coord
        self.shared_grid_coord = grid_coord
        self.shared_vote = vote

    def run(self):
        if self.verbose:
            print(Fore.GREEN)
            print("Run OmniMVS...")
            print(Style.RESET_ALL)
        
        for i, data in tqdm(enumerate(self.dbloader)):
            self.integrator.run_omni(data)
            
        del self.integrator.mvsnet
        del self.integrator.omnimvs
            
        for epoch in tqdm(range(500)):
            # self.coarse_mapper.run_omni(epoch)
            self.mapper.run_omni(epoch)

    def run_tmp(self):
        self.local_frag_idxs = self.integrator.local_frag_idxs
        self.local_frag_bounds = self.integrator.local_frag_bounds
        self.n_fragments = len(self.local_frag_idxs)
        num_iters = self.cfg['mapping']['iters']
        joint_num_iters = self.cfg['mapping']['joint_iters']

        frag_datas = []

        print('\nIntegrate OmniMVS Feature...')
        for frag_idx in tqdm(range(self.n_fragments)):
            # init bound / grid
            local_frag_bound = self.local_frag_bounds[frag_idx]
            self.set_bound(self.cfg, local_frag_bound)
            self.grid_init(self.cfg)

            for key, val in self.shared_c.items():
                val = val.to(self.cfg['omnimvs']['device'])
                val.share_memory_()
                self.shared_c[key] = val
            for key, val in self.shared_coord.items():
                val = val.to(self.cfg['omnimvs']['device'])
                val.share_memory_()
                self.shared_coord[key] = val
            for key, val in self.shared_grid_coord.items():
                val = val.to(self.cfg['omnimvs']['device'])
                val.share_memory_()
                self.shared_grid_coord[key] = val
            for key, val in self.shared_vote.items():
                val = val.to(self.cfg['omnimvs']['device'])
                val.share_memory_()
                self.shared_vote[key] = val

            self.target_depths = []
            self.target_colors = []
            self.use_depths = []
            self.use_colors = []
            self.target_entropys = []
            self.resps = []
            self.backproj_feats = {}
            self.backproj_feats['grid_middle'] = []
            self.backproj_feats['grid_fine'] = []
            self.backproj_feats['grid_color'] = []
            self.raw_feats = []
            self.probs = []
            self.target_c2w = []
            
            self.integrator.init_params(self)

            for i in range(len(self.local_frag_idxs[frag_idx])):
                idx = self.local_frag_idxs[frag_idx][i]
                data = BatchCollator.collate([self.integrator.omnimvs[idx]])
                self.integrator.run_omni(data)

            frag_datas.append(
                {
                    'target_depths': self.integrator.target_depths,
                    'target_colors': self.integrator.target_colors,
                    'use_depths': self.integrator.use_depths,
                    'use_colors': self.integrator.use_colors,
                    'target_entropys': self.integrator.target_entropys,
                    'target_c2w': self.integrator.target_c2w,
                    'backproj_feats': self.integrator.backproj_feats,
                    'bound': self.bound,
                    'c': self.integrator.c,
                    'coord': self.integrator.coord,
                }
            )
            torch.cuda.empty_cache()

        print('\nMapping...')
        frag_idx_list = [i for i in range(self.n_fragments)]
        for epoch in tqdm(range(100)):
            for iter in tqdm(range(num_iters // joint_num_iters)):
                random.shuffle(frag_idx_list)
                for idx, frag_idx in enumerate(frag_idx_list):
                    local_frag_bound = self.local_frag_bounds[frag_idx]
                    self.set_bound(self.cfg, local_frag_bound)
                    self.grid_init(self.cfg)

                    self.target_depths = frag_datas[frag_idx]['target_depths']
                    self.target_colors = frag_datas[frag_idx]['target_colors']
                    self.use_depths = frag_datas[frag_idx]['use_depths']
                    self.use_colors = frag_datas[frag_idx]['use_colors']
                    self.target_entropys = frag_datas[frag_idx]['target_entropys']
                    self.target_c2w = frag_datas[frag_idx]['target_c2w']
                    self.backproj_feats = frag_datas[frag_idx]['backproj_feats']
                    self.shared_c = frag_datas[frag_idx]['c']
                    self.shared_coord = frag_datas[frag_idx]['coord']
                
                    self.integrator.init_params(self)
                    self.mapper.init_params(self)
                    self.renderer.bound = self.bound
                    self.mesher.bound = self.bound
                    self.mesher.marching_cubes_bound = self.bound
                    # extend bound (for meshing)
                    self.mesher.marching_cubes_bound[0, 0] -= 1
                    self.mesher.marching_cubes_bound[2, 0] -= 1
                    self.mesher.marching_cubes_bound[0, 1] += 1
                    self.mesher.marching_cubes_bound[2, 1] += 1


                    self.mapper.run_omni_frag(epoch, iter, frag_idx, idx, save_log=(iter == (num_iters//joint_num_iters)-1))

                    torch.cuda.empty_cache()
                


# This part is required by torch.multiprocessing
if __name__ == '__main__':
    pass
