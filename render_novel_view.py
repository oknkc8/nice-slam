import argparse as Argparse
import os
import time

import numpy as np
import torch
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader

from src import config
from src.NICE_SLAM import NICE_SLAM
from src.tools.viz import SLAMFrontend
from src.utils.datasets import get_dataset
from src.omni_utils.geometry import *
from src.omni_utils.image import *

if __name__ == '__main__':

    parser = Argparse.ArgumentParser(
        description='Arguments to visualize the SLAM process.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one inconfig file')
    nice_parser = parser.add_mutually_exclusive_group(required=False)
    nice_parser.add_argument('--nice', dest='nice', action='store_true')
    nice_parser.add_argument('--imap', dest='nice', action='store_false')
    parser.set_defaults(nice=True)
    parser.add_argument('--save_rendering',
                        action='store_true', help='save rendering video to `vis.mp4` in output folder ')
    parser.add_argument('--vis_input_frame',
                        action='store_true', help='visualize input frames')
    parser.add_argument('--no_gt_traj',
                        action='store_true', help='not visualize gt trajectory')
    args = parser.parse_args()
    cfg = config.load_config(
        args.config, 'configs/nice_slam.yaml' if args.nice else 'configs/imap.yaml')
    scale = cfg['scale']
    output = cfg['data']['output'] if args.output is None else args.output
    if args.vis_input_frame:
        frame_reader = get_dataset(cfg, args, scale, device='cpu')
        frame_loader = DataLoader(
            frame_reader, batch_size=1, shuffle=False, num_workers=4)
    ckptsdir = f'{output}/ckpts'
    if os.path.exists(ckptsdir):
        ckpts = [os.path.join(ckptsdir, f)
                 for f in sorted(os.listdir(ckptsdir)) if 'tar' in f]
        if len(ckpts) > 0:
            ckpt_path = ckpts[-1]


    slam = NICE_SLAM(cfg, args)
    slam.load_pretrain_full(cfg, ckpt_path)
    slam.renderer.cam_method = 'perspective'
    slam.renderer.N_importance = 64
    slam.renderer.H = 264
    slam.renderer.W = 416
    slam.renderer.fx = 108.01766298948733
    slam.renderer.fy = 121.0972625615004
    slam.renderer.cx = 207.5
    slam.renderer.cy = 131.5
    
    c2w = slam.estimate_c2w_list[0]
    sign = 1
    for i in range(len(slam.estimate_c2w_list)):
        import pdb
        if i%5 == 0 and i != 0:
            sign *= -1
        
        # pdb.set_trace()
        c2w[:, 3] = slam.estimate_c2w_list[i][:, 3]
        c2w = c2w.detach().cpu().numpy()
        rotated_c2w = np.eye(4)
        rotated_c2w[:3, :] = rotateAxis(c2w, sign*2, -10, 0)
        c2w = torch.tensor(rotated_c2w).to('cuda:0')
        depth, _, color = slam.renderer.render_img(slam.shared_c, slam.shared_decoders, c2w, device='cuda:0', stage='color')
        
        depth = depth.detach().cpu().numpy()
        color = color.detach().cpu().numpy()
        depth[depth == 0] = EPS
        invdepth = colorMap('oliver', 1/depth, 0, 1/np.min(depth))
        
        invdepth = invdepth.astype(np.float64)
        color = np.clip(color, 0, 1)
        color = color.astype(np.float64) * 255
        prev_output = cv2.hconcat([color, invdepth]).astype(np.uint8)        
        
        prev_output = cv2.cvtColor(prev_output, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(f'novel_view_results_2/{i:04d}.png', prev_output)
        cv2.imshow('Novel View', prev_output)
        cv2.waitKey(1)