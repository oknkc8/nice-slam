import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.common import get_camera_from_tensor
from src.omni_utils.common import *
from src.omni_utils.image import *

import pdb


class Visualizer(object):
    """
    Visualize itermediate results, render out depth, color and depth uncertainty images.
    It can be called per iteration, which is good for debuging (to see how each tracking/mapping iteration performs).

    """

    def __init__(self, freq, inside_freq, vis_dir, renderer, verbose, device='cuda:0'):
        self.freq = freq
        self.device = device
        self.vis_dir = vis_dir
        self.verbose = verbose
        self.renderer = renderer
        self.inside_freq = inside_freq
        os.makedirs(f'{vis_dir}', exist_ok=True)

    def vis(self, idx, iter, gt_depth, gt_color, c2w_or_camera_tensor, c,
            decoders, summary_writer=None):
        """
        Visualization of depth, color images and save to file.

        Args:
            idx (int): current frame index.
            iter (int): the iteration number.
            gt_depth (tensor): ground truth depth image of the current frame.
            gt_color (tensor): ground truth color image of the current frame.
            c2w_or_camera_tensor (tensor): camera pose, represented in 
                camera to world matrix or quaternion and translation tensor.
            c (dicts): feature grids.
            decoders (nn.module): decoders.
        """
        iter += 1
        with torch.no_grad():
            if (idx % self.freq == 0) and (iter % self.inside_freq == 0):
                gt_depth_np = gt_depth.cpu().numpy()
                gt_color_np = gt_color.cpu().numpy()
                if len(c2w_or_camera_tensor.shape) == 1:
                    bottom = torch.from_numpy(
                        np.array([0, 0, 0, 1.]).reshape([1, 4])).type(
                            torch.float32).to(self.device)
                    c2w = get_camera_from_tensor(
                        c2w_or_camera_tensor.clone().detach())
                    c2w = torch.cat([c2w, bottom], dim=0)
                else:
                    c2w = c2w_or_camera_tensor

                depth, uncertainty, color = self.renderer.render_img(
                    c,
                    decoders,
                    c2w,
                    self.device,
                    stage='color',
                    gt_depth=gt_depth)
                depth_np = depth.detach().cpu().numpy()
                color_np = color.detach().cpu().numpy()
                depth_residual = np.abs(gt_depth_np - depth_np)
                depth_residual[gt_depth_np == 0.0] = 0.0
                color_residual = np.abs(gt_color_np - color_np)
                color_residual[gt_depth_np == 0.0] = 0.0

                # fig, axs = plt.subplots(3, 3)
                # fig.tight_layout(h_pad=2, w_pad=2)
                max_depth = np.max(gt_depth_np)
                min_depth = np.unique(gt_depth_np)[1]
                # axs[0, 0].imshow(gt_depth_np, cmap="plasma",
                #                  vmin=0, vmax=max_depth)
                # axs[0, 0].set_title('Input Depth')
                # axs[0, 0].set_xticks([])
                # axs[0, 0].set_yticks([])
                # axs[0, 1].imshow(depth_np, cmap="plasma",
                #                  vmin=0, vmax=max_depth)
                # axs[0, 1].set_title('Generated Depth')
                # axs[0, 1].set_xticks([])
                # axs[0, 1].set_yticks([])
                # axs[0, 2].imshow(depth_residual, cmap="plasma",
                #                  vmin=0, vmax=max_depth)
                # axs[0, 2].set_title('Depth Residual')
                # axs[0, 2].set_xticks([])
                # axs[0, 2].set_yticks([])
                
                tmp_gt_depth_np = gt_depth_np.copy()
                gt_depth_np[tmp_gt_depth_np == 0] = EPS
                depth_np[tmp_gt_depth_np == 0] = EPS
                depth_residual[tmp_gt_depth_np == 0] = EPS
                gt_invdepth_np = colorMap('oliver', 1/gt_depth_np, 0, 1/min_depth)
                invdepth_np = colorMap('oliver', 1/depth_np, 0, 1/min_depth)
                invdepth_residual = colorMap('oliver', 1/depth_residual, 1/max_depth, 1/np.min(depth_residual))
                
                # axs[1, 0].imshow(gt_invdepth_np)
                # axs[1, 0].set_title('Input InvDepth')
                # axs[1, 0].set_xticks([])
                # axs[1, 0].set_yticks([])
                # axs[1, 1].imshow(invdepth_np)
                # axs[1, 1].set_title('Generated InvDepth')
                # axs[1, 1].set_xticks([])
                # axs[1, 1].set_yticks([])
                # axs[1, 2].imshow(invdepth_residual)
                # axs[1, 2].set_title('Inv Depth Residual')
                # axs[1, 2].set_xticks([])
                # axs[1, 2].set_yticks([])
                
                gt_color_np = np.clip(gt_color_np, 0, 1)
                color_np = np.clip(color_np, 0, 1)
                color_residual = np.clip(color_residual, 0, 1)
                # axs[2, 0].imshow(gt_color_np, cmap="plasma")
                # axs[2, 0].set_title('Input RGB')
                # axs[2, 0].set_xticks([])
                # axs[2, 0].set_yticks([])
                # axs[2, 1].imshow(color_np, cmap="plasma")
                # axs[2, 1].set_title('Generated RGB')
                # axs[2, 1].set_xticks([])
                # axs[2, 1].set_yticks([])
                # axs[2, 2].imshow(color_residual, cmap="plasma")
                # axs[2, 2].set_title('RGB Residual')
                # axs[2, 2].set_xticks([])
                # axs[2, 2].set_yticks([])
                # plt.subplots_adjust(wspace=0, hspace=0)
                # plt.savefig(
                #     f'{self.vis_dir}/{idx:05d}_{iter:04d}.jpg', bbox_inches='tight', pad_inches=0.2)
                # plt.clf()
                
                gt_depth_np = gt_depth_np.astype(np.float64)
                gt_color_np = gt_color_np.astype(np.float64)
                color_np = color_np.astype(np.float64)
                color_residual = color_residual.astype(np.float64)
                
                # tmp_gt_depth_np = gt_depth_np.copy()
                # gt_depth_np[tmp_gt_depth_np == 0] = EPS
                # depth_np[tmp_gt_depth_np == 0] = EPS
                # depth_residual[tmp_gt_depth_np == 0] = EPS
                # gt_invdepth_np = colorMap('oliver', 1/gt_depth_np, 0, 1/min_depth)
                # invdepth_np = colorMap('oliver', 1/depth_np, 0, 1/min_depth)
                # invdepth_residual = colorMap('oliver', 1/depth_residual, 0, 1/min_depth)
                
                gt_depth_np = colorMap('plasma', gt_depth_np, 0)
                depth_np = colorMap('plasma', depth_np, 0)
                depth_residual = colorMap('plasma', depth_residual, 0)
                                
                prev_1 = cv2.hconcat([gt_depth_np.astype(np.float64), depth_np.astype(np.float64), depth_residual.astype(np.float64)])
                prev_2 = cv2.hconcat([gt_invdepth_np.astype(np.float64), invdepth_np.astype(np.float64), invdepth_residual.astype(np.float64)])
                prev_3 = cv2.hconcat([gt_color_np*255, color_np*255, color_residual*255])
                                
                prev_output = np.round(cv2.vconcat([prev_1, prev_2, prev_3])).astype(np.uint8)
                prev_output = cv2.cvtColor(prev_output, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f'{self.vis_dir}/output_{idx:05d}_{iter:04d}.png', prev_output)
                
                if summary_writer is not None:
                    prev_output = cv2.cvtColor(prev_output, cv2.COLOR_BGR2RGB)
                    prev_output = torch.tensor(prev_output, dtype=torch.float64).permute(2,0,1) / 255
                    summary_writer.add_image(f'{iter}/optimize', prev_output, idx)
                
                del prev_output
                
                if self.verbose:
                    print(
                        f'Saved rendering visualization of color/depth image at {self.vis_dir}/{idx:05d}_{iter:04d}.jpg')


    def vis_omni(self, epoch, iter, num_iter, use_depth, use_color, c2w_or_camera_tensor, c,
            decoders, stage, summary_writer=None, gt_depth=None, gt_color=None):
        """
        Visualization of depth, color images and save to file.

        Args:
            idx (int): current frame index.
            iter (int): the iteration number.
            gt_depth (tensor): ground truth depth image of the current frame.
            gt_color (tensor): ground truth color image of the current frame.
            c2w_or_camera_tensor (tensor): camera pose, represented in 
                camera to world matrix or quaternion and translation tensor.
            c (dicts): feature grids.
            decoders (nn.module): decoders.
        """
        iter += 1
        with torch.no_grad():
            if iter % self.inside_freq == 0:
                if gt_depth is None:
                    gt_depth = use_depth
                    gt_color = use_color
                gt_depth_np = gt_depth.cpu().numpy()
                gt_color_np = gt_color.cpu().numpy()
                if len(c2w_or_camera_tensor.shape) == 1:
                    bottom = torch.from_numpy(
                        np.array([0, 0, 0, 1.]).reshape([1, 4])).type(
                            torch.float32).to(self.device)
                    c2w = get_camera_from_tensor(
                        c2w_or_camera_tensor.clone().detach())
                    c2w = torch.cat([c2w, bottom], dim=0)
                else:
                    c2w = c2w_or_camera_tensor

                depth, uncertainty, color = self.renderer.render_img(
                    c,
                    decoders,
                    c2w,
                    self.device,
                    stage='color',
                    gt_depth=use_depth)
                    # gt_depth=None)
                depth_np = depth.detach().cpu().numpy()
                color_np = color.detach().cpu().numpy()
                depth_residual = np.abs(gt_depth_np - depth_np)
                depth_residual[gt_depth_np == 0.0] = 0.0
                color_residual = np.abs(gt_color_np - color_np)
                color_residual[gt_depth_np == 0.0] = 0.0

                max_depth = min(np.max(gt_depth_np), np.max(depth_np)) 
                min_depth = max(np.unique(gt_depth_np)[1], np.unique(depth_np)[1])
                
                gt_max_depth = np.max(gt_depth_np)
                gt_min_depth = np.unique(gt_depth_np)[1]
                
                gen_max_depth = np.max(depth_np)
                gen_min_depth = np.unique(depth_np)[1]
                
                tmp_gt_depth_np = gt_depth_np.copy()
                gt_depth_np[tmp_gt_depth_np == 0] = EPS
                depth_np[tmp_gt_depth_np == 0] = EPS
                depth_residual[tmp_gt_depth_np == 0] = EPS
                invdepth_np_2 = colorMap('oliver', 1/depth_np, 1/gen_max_depth, 1/gen_min_depth)
                gt_invdepth_np = colorMap('oliver', 1/gt_depth_np, 1/max_depth, 1/min_depth)
                invdepth_np = colorMap('oliver', 1/depth_np, 1/max_depth, 1/min_depth)
                # gt_invdepth_np = colorMap('oliver', 1/gt_depth_np, 1/gt_max_depth, 1/gt_min_depth)
                # invdepth_np = colorMap('oliver', 1/depth_np, 1/gen_max_depth, 1/gen_min_depth)
                invdepth_residual = colorMap('oliver', 1/depth_residual, 1/np.max(depth_residual), 1/(np.unique(depth_residual)[1]))
                
                gt_color_np = np.clip(gt_color_np, 0, 1)
                color_np = np.clip(color_np, 0, 1)
                color_residual = np.clip(color_residual, 0, 1)
                
                gt_depth_np = gt_depth_np.astype(np.float64)
                gt_color_np = gt_color_np.astype(np.float64)
                color_np = color_np.astype(np.float64)
                color_residual = color_residual.astype(np.float64)
                
                depth_np_2 = colorMap('plasma', depth_np, gen_min_depth, gen_max_depth)
                gt_depth_np = colorMap('plasma', gt_depth_np, min_depth, max_depth)
                depth_np = colorMap('plasma', depth_np, min_depth, max_depth)
                # gt_depth_np = colorMap('plasma', gt_depth_np, gt_min_depth, gt_max_depth)
                # depth_np = colorMap('plasma', depth_np, gen_min_depth, gen_max_depth)
                depth_residual = colorMap('plasma', depth_residual, 0, np.max(depth_residual))
                                
                # prev_1 = cv2.hconcat([gt_depth_np.astype(np.float64), depth_np.astype(np.float64), depth_residual.astype(np.float64)])
                # prev_2 = cv2.hconcat([gt_invdepth_np.astype(np.float64), invdepth_np.astype(np.float64), invdepth_residual.astype(np.float64)])
                prev_1 = cv2.hconcat([gt_depth_np.astype(np.float64), depth_np.astype(np.float64), depth_np_2.astype(np.float64)])
                prev_2 = cv2.hconcat([gt_invdepth_np.astype(np.float64), invdepth_np.astype(np.float64), invdepth_np_2.astype(np.float64)])
                prev_3 = cv2.hconcat([gt_color_np*255, color_np*255, color_residual*255])
                                
                prev_output = np.round(cv2.vconcat([prev_1, prev_2, prev_3])).astype(np.uint8)
                prev_output = cv2.cvtColor(prev_output, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f'{self.vis_dir}/{stage}_{epoch:05d}_{iter:04d}.png', prev_output)
                
                if summary_writer is not None:
                    prev_output = cv2.cvtColor(prev_output, cv2.COLOR_BGR2RGB)
                    prev_output = torch.tensor(prev_output, dtype=torch.float64).permute(2,0,1) / 255
                    summary_writer.add_image(f'{stage}', prev_output, epoch * num_iter + iter)
                
                del prev_output
                
                if self.verbose:
                    print(
                        f'\nSaved rendering visualization of color/depth image at {stage}_{epoch:05d}_{iter:04d}.png')
