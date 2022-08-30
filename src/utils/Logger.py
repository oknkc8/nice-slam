import os

import torch


class Logger(object):
    """
    Save checkpoints to file.

    """

    def __init__(self, cfg, args, slam
                 ):
        self.verbose = slam.verbose
        self.ckptsdir = slam.ckptsdir
        self.shared_c = slam.shared_c
        self.gt_c2w_list = slam.gt_c2w_list
        self.shared_decoders = slam.shared_decoders
        self.estimate_c2w_list = slam.estimate_c2w_list

    def log(self, idx, keyframe_dict, keyframe_list, selected_keyframes=None):
        path = os.path.join(self.ckptsdir, '{:05d}.tar'.format(idx))
        torch.save({
            'c': self.shared_c,
            'decoder_state_dict': self.shared_decoders.state_dict(),
            'gt_c2w_list': self.gt_c2w_list,
            'estimate_c2w_list': self.estimate_c2w_list,
            'keyframe_list': keyframe_list,
            # 'keyframe_dict': keyframe_dict, # to save keyframe_dict into ckpt, uncomment this line
            'selected_keyframes': selected_keyframes,
            'idx': idx,
        }, path, _use_new_zipfile_serialization=False)

        if self.verbose:
            print('Saved checkpoints at', path)
            
    def log_omni(self, epoch, costfusion, grufusion, optimizer):
        path = os.path.join(self.ckptsdir, '{:05d}.tar'.format(epoch))
        torch.save({
            'c': self.shared_c,
            'decoder_state_dict': self.shared_decoders.state_dict(),
            'grufusion_state_dict': {
                'grid_middle': grufusion['grid_middle'].state_dict(),
                'grid_fine': grufusion['grid_fine'].state_dict(),
                'grid_color': grufusion['grid_color'].state_dict(),
            },
            'costfusion_state_dict': {
                'grid_middle': costfusion['grid_middle'].state_dict(),
                'grid_fine': costfusion['grid_fine'].state_dict(),
                'grid_color': costfusion['grid_color'].state_dict(),
            },
            'optimizer_state_dict': optimizer.state_dict(),
        }, path, _use_new_zipfile_serialization=False)

        if self.verbose:
            print('Saved checkpoints at', path)

