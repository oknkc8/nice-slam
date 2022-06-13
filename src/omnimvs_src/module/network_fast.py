# network.py
#
# Author: Changhee Won (chwon@hanyang.ac.kr)
#
#
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from src.omnimvs_src.module.basic import *
# from module.network import CostCompute

# from PSMNET's SPP
# class FeatureLayers(torch.nn.Module):
#     def __init__(self, CH=32, use_rgb=False):
#         super(FeatureLayers, self).__init__()
#         layers = []
#         in_channel = 3 if use_rgb else 1
#         layers.append(Conv2D(in_channel,CH,3,2,1)) # conv[0_1]
#         layers += [Conv2D(CH,CH,3,1,1) for _ in range(2)] # conv[0_2-3]

#         for _ in range(3):
#             layers += [Conv2D(CH,CH,3,1,1) for _ in range(2)] #conv[1_x]

#         layers.append(Conv2D(CH,2 * CH,3,2,1)) # conv[2_1]
#         layers.append(Conv2D(2 * CH, 2 * CH,3,1,1)) # conv[2_2]
#         for _ in range(15): # rest conv[2_x] * 15
#             layers += [Conv2D(2 * CH,2 * CH,3,1,1) for _ in range(2)]

#         layers.append(Conv2D(2 * CH, 4 * CH,3,1,2, dilation=2)) # conv[3_1]
#         layers.append(Conv2D(4 * CH, 4 * CH,3,1,2, dilation=2)) # conv[3_2]
#         for _ in range(2): # rest conv[3_x] * 2
#             layers += [Conv2D(4 * CH, 4 * CH,3,1,2, dilation=2) for _ in range(2)] #conv[3_x]

#         for _ in range(3): # conv[4_x] * 3
#             layers += [Conv2D(4 * CH, 4 * CH,3,1,4, dilation=4) for _ in range(2)] #conv[4_x]
#         self.layers = torch.nn.ModuleList(layers)

#         self.branch1 = torch.nn.Sequential(torch.nn.AvgPool2d((64, 64), stride=(64,64)),
#                                     Conv2D(4 * CH, CH, 1, 1, 0, 1))
#         self.branch2 = torch.nn.Sequential(torch.nn.AvgPool2d((32, 32), stride=(32,32)),
#                                     Conv2D(4 * CH, CH, 1, 1, 0, 1))
#         self.branch3 = torch.nn.Sequential(torch.nn.AvgPool2d((16, 16), stride=(16,16)),
#                                     Conv2D(4 * CH, CH, 1, 1, 0, 1))
#         self.branch4 = torch.nn.Sequential(torch.nn.AvgPool2d((8, 8), stride=(8,8)),
#                                     Conv2D(4 * CH, CH, 1, 1, 0, 1))
#         self.lastconv = torch.nn.Sequential(Conv2D(10*CH, 4*CH, 3, 1, 1),
#                                     Conv2D(4* CH, CH, 1, 1, 0, 1, bn=False, relu=False))

#     def forward(self, im):
#         x = self.layers[0](im)
#         x = self.layers[2](self.layers[1](x)) # conv0

#         for i in range(3, 41, 2): #conv1, 2
#             x_ = self.layers[i](x)
#             x = self.layers[i+1](x_, residual=x_)
#         conv2_16 = x

#         for i in range(41,53,2):
#             x_ = self.layers[i](x)
#             x = self.layers[i+1](x_,residual=x_)

#         b1 = self.branch1(x)
#         b1 = F.upsample(b1, (x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
#         b2 = self.branch2(x)
#         b2 = F.upsample(b2, (x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
#         b3 = self.branch3(x)
#         b3 = F.upsample(b3, (x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
#         b4 = self.branch4(x)
#         b4 = F.upsample(b4, (x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
#         x = torch.cat((conv2_16, x, b4, b3, b2, b1), 1)
#         x = self.lastconv(x)
#         return x

class FeatureLayers(torch.nn.Module):
    def __init__(self, CH=32, use_rgb=False):
        super(FeatureLayers, self).__init__()
        layers = []
        in_channel = 3 if use_rgb else 1
        layers.append(Conv2D(in_channel,CH,5,2,2)) # conv[1]
        layers += [Conv2D(CH,CH,3,1,1) for _ in range(6)] # conv[2-7]
        layers.append(Conv2D(CH,CH,3,2,1)) #conv [8]
        layers += [Conv2D(CH,CH,3,1,1) for _ in range(6)] # conv[9-14]
        for d in range(2,5): # conv[15-20]
            layers += [Conv2D(CH,CH,3,1,d,dilation=d) for _ in range(2)]
        layers.append(Conv2D(CH,CH,3,1,1,bn=False,relu=False)) # conv[21]
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, im):
        x = self.layers[0](im)
        for i in range(1,7,2):
            x_ = self.layers[i](x)
            x = self.layers[i+1](x_,residual=x)
        x = self.layers[7](x)
        for i in range(8,20,2):
            x_ = self.layers[i](x)
            x = self.layers[i+1](x_,residual=x)
        x = self.layers[20](x)
        return x

class SphericalSweep(torch.nn.Module):
    def __init__(self, CH=32, circular_pad=False):
        super(SphericalSweep, self).__init__()
        _Conv2D = HorizontalCircularConv2D if circular_pad else Conv2D
        self.transfer_conv = _Conv2D(CH,CH,3,1,1,bn=False,relu=False)

    def forward(self, feature, grid):
        num_invdepth = grid.shape[1]
        sweep = [F.grid_sample(feature, grid[:,d,...],
            align_corners=True).unsqueeze(1) for d in range(0, num_invdepth)]
        sweep = torch.cat(sweep, 1) # -> B x N/2 x CH x H x W
        B, N, C, H, W = sweep.shape
        sweep = sweep.view((B * N, C, H, W))
        spherical_feature = self.transfer_conv(sweep).view(
            (B, N, C, H, W))
        return spherical_feature


class CostCompute(torch.nn.Module):
    def __init__(self, CH=32, circular_pad=False):
        super(CostCompute, self).__init__()
        _Conv3D = HorizontalCircularConv3D if circular_pad else Conv3D
        _SeparableConv3D = HCircularSeparableConv3D if circular_pad \
            else SeparableConv3D
        CH *= 2
        self.fusion = _Conv3D(2*CH,CH,3,1,1)
        convs = []
        convs += [_SeparableConv3D(CH,CH,3,1,1)]
        convs += [_Conv3D(CH,2*CH,3,2,1),
                        _SeparableConv3D(2*CH,2*CH,3,1,1)]
        convs += [_Conv3D(2*CH,2*CH,3,2,1),
                        _SeparableConv3D(2*CH,2*CH,3,1,1)]
        convs += [_Conv3D(2*CH,2*CH,3,2,1),
                        _SeparableConv3D(2*CH,2*CH,3,1,1)]
        convs += [_Conv3D(2*CH,4*CH,3,(1,2,2),1),
                        _SeparableConv3D(4*CH,4*CH,3,1,1)]

        self.convs = torch.nn.ModuleList(convs)
        self.deconv1 = DeConv3D(4*CH,2*CH,3,2,1,out_pad=1)
        self.deconv2 = DeConv3D(2*CH,2*CH,3,2,1,out_pad=1)
        self.deconv3 = DeConv3D(2*CH,2*CH,3,2,1,out_pad=1)
        self.deconv4 = DeConv3D(2*CH,CH,3,2,1,out_pad=1)
        self.deconv5 = DeConv3D(CH,1,3,2,1,out_pad=1,bn=False,relu=False)

    def forward(self, feats):
        c = self.fusion(feats) # 2 2 4
        c1 = self.convs[0](c) # 2 2 4
        c1 = F.interpolate(c1, scale_factor=(2, 1, 1),
            mode='trilinear', align_corners=True) # 2 2 2
        c = self.convs[1](c) # 4 4 8
        c2 = self.convs[2](c) # 4 4 8
        c2 = F.interpolate(c2, scale_factor=(2, 1, 1),
            mode='trilinear', align_corners=True) # 4 4 4
        c = self.convs[3](c) # 8 8 16
        c3 = self.convs[4](c) # 8 8 16
        c3 = F.interpolate(c3, scale_factor=(2, 1, 1),
            mode='trilinear', align_corners=True) # 8 8 8
        c = self.convs[5](c) # 16 16 32
        c4 = self.convs[6](c) # 16 16 32
        c4 = F.interpolate(c4, scale_factor=(2, 1, 1),
            mode='trilinear', align_corners=True)# 161616
        c = self.convs[7](c) # 8 8 16
        c5 = self.convs[8](c) # 8 8 16

        c = self.deconv1(c5, residual=c4) # 16 16 16
        c = self.deconv2(c, residual=c3) # 8 8 8
        c = self.deconv3(c, residual=c2) # 4 4 4
        c = self.deconv4(c, residual=c1) # 2 2 2
        costs = self.deconv5(c) # 1 1 1
        return costs

class OmniMVSNet(torch.nn.Module):

    def __init__(self, varargin=None):
        super(OmniMVSNet, self).__init__()
        opts = Edict()
        opts.CH = 32
        opts.use_rgb = False
        opts.circular_pad = False
        self.opts = argparse(opts, varargin)
        self.feature_layers = FeatureLayers(self.opts.CH, self.opts.use_rgb)
        self.spherical_sweep = SphericalSweep(
            self.opts.CH, self.opts.circular_pad)
        self.cost_computes = CostCompute(
            self.opts.CH, self.opts.circular_pad)

    def forward(self, imgs, grids, invdepth_indices,
            off_indices=[], out_cost=True):
        feats = [self.feature_layers(x) for x in imgs]
        input_idx = list(range(len(imgs)))
        if self.training:
            input_idx = random_index(len(imgs))
        batch_size = feats[0].shape[0]
        if batch_size > 1:
            spherical_feats = []
            for b in range(batch_size):
                bat_spherical_feats = [\
                    self.spherical_sweep(
                        feats[i][b,...].unsqueeze(0), grids[b][i]) \
                    for i in input_idx]
                spherical_feats.append(
                    torch.cat(bat_spherical_feats,2).permute([0, 2, 1, 3, 4]))
            spherical_feats = torch.cat(spherical_feats, 0)
        else:
            if len(grids) == 1:
                grids = grids[0]
            spherical_feats = torch.cat(
                [self.spherical_sweep(feats[i], grids[i]) \
                    for i in input_idx],2).permute([0, 2, 1, 3, 4])
        costs = self.cost_computes(spherical_feats)
        # softargmax
        costs = torch.squeeze(costs, 1)
        # costs = F.interpolate(costs.squeeze(1), scale_factor=2,
        #         mode='bilinear', align_corners=True)
        prob = F.softmax(costs, 1)
        disp = torch.mul(prob, invdepth_indices)
        disp = torch.sum(disp, 1)
        if out_cost:
            return disp, prob, costs
        else:
            return disp


