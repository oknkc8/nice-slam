# network.py
# architecture submitted to PAMI
# Author: Changhee Won (chwon@hanyang.ac.kr)
#
#
import torch
import torch.nn.functional as F
from src.omnimvs_src.module.basic import *
from src.omnimvs_src.module.basic_sparse import *

import torchsparse
import torchsparse.nn as spnn
from torchsparse.tensor import PointTensor
from torchsparse.utils import *

from src.omni_utils.torchsparse_utils import *

import pdb

# torch.autograd.set_detect_anomaly(True)

class FeatureLayers(torch.nn.Module):
    def __init__(self, CH=32, use_rgb=False):
        super(FeatureLayers, self).__init__()
        layers = []
        in_channel = 3 if use_rgb else 1
        layers.append(Conv2D(in_channel,CH,5,2,2)) # conv[1]
        layers += [Conv2D(CH,CH,3,1,1) for _ in range(10)] # conv[2-11]
        for d in range(2,5): # conv[12-17]
            layers += [Conv2D(CH,CH,3,1,d,dilation=d) for _ in range(2)]
        layers.append(Conv2D(CH,CH,3,1,1,bn=False,relu=False)) # conv[18]
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, im):
        x = self.layers[0](im)
        for i in range(1,17,2):
            x_ = self.layers[i](x)
            x = self.layers[i+1](x_,residual=x)
        x = self.layers[17](x)
        return x

class SphericalSweep(torch.nn.Module):
    def __init__(self, CH=32, circular_pad=False):
        super(SphericalSweep, self).__init__()
        _Conv2D = HorizontalCircularConv2D if circular_pad else Conv2D
        # self.transfer_conv = \
        #     HorizontalCircularConv2D(CH,CH,3,2,1,bn=False,relu=False)
        self.transfer_conv = \
            _Conv2D(CH,CH,3,2,1,bn=False,relu=False)

    def forward(self, feature, grid):
        num_invdepth = grid.shape[1]
        sweep = [F.grid_sample(feature, grid[:,d,...],
            align_corners=True).unsqueeze(1) for d in range(0, num_invdepth)]
        sweep = torch.cat(sweep, 1) # -> B x N/2 x CH x H x W
        B, N, C, H, W = sweep.shape
        sweep = sweep.view((B * N, C, H, W))
        spherical_feature = self.transfer_conv(sweep).view(
            (B, N, C, H // 2, W // 2))
        return spherical_feature



class CostCompute(torch.nn.Module):
    def __init__(self, CH=32, circular_pad=False):
        super(CostCompute, self).__init__()
        _Conv3D = HorizontalCircularConv3D if circular_pad else Conv3D
        CH *= 2
        self.fusion = _Conv3D(2*CH,CH,3,1,1)
        convs = []
        convs += [_Conv3D(CH,CH,3,1,1),
                        _Conv3D(CH,CH,3,1,1),
                        _Conv3D(CH,CH,3,1,1)]
        convs += [_Conv3D(CH,2*CH,3,2,1),
                        _Conv3D(2*CH,2*CH,3,1,1),
                        _Conv3D(2*CH,2*CH,3,1,1)]
        convs += [_Conv3D(2*CH,2*CH,3,2,1),
                        _Conv3D(2*CH,2*CH,3,1,1),
                        _Conv3D(2*CH,2*CH,3,1,1)]
        convs += [_Conv3D(2*CH,2*CH,3,2,1),
                        _Conv3D(2*CH,2*CH,3,1,1),
                        _Conv3D(2*CH,2*CH,3,1,1)]
        convs += [_Conv3D(2*CH,4*CH,3,2,1),
                        _Conv3D(4*CH,4*CH,3,1,1),
                        _Conv3D(4*CH,4*CH,3,1,1)]
        self.convs = torch.nn.ModuleList(convs)
        self.deconv1 = DeConv3D(4*CH,2*CH,3,2,1,out_pad=1)
        self.deconv2 = DeConv3D(2*CH,2*CH,3,2,1,out_pad=1)
        self.deconv3 = DeConv3D(2*CH,2*CH,3,2,1,out_pad=1)
        self.deconv4 = DeConv3D(2*CH,CH,3,2,1,out_pad=1)
        self.deconv5 = DeConv3D(CH,1,3,2,1,out_pad=1,bn=False,relu=False)

    def forward(self, feats):
        c = self.fusion(feats)
        raw_feat = c
        c = self.convs[0](c)
        c1 = self.convs[1](c)
        c1 = self.convs[2](c1)
        c = self.convs[3](c)
        c2 = self.convs[4](c)
        c2 = self.convs[5](c2)
        c = self.convs[6](c)
        c3 = self.convs[7](c)
        c3 = self.convs[8](c3)
        c = self.convs[9](c)
        c4 = self.convs[10](c)
        c4 = self.convs[11](c4)
        c = self.convs[12](c)
        c5 = self.convs[13](c)
        c5 = self.convs[14](c5)
        c = self.deconv1(c5, residual=c4)
        c = self.deconv2(c, residual=c3)
        c = self.deconv3(c, residual=c2)
        c = self.deconv4(c, residual=c1)
        geo_feat = c
        costs = self.deconv5(c)
        return costs, raw_feat, geo_feat
        # return costs, raw_feat
        
        
class CostFusion(torch.nn.Module):
    def __init__(self, ch_in=64, ch_out=32):
        super(CostFusion, self).__init__()
        
        CH = ch_out if ch_in != 1 else 8
            
        self.enc_conv0 = ConvBnReLU3D(ch_in, CH)
        
        self.enc_conv1 = ConvBnReLU3D(CH, 2*CH, stride=2)
        self.enc_conv2 = ConvBnReLU3D(2*CH, 2*CH)
        
        self.enc_conv3 = ConvBnReLU3D(2*CH, 4*CH, stride=2)
        self.enc_conv4 = ConvBnReLU3D(4*CH, 4*CH)
        
        self.dec_conv0 = ConvBnReLU3D(4*CH, 2*CH)
        self.dec_conv1 = ConvBnReLU3D(2*CH, 2*CH)
        
        self.dec_conv2 = ConvBnReLU3D(2*CH, CH)
        self.dec_conv3 = ConvBnReLU3D(CH, CH)
        
        self.dec_conv4 = ConvBnReLU3D(CH, ch_out, bn=False, relu=False, bias=False)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):        
        conv0 = self.enc_conv0(x)
        conv2 = self.enc_conv2(self.enc_conv1(conv0))
        x = self.enc_conv4(self.enc_conv3(conv2))
        
        conv0_size = conv0.shape[-3:]
        conv2_size = conv2.shape[-3:]
        
        x = F.upsample(x, conv2_size, mode='trilinear', align_corners=True)
        x = conv2 + self.dec_conv1(self.dec_conv0(x))
        x = F.upsample(x, conv0_size, mode='trilinear', align_corners=True)
        x = conv0 + self.dec_conv3(self.dec_conv2(x))
        x = self.dec_conv4(x)
        
        del conv2
        del conv0
        
        return x


class CostFusion_tmp(torch.nn.Module):
    def __init__(self, ch_in=64, ch_out=32):
        super(CostFusion_tmp, self).__init__()
        
        CH = ch_out if ch_in != 1 else 8

        self.conv1 = ConvBnReLU3D(ch_in, CH)
        self.conv2 = ConvBnReLU3D(CH, CH)

        self.conv3 = ConvBnReLU3D(CH, 2*CH, stride=2)
        self.conv4 = ConvBnReLU3D(2*CH, 2*CH)

        self.conv5 = ConvBnReLU3D(2*CH, 4*CH, stride=2)
        self.conv6 = ConvBnReLU3D(4*CH, 4*CH)

        self.deconv1 = DeConvBnReLU3D(4*CH, 2*CH, stride=2)
        self.deconv2 = DeConvBnReLU3D(2*CH, 2*CH)

        self.deconv3 = DeConvBnReLU3D(2*CH, CH, stride=2)
        self.deconv4 = DeConvBnReLU3D(CH, CH)

        self.deconv5 = DeConvBnReLU3D(CH, ch_out, bn=False, relu=False, bias=False)
        
        self.residual1 = ConvBnReLU3D(CH, CH)
        self.residual2 = ConvBnReLU3D(CH, CH)
        self.residual3 = ConvBnReLU3D(2*CH, 2*CH)
        self.residual4 = ConvBnReLU3D(2*CH, 2*CH)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        grid_shape = x.shape[-3:]

        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        res1 = self.residual1(c1)
        res2 = self.residual2(c2)

        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        res3 = self.residual3(c3)
        res4 = self.residual3(c4)

        c5 = self.conv5(c4)
        c6 = self.conv6(c5)

        c = self.deconv1(c6, residual=res4)
        c = self.deconv2(c, residual=res3)
        c = self.deconv3(c, residual=res2)
        c = self.deconv4(c, residual=res1)

        c = self.deconv5(c)
        c = F.interpolate(c, grid_shape, mode='trilinear', align_corners=True)

        return c


class CostFusion_Residual(torch.nn.Module):
    def __init__(self, ch_in=64, ch_out=32):
        super(CostFusion_Residual, self).__init__()
        
        CH = ch_out if ch_in != 1 else 8
            
        self.enc_conv0 = ConvBnReLU3D(ch_in, CH)
        
        self.enc_conv1 = ConvBnReLU3D(CH, 2*CH, stride=2)
        self.enc_conv2 = ResidualBlock3D(2*CH, 2*CH)
        
        self.enc_conv3 = ConvBnReLU3D(2*CH, 4*CH, stride=2)
        self.enc_conv4 = ResidualBlock3D(4*CH, 4*CH)
        
        self.dec_conv0 = ConvBnReLU3D(4*CH, 2*CH)
        self.dec_conv1 = ResidualBlock3D(2*CH, 2*CH)
        
        self.dec_conv2 = ConvBnReLU3D(2*CH, CH)
        self.dec_conv3 = ResidualBlock3D(CH, CH)
        
        self.dec_conv4 = ConvBnReLU3D(CH, ch_out, bn=False, relu=False, bias=False)

    def forward(self, x):        
        conv0 = self.enc_conv0(x)
        conv2 = self.enc_conv2(self.enc_conv1(conv0))
        x = self.enc_conv4(self.enc_conv3(conv2))
        
        conv0_size = conv0.shape[-3:]
        conv2_size = conv2.shape[-3:]
        
        x = F.upsample(x, conv2_size, mode='trilinear', align_corners=True)
        x = conv2 + self.dec_conv1(self.dec_conv0(x))
        x = F.upsample(x, conv0_size, mode='trilinear', align_corners=True)
        x = conv0 + self.dec_conv3(self.dec_conv2(x))
        x = self.dec_conv4(x)
        
        del conv2
        del conv0
        
        return x
    
    
class GRUFusion(torch.nn.Module):
    def __init__(self, ch_in=64, ch_out=64):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.gru = torch.nn.GRUCell(ch_in, ch_out)
        
    def forward(self, x, h=None):
        grid_dim = [*x.shape]
        grid_dim[1] = self.ch_out
        x = x.reshape(self.ch_in, -1).T
        if h is None:
            ret = self.gru(x)
            ret = ret.reshape(*grid_dim)
            return ret
        else:
            h = h.reshape(self.ch_out, -1).T
            ret = self.gru(x, h)
            ret = ret.reshape(*grid_dim)
            return ret
        
        
class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128, ks=3, pres=1, vres=1):
        super(ConvGRU, self).__init__()
        self.convz = SConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, ks)
        self.convr = SConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, ks)
        self.convq = SConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, ks)

    def forward(self, h, x):
        '''

        :param h: PointTensor
        :param x: PointTensor
        :return: h.F: Tensor (N, C)
        '''
        # hx = PointTensor(torch.cat([h.F, x.F], dim=1), h.C)
        hx = torchsparse.cat([h, x])

        z = torch.sigmoid(self.convz(hx).F)
        r = torch.sigmoid(self.convr(hx).F)
        x.F = torch.cat([r * h.F, x.F], dim=1)
        q = torch.tanh(self.convq(x).F)

        h.F = (1 - z) * h.F + z * q
        return h

"""
class CostFusion_Sparse(nn.Module):
    def __init__(self, ch_in=64, ch_out=32, pres=1, vres=1):
        super(CostFusion, self).__init__()
        
        self.pres = pres
        self.vres = vres
        CH = 32 if ch_in != 1 else 8
        
        self.conv0 = BasicConvolutionBlock(ch_in, CH)
        
        self.conv1 = BasicConvolutionBlock(CH, 2*CH, ks=3, stride=2)
        self.conv2 = BasicConvolutionBlock(2*CH, 2*CH, relu=False)
        
        self.conv3 = BasicConvolutionBlock(2*CH, 4*CH, ks=3, stride=2)
        self.conv4 = BasicConvolutionBlock(4*CH, 4*CH, relu=False)
        
        self.deconv0 = BasicDeconvolutionBlock(4*CH, 2*CH, ks=3, stride=2)
        self.deconv1 = BasicDeconvolutionBlock(2*CH, 2*CH)
        
        self.deconv2 = BasicDeconvolutionBlock(2*CH, CH, ks=3, stride=2)
        self.deconv3 = BasicDeconvolutionBlock(CH, CH)
        
        self.deconv4 = BasicDeconvolutionBlock(CH, ch_out, bn=False, relu=False)
        
        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2*CH, 2*CH),
                nn.BatchNorm1d(2*CH),
                nn.LeakyReLU(True),
            ),
            nn.Sequential(
                nn.Linear(CH, CH),
                nn.BatchNorm1d(CH),
                nn.LeakyReLU(True),
            )
        ])
        
        # self.weight_initialization()
        
    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: PointTensor
        x0 = self.conv0(x)
        
        x1 = self.conv1(x0)
        # x1.F = x1.F + x.F
        x2 = self.conv2(x1)
        
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        
        y0 = self.deconv0(x4)
        y1 = self.deconv1(y0)
        # y1.F = y1.F + self.point_transforms[0](x2.F)
        y1.F = y1.F + x2.F
        
        y2 = self.deconv2(y1)
        y3 = self.deconv3(y2)
        # y3.F = y3.F + self.point_transforms[1](x0.F)
        y3.F = y3.F + x0.F
        
        y4 = self.deconv4(y3)
        
        return y4
"""
        

class FeatureNet_Panorama(torch.nn.Module):
    def __init__(self):
        super(FeatureNet_Panorama, self).__init__()

        self.conv1 = ConvBnReLU2D(3, 16)
        self.conv2 = ConvBnReLU2D(16, 16)
        
        self.conv3 = ConvBnReLU2D(16, 32, stride=2)
        self.conv4 = ConvBnReLU2D(32, 32)
        
        self.conv5 = ConvBnReLU2D(32, 64, stride=2)
        self.conv6 = ConvBnReLU2D(64, 64)
        
        self.conv7 = DeConvBnReLU2D(64, 64, stride=2)
        self.conv8 = DeConvBnReLU2D(64, 64, stride=2)
        self.conv9 = DeConvBnReLU2D(64, 64)
        
        self.residual1 = ConvBnReLU2D(16, 64)
        self.residual2 = ConvBnReLU2D(32, 64)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)

        x = self.conv5(c4)
        x = self.conv6(x)
        
        res2 = self.residual1(c2)
        res4 = self.residual2(c4)
        
        x = self.conv7(x, res4)
        x = self.conv8(x, res2)
        x = self.conv9(x)
        
        del res2
        del res4
        del c1
        del c2
        del c3
        del c4
        
        return x


class OmniMVSNet(torch.nn.Module):

    def __init__(self, varargin=None):
        super(OmniMVSNet, self).__init__()
        opts = Edict()
        opts.CH = 32
        opts.num_invdepth = 192
        opts.use_rgb = True
        opts.upsample = False
        opts.circular_pad = False
        self.opts = argparse(opts, varargin)
        self.feature_layers = FeatureLayers(self.opts.CH, self.opts.use_rgb)
        self.spherical_sweep = SphericalSweep(
            self.opts.CH, self.opts.circular_pad)
        self.cost_computes = CostCompute(
            self.opts.CH, self.opts.circular_pad)

    def forward(self, imgs, grids, invdepth_indices,
            off_indices=[], out_cost=True, integrate=False):
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
        costs, raw_feat, geo_feat = self.cost_computes(spherical_feats)
        # costs, raw_feat = self.cost_computes(spherical_feats)
        
        # import pdb
        # pdb.set_trace()
        if self.opts.upsample:
            costs = F.interpolate(costs.squeeze(1), scale_factor=2,
                    mode='bilinear', align_corners=True)
        else:
            costs = torch.squeeze(costs, 1)
        # costs = torch.squeeze(costs, 1)
        
        if integrate:
            return costs, raw_feat, geo_feat

        prob = F.softmax(costs, 1)
        disp = torch.mul(prob, invdepth_indices)
        disp = torch.sum(disp, 1)
        if out_cost:
            return disp, prob, costs, raw_feat, geo_feat
            # return disp, prob, costs, raw_feat
        else:
            return disp

