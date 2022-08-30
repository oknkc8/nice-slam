"""
Copied from:
https://github.com/zju3dv/NeuralRecon/blob/master/models/modules.py
"""
import torch
import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn
from torchsparse.tensor import PointTensor
from torchsparse.utils import *

from src.omni_utils.torchsparse_utils import *

import pdb

class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, bn=True, relu=True):
        super().__init__()
        self.bn = None
        self.relu = None
        self.net = spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride)
        if bn:
            self.bn = spnn.BatchNorm(outc)
        if relu:
            self.relu = spnn.LeakyReLU(True)

    def forward(self, x):
        out = self.net(x)
        if self.bn is not None:
            out = self.bn(out)
        if self.relu is not None:
            out = self.relu(out)
        return out


class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, bn=True, relu=True):
        super().__init__()
        self.bn = None
        self.relu = None
        self.net = spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        stride=stride,
                        transposed=True)
        if bn:
            self.bn = spnn.BatchNorm(outc)
        if relu:
            self.relu = spnn.LeakyReLU(True)

    def forward(self, x):
        out = self.net(x)
        if self.bn is not None:
            out = self.bn(out)
        if self.relu is not None:
            out = self.relu(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride), spnn.BatchNorm(outc),
            spnn.LeakyReLU(True),
            spnn.Conv3d(outc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=1), spnn.BatchNorm(outc))

        self.downsample = nn.Sequential() if (inc == outc and stride == 1) else \
            nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1, stride=stride),
                spnn.BatchNorm(outc)
            )

        self.relu = spnn.LeakyReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class SPVCNN(nn.Module):        
    def __init__(self, ch_in=64, ch_out=32, dropout=True):
        super().__init__()

        self.dropout = dropout

        CH = ch_out if ch_in != 1 else 8

        self.stem = nn.Sequential(
            spnn.Conv3d(ch_in, CH, kernel_size=3, stride=1),
            spnn.BatchNorm(CH), spnn.ReLU(True)
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(CH, CH, ks=2, stride=2, dilation=1),
            ResidualBlock(CH, 2*CH, ks=3, stride=1, dilation=1),
            ResidualBlock(2*CH, 2*CH, ks=3, stride=1, dilation=1),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(2*CH, 2*CH, ks=2, stride=2, dilation=1),
            ResidualBlock(2*CH, 4*CH, ks=3, stride=1, dilation=1),
            ResidualBlock(4*CH, 4*CH, ks=3, stride=1, dilation=1),
        )

        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock(4*CH, 2*CH, ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(2*CH + 2*CH, 2*CH, ks=3, stride=1,
                              dilation=1),
                ResidualBlock(2*CH, 2*CH, ks=3, stride=1, dilation=1),
            )
        ])

        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock(2*CH, CH, ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(CH + CH, CH, ks=3, stride=1,
                              dilation=1),
                ResidualBlock(CH, CH, ks=3, stride=1, dilation=1),
            )
        ])

        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(CH, 4*CH),
                nn.BatchNorm1d(4*CH),
                nn.LeakyReLU(True),
            ),
            nn.Sequential(
                nn.Linear(4*CH, CH),
                nn.BatchNorm1d(CH),
                nn.LeakyReLU(True),
            )
        ])

        self.weight_initialization()

        if self.dropout:
            self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        # z: SparseTensor
        x0 = self.stem(z)

        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        # x2.F = x2.F + self.point_transforms[0](x0.F)

        if self.dropout:
            x2.F = self.dropout(x2.F)
        y3 = self.up1[0](x2)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.up1[1](y3)

        y4 = self.up2[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.up2[1](y4)
        # y4.F = y4.F + self.point_transforms[1](x2.F)

        return y4


class SConv3d(nn.Module):
    def __init__(self, inc, outc, pres, vres, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = spnn.Conv3d(inc,
                               outc,
                               kernel_size=ks,
                               dilation=dilation,
                               stride=stride)
        self.point_transforms = nn.Sequential(
            nn.Linear(inc, outc),
            # nn.BatchNorm1d(outc),
            # nn.ReLU(True),
        )
        self.pres = pres
        self.vres = vres

        self.weight_initialization()
        
    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z):
        # return: SparseTensor
        # x = initial_voxelize(z, self.pres, self.vres)
        x = self.net(z)
        x.F = x.F + self.point_transforms(z.F)
        return x
        # out = voxel_to_point(x, z, nearest=False)
        # out.F = out.F + self.point_transforms(z.F)
        # return out

    
class FilteringConv(nn.Module):
    def __init__(self, inc=64, outc=64):
        super(FilteringConv, self).__init__()
        self.conv1 = BasicConvolutionBlock(inc=inc, outc=inc//2, ks=3)
        self.conv2 = BasicConvolutionBlock(inc=inc//2, outc=outc, ks=3, bn=False, relu=False)
        
    def forward(self, x):
        return self.conv2(self.conv1(x))