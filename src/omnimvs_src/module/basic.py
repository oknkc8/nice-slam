
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _triple
from torch.autograd import Variable
from src.omni_utils.common import *

import pdb

class Conv2D(torch.nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride=1, pad=1,
                 dilation=1, bn=True, relu=True):
        super(Conv2D, self).__init__()
        self.opts = Edict()
        self.opts.bn = bn
        self.opts.relu = relu
        self.conv = torch.nn.Conv2d(ch_in, ch_out, kernel_size,
                                    stride, padding=pad, dilation=dilation)
        if self.opts.bn:
            self.bn = torch.nn.BatchNorm2d(ch_out)

    def forward(self, x, residual=None):
        x = self.conv(x)
        if self.opts.bn:
            x = self.bn(x)
        if not residual is None:
            x = x + residual
        if self.opts.relu:
            x = F.relu(x)
        return x


class Conv3D(torch.nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride=1, pad=1,
                 dilation=1, bn=True, relu=True):
        super(Conv3D, self).__init__()
        self.opts = Edict()
        self.opts.bn = bn
        self.opts.relu = relu
        self.conv = torch.nn.Conv3d(ch_in, ch_out, kernel_size,
                                    stride, padding=pad, dilation=dilation)
        if self.opts.bn:
            self.bn = torch.nn.BatchNorm3d(ch_out)

    def forward(self, x, residual=None):
        x = self.conv(x)
        if self.opts.bn:
            x = self.bn(x)
        if not residual is None:
            x = x + residual
        if self.opts.relu:
            x = F.relu(x)
        return x

class DeConv3D(torch.nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride=1, pad=1,
                 dilation=1, out_pad=0, bn=True, relu=True):
        super(DeConv3D, self).__init__()
        self.opts = Edict()
        self.opts.bn = bn
        self.opts.relu = relu
        self.conv = torch.nn.ConvTranspose3d(ch_in, ch_out, kernel_size,
                                             stride, padding=pad, dilation=dilation, output_padding=out_pad)
        if self.opts.bn:
            self.bn = torch.nn.BatchNorm3d(ch_out)

    def forward(self, x, residual=None):
        x = self.conv(x)
        if self.opts.bn:
            x = self.bn(x)
        if not residual is None:
            x = x + residual
        if self.opts.relu:
            x = F.relu(x)
        return x


class ConvBnReLU3D(torch.nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, pad=1,
                 bn=True, relu=True, bias=True):
        super(ConvBnReLU3D, self).__init__()
        self.conv = torch.nn.Conv3d(ch_in, ch_out, kernel_size,
                                    stride, padding=pad, bias=bias)
        self.relu = relu
        self.bn = None
        if bn:
            # self.bn = torch.nn.BatchNorm3d(ch_out)
            self.bn = torch.nn.GroupNorm(4, ch_out)

    def forward(self, x, residual=None):
        x = self.conv(x)
        if self.bn is not None:
            x= self.bn(x)
        if residual is not None:
            if x.shape[-3:] != residual.shape[-3:]:
                residual = F.interpolate(residual, x.shape[-3:], mode='trilinear', align_corners=True)
            x = x + residual
        if self.relu:
            return F.leaky_relu(x)
            # return F.relu(x)
        else:
            return x

class DeConvBnReLU3D(torch.nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, pad=1, out_pad=0,
                 bn=True, relu=True, bias=True):
        super(DeConvBnReLU3D, self).__init__()
        self.conv = torch.nn.ConvTranspose3d(ch_in, ch_out, kernel_size,
                                             stride, padding=pad, output_padding=out_pad, bias=bias)
        self.relu = relu
        self.bn = None
        if bn:
            # self.bn = torch.nn.BatchNorm3d(ch_out)
            self.bn = torch.nn.GroupNorm(4, ch_out)

    def forward(self, x, residual=None):
        x = self.conv(x)
        if self.bn is not None:
            x= self.bn(x)
        if residual is not None:
            if x.shape[-3:] != residual.shape[-3:]:
                residual = F.interpolate(residual, x.shape[-3:], mode='trilinear', align_corners=True)
            x = x + residual
        if self.relu:
            return F.leaky_relu(x)
            # return F.relu(x)
        else:
            return x

class ResidualBlock3D(torch.nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, pad=1):
        super(ResidualBlock3D, self).__init__()
        self.net = torch.nn.Sequential(
                        torch.nn.Conv3d(ch_in, ch_out, kernel_size,
                                        stride, padding=pad, bias=False),
                        torch.nn.BatchNorm3d(ch_out),
                        torch.nn.LeakyReLU(True),
                        torch.nn.Conv3d(ch_out, ch_out, kernel_size,
                                        stride, padding=pad, bias=False),
                        torch.nn.BatchNorm3d(ch_out))
        self.relu = torch.nn.LeakyReLU(True)
    def forward(self, x):
        out = self.relu(self.net(x) + x)
        return out


class HorizontalCircularConv2D(torch.nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride=1, pad=1,
                 dilation=1, bn=True, relu=True):
        super(HorizontalCircularConv2D, self).__init__()
        self.opts = Edict()
        self.opts.bn = bn
        self.opts.relu = relu
        self.circ_pad = [pad, pad, 0, 0]
        self.conv = torch.nn.Conv2d(ch_in, ch_out, kernel_size,
                                    stride, padding=[pad,0],
                                    dilation=dilation)
        if self.opts.bn:
            self.bn = torch.nn.BatchNorm2d(ch_out)

    def forward(self, x, residual=None):
        x = F.pad(x, self.circ_pad, mode='circular')
        x = self.conv(x)
        if self.opts.bn:
            x = self.bn(x)
        if not residual is None:
            x = x + residual
        if self.opts.relu:
            x = F.relu(x)
        return x

class HorizontalCircularConv3D(torch.nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride=1, pad=1,
                 dilation=1, bn=True, relu=True):
        super(HorizontalCircularConv3D, self).__init__()
        self.opts = Edict()
        self.opts.bn = bn
        self.opts.relu = relu
        self.circ_pad = [pad, pad, 0, 0, 0, 0]
        self.conv = torch.nn.Conv3d(ch_in, ch_out, kernel_size,
                                    stride, padding=[pad,pad,0],
                                    dilation=dilation)
        if self.opts.bn:
            self.bn = torch.nn.BatchNorm3d(ch_out)

    def forward(self, x, residual=None):
        x = F.pad(x, self.circ_pad, mode='circular')
        x = self.conv(x)
        if self.opts.bn:
            x = self.bn(x)
        if not residual is None:
            x = x + residual
        if self.opts.relu:
            x = F.relu(x)
        return x

class UpsampleConv3D(torch.nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride=1, pad=1,
                 dilation=1, bn=True, relu=True):
        super(UpsampleConv3D, self).__init__()
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear',
                                          align_corners=True)
        self.conv = Conv3D(ch_in, ch_out, kernel_size, stride, pad, dilation,
                           bn, relu)

    def forward(self, x, residual=None):
        x = self.upsample(x)
        x = self.conv(x, residual)
        return x


class _NestedConv2d(torch.nn.Conv2d):
    def __init__(self, ch_ins, ch_outs, kernel_size, stride=1,
                 pad=1, dilation=1, padding_mode='zeros'):
        super(_NestedConv2d, self).__init__(
            np.sum(ch_ins), np.sum(ch_outs), kernel_size, stride, pad,
            dilation, padding_mode=padding_mode)
        self.ch_ins = np.copy(ch_ins)
        self.ch_outs = np.copy(ch_outs)

    def conv2d_forward(self, input, weight, bias):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2,
                                self.padding[1] // 2,
                                (self.padding[0] + 1) // 2,
                                self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, bias, self.stride,
                            torch.nn._pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, x, level=0):
        ch_in = np.sum(self.ch_ins[:level+1])
        ch_out = np.sum(self.ch_outs[:level+1])
        return self.conv2d_forward(x, self.weight[:ch_out, :ch_in, ...],
                                self.bias[:ch_out])

class SeparableConv3D(torch.nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride=1, pad=1,
                 dilation=1, bn=True, relu=True):
        super(SeparableConv3D, self).__init__()
        self.opts = Edict()
        self.opts.bn = bn
        self.opts.relu = relu

        k = _triple(kernel_size)
        s = _triple(pad)
        p = _triple(pad)
        d = _triple(dilation)

        self.conv_spatial = torch.nn.Conv3d(ch_in, ch_in, (1, k[1], k[2]),
                stride=(1, s[1], s[2]), padding=(0, p[1], p[2]),
                dilation=(d[0], d[1], 1))

        self.conv_depth = torch.nn.Conv3d(ch_in, ch_out, (k[0], 1, 1),
                stride=(s[0], 1, 1), padding=(p[0], 0, 0),
                dilation=(d[0], 1, 1))

        if self.opts.bn:
            self.bn = torch.nn.BatchNorm3d(ch_out)

    def forward(self, x, residual=None):
        x = self.conv_spatial(x)
        x = self.conv_depth(x)
        if self.opts.bn:
            x = self.bn(x)
        if not residual is None:
            x = x + residual
        if self.opts.relu:
            x = F.relu(x)
        return x

class HCircularSeparableConv3D(torch.nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride=1, pad=1,
                 dilation=1, bn=True, relu=True):
        super(HCircularSeparableConv3D, self).__init__()
        self.opts = Edict()
        self.opts.bn = bn
        self.opts.relu = relu
        self.circ_pad = [pad, pad, 0, 0, 0, 0]

        k = _triple(kernel_size)
        s = _triple(stride)
        p = _triple(pad)
        d = _triple(dilation)

        self.conv_spatial = torch.nn.Conv3d(ch_in, ch_in, (1, k[1], k[2]),
                stride=(1, s[1], s[2]), padding=(0, p[1], 0),
                dilation=(d[0], d[1], 1))

        self.conv_depth = torch.nn.Conv3d(ch_in, ch_out, (k[0], 1, 1),
                stride=(s[0], 1, 1), padding=(p[0], 0, 0),
                dilation=(d[0], 1, 1))

        if self.opts.bn:
            self.bn = torch.nn.BatchNorm3d(ch_out)

    def forward(self, x, residual=None):
        x = F.pad(x, self.circ_pad, mode='circular')
        x = self.conv_spatial(x)
        x = self.conv_depth(x)
        if self.opts.bn:
            x = self.bn(x)
        if not residual is None:
            x = x + residual
        if self.opts.relu:
            x = F.relu(x)
        return x

class _NestedConv3d(torch.nn.Conv3d):
    def __init__(self, ch_ins, ch_outs, kernel_size, stride=1,
                 pad=1, dilation=1, padding_mode='zeros'):
        super(_NestedConv3d, self).__init__(
            np.sum(ch_ins), np.sum(ch_outs), kernel_size, stride, pad,
            dilation, padding_mode=padding_mode)
        self.ch_ins = np.copy(ch_ins)
        self.ch_outs = np.copy(ch_outs)

    def conv3d_forward(self, input, weight, bias):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[2] + 1) // 2,
                                self.padding[2] // 2,
                                (self.padding[1] + 1) // 2,
                                self.padding[1] // 2,
                                (self.padding[0] + 1) // 2,
                                self.padding[0] // 2)
            return F.conv3d(F.pad(input, expanded_padding, mode='circular'),
                            weight, bias, self.stride,
                            torch.nn._triple(0), self.dilation, self.groups)
        return F.conv3d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, x, level=0):
        ch_in = np.sum(self.ch_ins[:level+1])
        ch_out = np.sum(self.ch_outs[:level+1])
        return self.conv3d_forward(x, self.weight[:ch_out, :ch_in, ...],
                                self.bias[:ch_out])

class _NestedConvTranspose3d(torch.nn.ConvTranspose3d):
    def __init__(self, ch_ins, ch_outs, kernel_size, stride=1,
                 pad=1, out_pad=0, dilation=1, padding_mode='zeros'):
        super(_NestedConvTranspose3d, self).__init__(
            np.sum(ch_ins), np.sum(ch_outs), kernel_size, stride, pad,
            out_pad, dilation=dilation, padding_mode=padding_mode)
        self.ch_ins = np.copy(ch_ins)
        self.ch_outs = np.copy(ch_outs)

    def forward(self, x, output_size=None, level=0):
        ch_in = np.sum(self.ch_ins[:level+1])
        ch_out = np.sum(self.ch_outs[:level+1])
        
        output_padding = self._output_padding(input, output_size, self.stride, 
            self.padding, self.kernel_size)

        return F.conv_transpose3d(x, self.weight[:ch_in, :ch_out, ...],
            self.bias[:ch_out], self.stride, self.padding, output_padding,
            self.groups, self.dilation)

class NestedConv2D(torch.nn.Module):
    def __init__(self, ch_ins, ch_outs, kernel_size, stride=1,
                 pad=1, dilation=1, bn=True, relu=True):
        super(NestedConv2D, self).__init__()
        self.opts = Edict()
        self.opts.bn = bn
        self.opts.relu = relu
        self.num_level = len(ch_ins)
        self.conv = _NestedConv2d(ch_ins, ch_outs, kernel_size, stride, pad,
            dilation)
        if self.opts.bn:
            self.bns = torch.nn.ModuleList(
                [torch.nn.BatchNorm2d(
                    np.sum(ch_outs[:l+1])) for l in range(self.num_level)])

    def forward(self, x, level=0, residual=None):
        x = self.conv(x, level=level)
        if self.opts.bn:
            x = self.bns[level](x)
        if not residual is None:
            x = x + residual
        if self.opts.relu:
            x = F.relu(x)
        return x

class NestedConv3D(torch.nn.Module):
    def __init__(self, ch_ins, ch_outs, kernel_size, stride=1,
                 pad=1, dilation=1, bn=True, relu=True):
        super(NestedConv3D, self).__init__()
        self.opts = Edict()
        self.opts.bn = bn
        self.opts.relu = relu
        self.num_level = len(ch_ins)
        self.conv = _NestedConv3d(ch_ins, ch_outs, kernel_size, stride, pad,
            dilation)
        if self.opts.bn:
            self.bns = torch.nn.ModuleList(
                [torch.nn.BatchNorm3d(
                    np.sum(ch_outs[:l+1])) for l in range(self.num_level)])

    def forward(self, x, level=0, residual=None):
        x = self.conv(x, level=level)
        if self.opts.bn:
            x = self.bns[level](x)
        if not residual is None:
            x = x + residual
        if self.opts.relu:
            x = F.relu(x)
        return x

class NestedDeConv3D(torch.nn.Module):
    def __init__(self, ch_ins, ch_outs, kernel_size, stride=1,
                 pad=1, dilation=1, out_pad=0, bn=True, relu=True):
        super(NestedDeConv3D, self).__init__()
        self.opts = Edict()
        self.opts.bn = bn
        self.opts.relu = relu
        self.num_level = len(ch_ins)
        self.conv = _NestedConvTranspose3d(ch_ins, ch_outs, kernel_size,stride,
            pad, out_pad, dilation)
        if self.opts.bn:
            self.bns = torch.nn.ModuleList(
                [torch.nn.BatchNorm3d(
                    np.sum(ch_outs[:l+1])) for l in range(self.num_level)])

    def forward(self, x, level=0, residual=None):
        x = self.conv(x, level=level)
        if self.opts.bn:
            x = self.bns[level](x)
        if not residual is None:
            x = x + residual
        if self.opts.relu:
            x = F.relu(x)
        return x