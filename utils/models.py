# -*- coding: utf-8 -*- #
# Author: Henry
# Date:   2020/7/16

import torch
import torch.nn as nn
import copy


def weights_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)


def get_activation(opt):
    activations = {"lrelu": nn.LeakyReLU(opt.lrelu_alpha, inplace=True),
                   "elu": nn.ELU(alpha=1.0, inplace=True),
                   "prelu": nn.PReLU(num_parameters=1, init=0.25),
                   "selu": nn.SELU(inplace=True),
                   "relu": nn.ReLU(inplace=True)
                   }
    return activations[opt.activation]


def upsample(x, size):
    x_up = torch.nn.functional.interpolate(x, size=size, mode='bicubic', align_corners=True)
    return x_up


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, opt, generator=False):
        super(ConvBlock, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, stride=1, padding=padd))
        if generator and opt.batch_norm:
            self.add_module('norm', nn.BatchNorm2d(out_channel))
        self.add_module(opt.activation, get_activation(opt))


class UpBlock(torch.nn.Module):
    def __init__(self, num_filters, kernel=3, stride=1, padding=1, bias=True):
        super(UpBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel, stride, padding, bias=bias)
        self.deconv1 = nn.ConvTranspose2d(num_filters, num_filters, kernel, stride, padding)
        self.deconv2 = nn.ConvTranspose2d(num_filters, num_filters, kernel, stride, padding)
        self.prelu = nn.PReLU()

    def forward(self, x, size):
        size[1] = x.shape[1]
        h0 = self.prelu(self.deconv1(x, output_size=size))
        l0 = self.prelu(self.conv1(h0))
        h1 = self.prelu(self.deconv2(l0 - x, output_size=size))
        return h1 + h0


class AttentionLayer(nn.Module):
    def __init__(self, channels, reduction=4):
        super(AttentionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.con_du = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.con_du(y)
        return x * y


class PyConv2dAttention(nn.Module):
    """PyConv2d with padding (general case). Applies a 2D PyConv over an input signal composed of several input planes.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels for output feature map
        pyconv_kernels (list): Spatial size of the kernel for each pyramid level
        pyconv_groups (list): Number of blocked connections from input channels to output channels for each pyramid level
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``False``
    """

    def __init__(self, in_channels, out_channels, r=2, pyconv_kernels=None, pyconv_groups=None,
                 stride=1, dilation=1, bias=False):
        super(PyConv2dAttention, self).__init__()

        if pyconv_groups is None:
            pyconv_groups = [1, 4, 8]
        if pyconv_kernels is None:
            pyconv_kernels = [3, 5, 7]

        d = max(int(in_channels / r), 32)

        split_channels = self._set_channels(out_channels, len(pyconv_kernels))
        self.pyconv_levels = nn.ModuleList([])
        for i in range(len(pyconv_kernels)):
            self.pyconv_levels.append(nn.Conv2d(in_channels, split_channels[i], kernel_size=pyconv_kernels[i],
                                                stride=stride, padding=pyconv_kernels[i] // 2, groups=pyconv_groups[i],
                                                dilation=dilation, bias=bias))

        self.fc = nn.Linear(out_channels, d)
        self.fcs = nn.ModuleList([])
        for _ in range(len(pyconv_kernels)):
            self.fcs.append(nn.Linear(d, out_channels))

        self.softmax = nn.Softmax(dim=1)

        self.att1 = AttentionLayer(split_channels[0])
        self.att2 = AttentionLayer(split_channels[1])
        self.att3 = AttentionLayer(split_channels[2])

    def forward(self, x):
        feas = [layer(x) for layer in self.pyconv_levels]
        out = [self.att1(feas[0]), self.att2(feas[1]), self.att3(feas[2])]
        out = torch.cat(out, dim=1)
        return out

    def _set_channels(self, out_channels, levels):
        if levels == 1:
            split_channels = [out_channels]
        elif levels == 2:
            split_channels = [out_channels // 2 for _ in range(2)]
        elif levels == 3:
            split_channels = [out_channels // 4, out_channels // 4, out_channels // 2]
        elif levels == 4:
            split_channels = [out_channels // 4 for _ in range(4)]
        else:
            raise NotImplementedError
        return split_channels


class RDBN(nn.Module):
    def __init__(self, nf=64, gc=32, n=3):
        super(RDBN, self).__init__()
        self.conv = nn.ModuleList()
        for i in range(n):
            if i == n - 1:

                self.conv.add_module('PyConv%d' % i, PyConv2dAttention(nf + gc * i, nf))
            else:
                self.conv.add_module('Conv%d' % i, nn.Conv2d(nf + gc * i, gc, 3, padding=1))

        # self.pyconv_att = PyConv2dAttention(nf, nf, attention=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.05, inplace=True)
        # self.al = AttentionLayer(nf)
        self.n = n

    def forward(self, x):
        out = [x]

        for i, layer in enumerate(self.conv):
            if i == 0:
                out.append(self.lrelu(layer(x)))
            elif i == self.n - 1:
                out.append(layer(torch.cat(out, dim=1)))
            else:
                out.append(self.lrelu(layer(torch.cat(out, dim=1))))

        y = out[-1]
        return x + y


class PIPNet(nn.Module):
    def __init__(self, opt):
        super(PIPNet, self).__init__()
        self.activation = get_activation(opt)
        if opt.batch_norm:
            self.bn = nn.BatchNorm2d(opt.nf)

        self.head = ConvBlock(opt.nc_im, opt.nf, opt.ker_size, 1, opt)

        self.body = nn.ModuleList()
        _first_stage = nn.Sequential()
        for i in range(opt.G_num_blocks):
            block = RDBN(opt.nf, opt.gc, opt.nl_in_block)
            _first_stage.add_module('block%d' % i, block)
        self.body.append(_first_stage)

        self.tail = nn.Sequential(
            nn.Conv2d(opt.nf, opt.nc_im, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, in_data, real_shapes, noise_amp):
        x_prev_out = self.head(in_data)
        for idx, block in enumerate(self.body):
            x_prev_out = upsample(x_prev_out, size=[real_shapes[idx + 1][2], real_shapes[idx + 1][3]])
            x_prev_out = block(x_prev_out + noise[idx + 1] * noise_amp[idx])

        out = self.tail(x_prev_out)
        return out

    def init_next_stage(self):
        self.body.append(copy.deepcopy(self.body[-1]))


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(opt.nc_im, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            # 把图像大小变成1*1
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1)
