# -*- coding: utf-8 -*- #
# Author: Henry
# Date:   2020/7/16

import torch
import torch.nn as nn
import copy
import cv2

from utils.imresize import my_torch2uint8
from utils.functions import norm


def weights_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)
    # classname = m.__class__.__name__
    # if classname.find('Conv') != -1:
    #     nn.init.normal_(m.weight.data, 0.0, 0.02)
    # elif classname.find('BatchNorm') != -1:
    #     nn.init.normal_(m.weight.data, 1.0, 0.02)
    #     nn.init.constant_(m.bias.data, 0)


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
    # x_np = my_torch2uint8(x_up)
    #
    # for i in range(x_np.shape[0]):
    #     for j in range(x_np.shape[-1]):
    #         temp = x_np[i, :, :, j]
    #         blur = cv2.bilateralFilter(temp, 5, 20, 10)
    #         x_np[i, :, :, j] = blur
    #
    # out = torch.tensor(x_np, dtype=torch.float32, requires_grad=True, device='cuda')
    # out = out.permute(0, 3, 1, 2) / 255
    # out = norm(out)
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
                 stride=1, dilation=1, bias=False, attention=False):
        super(PyConv2dAttention, self).__init__()

        if pyconv_groups is None:
            pyconv_groups = [1, 4, 8]
        if pyconv_kernels is None:
            pyconv_kernels = [3, 5, 7]

        d = max(int(in_channels / r), 32)
        self.attention = attention

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

        # feas_u = torch.cat(feas, dim=1)  # N * C * H * W
        # if not self.attention:
        #     return feas_u
        #
        # else:
        #     # for i, fea in enumerate(feas):
        #     #     if i == 0:
        #     #         feas_new = fea.unsqueeze(dim=1)
        #     #     else:
        #     #         feas_new = torch.cat([feas_new, fea.unsqueeze(dim=1)], dim=1)
        #     feas_new = feas_u.unsqueeze(dim=1)
        #
        #     fea_s = feas_u.mean(-1).mean(-1)
        #     fea_z = self.fc(fea_s)
        #     for i, fc in enumerate(self.fcs):
        #         vector = fc(fea_z).unsqueeze_(dim=1)
        #         if i == 0:
        #             attention_vectors = vector
        #         else:
        #             attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        #
        #     attention_vectors = self.softmax(attention_vectors)
        #     attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        #
        #     fea_v = (feas_new * attention_vectors).sum(dim=1)
        #     return fea_v

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
                # self.conv.add_module('Conv%d' % i, nn.Conv2d(nf + gc * i, nf, 3, padding=1))
                self.conv.add_module('PyConv%d' % i, PyConv2dAttention(nf + gc * i, nf, attention=True))
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
        # y = self.pyconv_att(out[-1])
        # y = self.al(out[-1])
        return x + y


# class PyConvBlock(nn.Module):
#     expansion = 4
#
#     def __init__(self, opt, inplanes, planes, stride=1, downsample=None, norm_layer=None, pyconv_groups=1, pyconv_kernels=1):
#         super(PyConvBlock, self).__init__()
#         if norm_layer is None and opt.batch_norm:
#             norm_layer = nn.BatchNorm2d
#         # Both self.conv2 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv1x1(inplanes, planes)
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
#         self.bn1 = norm_layer(planes)
#         self.conv2 = get_pyconv(planes, planes, pyconv_kernels=pyconv_kernels, stride=stride,
#                                 pyconv_groups=pyconv_groups)
#         self.bn2 = norm_layer(planes)
#         self.conv3 = conv1x1(planes, planes * self.expansion)
#         self.bn3 = norm_layer(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#
#         return out


class PIPNet(nn.Module):
    def __init__(self, opt):
        super(PIPNet, self).__init__()
        self.activation = get_activation(opt)
        if opt.batch_norm:
            self.bn = nn.BatchNorm2d(opt.nf)

        # self.head = nn.Conv2d(opt.nc_im, opt.nf, kernel_size=3, padding=1)
        self.head = ConvBlock(opt.nc_im, opt.nf, opt.ker_size, 1, opt, generator=True)

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

        # self.upsample = UpBlock(opt.nf)

    def forward(self, noise, in_data, real_shapes, noise_amp):
        # x = self.head(noise[0])
        x_prev_out = self.head(in_data)
        # x_prev_out = self.body[0](x)
        # for idx, block in enumerate(self.body[1:], 1):
        for idx, block in enumerate(self.body):
            # x_prev_out = upsample(x_prev_out, size=[real_shapes[idx][2], real_shapes[idx][3]])
            x_prev_out = upsample(x_prev_out, size=[real_shapes[idx + 1][2], real_shapes[idx + 1][3]])
            # x_prev_out = self.upsample(x_prev_out, real_shapes[idx])
            # x_prev_out = block(x_prev_out + noise[idx] * noise_amp[idx])
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
        batch_size = x.size(0)
        # return torch.sigmoid(self.net(x).view(batch_size))  # 将一个batch中每个的输出并列
        return self.net(x).view(-1)

# class Discriminator(nn.Module):
#     def __init__(self, ngpu):
#         super(Discriminator, self).__init__()
#         nc = 3
#         ndf = 64
#         self.ngpu = ngpu
#         self.main = nn.Sequential(
#             # input is (nc) x 64 x 64
#             nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf) x 32 x 32
#             nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*2) x 16 x 16
#             nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*4) x 8 x 8
#             nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*8) x 4 x 4
#             nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, input):
#         return self.main(input).view(-1)
