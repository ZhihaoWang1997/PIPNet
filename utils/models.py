import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import functools

from .models_new import PyConv2d


# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv2d') != -1:
#         m.weight.data.normal_(0.0, 0.02)
#     elif classname.find('Norm') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)


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
                   "selu": nn.SELU(inplace=True)
                   }
    return activations[opt.activation]


def upsample(x, size):
    x_up = torch.nn.functional.interpolate(x, size=size, mode='bicubic', align_corners=True)
    return x_up


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, opt, generator=False):
        super(ConvBlock, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, stride=1, padding=padd))
        if generator and opt.batch_norm:
            self.add_module('norm', nn.BatchNorm2d(out_channel))
        self.add_module(opt.activation, get_activation(opt))


class ResidualDenseBlock5C(nn.Module):
    def __init__(self, nf=64, gc=32, kernel=3, pad=1, bias=True):
        super(ResidualDenseBlock5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        # self.conv1 = nn.Conv2d(nf, gc, kernel, 1, pad, bias=bias)
        # self.conv2 = nn.Conv2d(nf + gc, gc, kernel, 1, pad, bias=bias)
        # self.conv3 = nn.Conv2d(nf + 2 * gc, gc, kernel, 1, pad, bias=bias)
        # self.conv4 = nn.Conv2d(nf + 3 * gc, gc, kernel, 1, pad, bias=bias)
        # self.conv5 = nn.Conv2d(nf + 4 * gc, nf, kernel, 1, pad, bias=bias)
        self.conv1 = PyConv2d(nf, gc)
        self.conv2 = PyConv2d(nf + gc, gc)
        self.conv3 = PyConv2d(nf + 2 * gc, gc)
        self.conv4 = PyConv2d(nf + 3 * gc, gc)
        self.conv5 = PyConv2d(nf + 4 * gc, nf)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block"""
    def __init__(self, opt):
        super(
            RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock5C(opt.nf, opt.gc, opt.ker_size, opt.padd_size)
        self.RDB2 = ResidualDenseBlock5C(opt.nf, opt.gc, opt.ker_size, opt.padd_size)
        self.RDB3 = ResidualDenseBlock5C(opt.nf, opt.gc, opt.ker_size, opt.padd_size)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class GrowingGenerator(nn.Module):
    def __init__(self, opt):
        super(GrowingGenerator, self).__init__()
        self.opt = opt

        RRDB_block_f = functools.partial(ResidualDenseBlock5C)
        self.RRDB_trunk = make_layer(RRDB_block_f, 1)

        self._pad = nn.ZeroPad2d(1)
        # self._pad_block = nn.ZeroPad2d(opt.num_layer - 1)

        # main stream
        self.head = nn.Conv2d(opt.nc_im, opt.nf, opt.ker_size, 1, padding=0)

        self.body = torch.nn.ModuleList([])
        _first_stage = nn.Sequential()
        for i in range(opt.G_num_blocks):
            _first_stage.add_module('block%d' % i, self.RRDB_trunk)
        self.body.append(_first_stage)

        self.tail = nn.Sequential(
            nn.Conv2d(opt.nf, opt.nc_im, kernel_size=opt.ker_size, padding=0),
            nn.Tanh())

    def init_next_stage(self):
        self.body.append(copy.deepcopy(self.body[-1]))

    def forward(self, noise, real_shapes, noise_amp):
        x = self.head(self._pad(noise[0]))  # SR不变

        # we do some upsampling for training models for unconditional generation to increase
        # the image diversity at the edges of generated images
        # x = upsample(x, size=[x.shape[2] + 2, x.shape[3] + 2])
        # x = self._pad_block(x)
        x_prev_out = self.body[0](x)  # SR不变

        for idx, block in enumerate(self.body[1:], 1):
            # x_prev_out_1 = upsample(x_prev_out, size=[real_shapes[idx][2], real_shapes[idx][3]])
            # x_prev_out_2 = upsample(x_prev_out, size=[real_shapes[idx][2] + self.opt.num_layer * 2,
            #                                           real_shapes[idx][3] + self.opt.num_layer * 2])
            # x_prev = block(x_prev_out_1 + noise[idx] * noise_amp[idx])
            # x_prev_out = x_prev + x_prev_out_1
            x_prev_out = upsample(x_prev_out, size=[real_shapes[idx][2], real_shapes[idx][3]])
            # x_prev_out = block(x_prev_out + noise[idx] * noise_amp[idx]) + x_prev_out
            x_prev_out = block(x_prev_out + noise[idx] * noise_amp[idx])

        out = self.tail(self._pad(x_prev_out))
        return out


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        self.opt = opt
        N = int(opt.nf)

        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, 0, opt)

        self.body = nn.Sequential()
        for i in range(opt.D_num_layer):
            block = ConvBlock(N, N, opt.ker_size, 0, opt)
            self.body.add_module('block%d' % i, block)

        self.tail = nn.Conv2d(N, 1, kernel_size=opt.ker_size, padding=0)
        # self.avgpool = nn.AvgPool2d(3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        head = self.head(x)
        body = self.body(head)
        out = self.tail(body)
        # out = self.avgpool(out)
        return torch.mean(self.sigmoid(out))
        # return out


def swish(x):
    return x * F.sigmoid(x)


# class Discriminator(nn.Module):
#     def __init__(self, opt):
#         super(Discriminator, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
#
#         self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#
#         self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)
#
#         self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
#         self.bn4 = nn.BatchNorm2d(128)
#
#         self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
#         self.bn5 = nn.BatchNorm2d(256)
#
#         self.conv6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
#         self.bn6 = nn.BatchNorm2d(256)
#
#         self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
#         self.bn7 = nn.BatchNorm2d(512)
#
#         self.conv8 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
#         self.bn8 = nn.BatchNorm2d(512)
#
#         # Replaced original paper FC layers with FCN
#         self.conv9 = nn.Conv2d(512, 1, 1, stride=1, padding=1)
#         self.avgpool = nn.AvgPool2d(3, 1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         x = swish(self.conv1(x))
#
#         x = swish(self.bn2(self.conv2(x)))
#         x = swish(self.bn3(self.conv3(x)))
#         x = swish(self.bn4(self.conv4(x)))
#         x = swish(self.bn5(self.conv5(x)))
#         x = swish(self.bn6(self.conv6(x)))
#         x = swish(self.bn7(self.conv7(x)))
#         x = swish(self.bn8(self.conv8(x)))
#
#         x = self.conv9(x)
#         x = self.avgpool()
#         # 变成列向量
#         return F.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)
