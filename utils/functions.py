import torch
import torch.nn as nn
from torchvision.models import vgg19
import numpy as np
import math
from skimage import io as img
from skimage import color
import os
import random
import datetime
import dateutil.tz
import matplotlib.pyplot as plt

from utils.imresize import imresize


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def norm(x):
    out = (x - 0.5) * 2
    return out.clamp(-1, 1)


def get_minmax(x, opt):
    opt.hsi_max, opt.hsi_min = np.max(x), np.min(x)
    return None


def upsampling(im, sx, sy):
    m = nn.Upsample(size=[round(sx), round(sy)], mode='bilinear', align_corners=True)
    return m(im)


def move_to_gpu(t):
    if torch.cuda.is_available():
        t = t.to(torch.device('cuda'))
    return t


def move_to_cpu(t):
    t = t.to(torch.device('cpu'))
    return t


def convert_image_np(inp, opt):
    if inp.shape[1] == 3:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1, :, :, :])
        inp = inp.numpy().transpose((1, 2, 0))
    else:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1, -1, :, :])
        inp = inp.numpy().transpose((0, 1))

    inp = np.clip(inp, 0, 1)
    return inp


def save_image(name, image, opt, flag=True):
    out_image = convert_image_np(image, opt)
    if flag:
        plt.imsave(name, out_image, vmin=0, vmax=1)
    return out_image


def generate_noise(size, num_samp=1, device='cuda', type='gaussian', scale=1):
    if type == 'gaussian':
        noise = torch.randn(num_samp, size[0], round(size[1] / scale), round(size[2] / scale), device=device)
        noise = upsampling(noise, size[1], size[2])
    elif type == 'gaussian_mixture':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device) + 5
        noise2 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        noise = noise1 + noise2
    elif type == 'uniform':
        noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)
    else:
        raise NotImplementedError
    return noise


def sample_random_noise(depth, reals_shapes, opt):
    noise = []
    for d in range(depth + 1):
        if d == 0:
            noise.append(generate_noise([opt.nc_im, reals_shapes[d][2], reals_shapes[d][3]],
                                        device=opt.device).detach())
        else:
            if opt.train_mode == "generation" or opt.train_mode == "animation":
                noise.append(generate_noise([opt.nf, reals_shapes[d][2], reals_shapes[d][3]],
                                            device=opt.device).detach())
            else:
                noise.append(generate_noise([opt.nf, reals_shapes[d][2], reals_shapes[d][3]],
                                            device=opt.device).detach())

        # noise.append(torch.randn(size=reals_shapes[d], device=opt.device).detach())
        #
        # noise.append(generate_noise([opt.nc_im, reals_shapes[d][2], reals_shapes[d][3]],
        #                             device=opt.device).detach())

    return noise


def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    MSGGan = False
    if MSGGan:
        alpha = torch.rand(1, 1)
        alpha = alpha.to(device)  # cuda() #gpu) #if use_cuda else alpha

        interpolates = [alpha * rd + ((1 - alpha) * fd) for rd, fd in zip(real_data, fake_data)]
        interpolates = [i.to(device) for i in interpolates]
        interpolates = [torch.autograd.Variable(i, requires_grad=True) for i in interpolates]

        disc_interpolates = netD(interpolates)
    else:
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(device)  # cuda() #gpu) #if use_cuda else alpha

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.to(device)  # .cuda()
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                    # .cuda(), #if use_cuda else torch.ones(
                                    # disc_interpolates.size()),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    # LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def read_image(opt):
    x = img.imread('%s' % opt.input_name)
    x = np2torch(x, opt)
    x = x[:, 0:3, :, :]
    return x


def my_read_image(path, opt):
    # x = img.imread('%s' % opt.input_name)
    x = img.imread(path)
    x = np2torch(x, opt)
    x = x[:, 0:3, :, :]
    return x


def np2torch(x, opt):
    x = x.astype(np.float32)
    if opt.nc_im == 3:
        x = x[:, :, :, None]
        x = x.transpose((3, 2, 0, 1)) / 255
    else:
        x = color.rgb2gray(x)
        x = x[:, :, None, None]
        x = x.transpose(3, 2, 0, 1)
    x = torch.from_numpy(x)
    # if not (opt.not_cuda):
        # x = move_to_gpu(x)
    # x = x.type(torch.cuda.FloatTensor) if not (opt.not_cuda) else x.type(torch.FloatTensor)
    x = x.float()
    x = norm(x)
    return x


def save_networks(netG, netDs, opt):
    torch.save(netG.state_dict(), '%s/netG.pth' % (opt.outf))
    # torch.save(netG.module.state_dict(), '%s/netG.pth' % (opt.outf))
    if isinstance(netDs, list):
        for i, netD in enumerate(netDs):
            torch.save(netD.state_dict(), '%s/netD_%s.pth' % (opt.outf, str(i)))
            # torch.save(netD.module.state_dict(), '%s/netD_%s.pth' % (opt.outf, str(i)))
    else:
        torch.save(netDs.state_dict(), '%s/netD.pth' % (opt.outf))
        # torch.save(netDs.module.state_dict(), '%s/netD.pth' % (opt.outf))
    # torch.save(z, '%s/z_opt.pth' % (opt.outf))


def adjust_scales2image(real_, opt):
    opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3]]), 1)
    real = imresize(real_, opt.scale1, opt)

    # opt.stop_scale = opt.train_stages - 1
    opt.scale_factor = math.pow(opt.min_size / (min(real.shape[2], real.shape[3])), 1 / opt.stop_scale)
    return real


def create_reals_pyramid(real, opt):
    reals = []
    # use old rescaling method for harmonization
    if opt.train_mode == "harmonization":
        for i in range(opt.stop_scale):
            scale = math.pow(opt.scale_factor, opt.stop_scale - i)
            curr_real = imresize(real, scale, opt)
            reals.append(curr_real)
    # use new rescaling method for all other tasks
    else:
        for i in range(opt.stop_scale):
            scale = math.pow(opt.scale_factor,
                             ((opt.stop_scale - 1) / math.log(opt.stop_scale)) * math.log(opt.stop_scale - i) + 1)
            curr_real = imresize(real, scale, opt)
            reals.append(curr_real)
    reals.append(real)
    return reals


def generate_dir2save(opt):
    # training_image_name = opt.input_name[:-4].split("/")[-1]
    training_image_name = opt.input_name.split("/")[-1]
    dir2save = 'TrainedModels/{}/x{}_'.format(training_image_name, opt.max_size / opt.min_size)
    dir2save += opt.timestamp

    # dir2save += "_{}".format(opt.train_mode)
    # if opt.train_mode == "harmonization" or opt.train_mode == "editing":
    #     if opt.fine_tune:
    #         dir2save += "_{}".format("fine-tune")
    # dir2save += "_train_depth_{}_lr_scale_{}".format(opt.train_depth, opt.lr_scale)
    # if opt.batch_norm:
    #     dir2save += "_BN"
    # dir2save += "_act_" + opt.activation
    # if opt.activation == "lrelu":
    #     dir2save += "_" + str(opt.lrelu_alpha)

    return dir2save


def post_config(opt):
    # init fixed parameters
    opt.device = torch.device("cpu" if opt.not_cuda else "cuda")
    opt.noise_amp_init = opt.noise_amp
    opt.timestamp = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S')

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    opt.stop_scale = opt.train_stages - 1

    if torch.cuda.is_available() and opt.not_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    return opt


def load_config(opt):
    if not os.path.exists(opt.model_dir):
        print("Model not found: {}".format(opt.model_dir))
        exit()

    with open(os.path.join(opt.model_dir, 'parameters.txt'), 'r') as f:
        params = f.readlines()
        for param in params:
            param = param.split("-")
            param = [p.strip() for p in param]
            param_name = param[0]
            param_value = param[1]
            try:
                param_value = int(param_value)
            except ValueError:
                try:
                    param_value = float(param_value)
                except ValueError:
                    pass
            setattr(opt, param_name, param_value)
    return opt


# 计算生成器content loss所需的特征图
class FeatureExtractor(nn.Module):
    def __init__(self, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        model = vgg19(pretrained=True)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer+1)])

    def forward(self, x):
        return self.features(x)
