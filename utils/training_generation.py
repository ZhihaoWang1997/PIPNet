import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
import cv2
import numpy as np
from tqdm import tqdm
from skimage import io as img
from PIL import Image
import nni

import utils.functions as functions
import utils.models as models
import utils.datasets as datasets
from utils.imresize import my_torch2uint8


def train(opt):
    print("Training model with the following parameters:")
    print("\t number of stages: {}".format(opt.train_stages))
    print("\t number of concurrently trained stages: {}".format(opt.train_depth))
    print("\t learning rate scaling: {}".format(opt.lr_scale))
    print("\t non-linearity: {}".format(opt.activation))

    # 加载数据集
    train_loader = DataLoader(datasets.NWPU(opt.train_images, opt),
                              batch_size=opt.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(datasets.NWPU(opt.val_images, opt),
                            batch_size=1, shuffle=True, num_workers=2)
    test_loader = DataLoader(datasets.NWPU(opt.test_images, opt),
                             batch_size=opt.batch_size, shuffle=False, num_workers=2)

    temp, _ = next(iter(train_loader))
    shapes = [temp[i].shape for i in range(len(temp))]
    print("Training on image pyramid: {}".format(shapes))
    del temp

    generator = init_G(opt)
    noise_amp = []

    # for scale_num in range(opt.stop_scale + 1):
    for scale_num in range(opt.stop_scale):
        opt.out_ = functions.generate_dir2save(opt)
        opt.outf = '%s/%d' % (opt.out_, scale_num)
        opt.logs_out = opt.out_ + '/logs'
        try:
            os.makedirs(opt.outf)
            os.makedirs(opt.logs_out)
        except OSError:
            print(OSError)
            pass

        d_curr = init_D(opt)
        if scale_num > 0:
            d_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out_, scale_num - 1)))
            # generator = generator.module
            generator.init_next_stage()

        writer = SummaryWriter(log_dir=opt.logs_out)
        noise_amp, generator, d_curr = train_single_scale(d_curr, generator, shapes,
                                                          train_loader, val_loader, test_loader,
                                                          noise_amp, opt, scale_num, writer)

        # torch.save(fixed_noise, '%s/fixed_noise.pth' % opt.out_)
        torch.save(generator, '%s/G.pth' % opt.out_)
        # torch.save(reals, '%s/reals.pth' % opt.out_)
        torch.save(noise_amp, '%s/noise_amp.pth' % opt.out_)
        del d_curr
    writer.close()
    return


def train_single_scale(netD, netG, shapes, train_loader, val_loader, test_loader, noise_amp, opt, depth, writer):

    alpha = opt.alpha

    ############################
    # define optimizers, learning rate schedulers, and learning rates for lower stages
    ###########################
    # setup optimizers for D
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, opt.beta2))

    # setup optimizers for G
    # remove gradients from stages that are not trained
    for block in netG.body[:-opt.train_depth]:
        for param in block.parameters():
            param.requires_grad = False

    # set different learning rate for lower stages
    parameter_list = [
        {"params": block.parameters(), "lr": opt.lr_g * (opt.lr_scale ** (len(netG.body[-opt.train_depth:]) - 1 - idx))}
        for idx, block in enumerate(netG.body[-opt.train_depth:])]

    # add parameters of head and tail to training
    if depth - opt.train_depth < 0:
        parameter_list += [{"params": netG.head.parameters(), "lr": opt.lr_g * (opt.lr_scale ** depth)}]
    parameter_list += [{"params": netG.tail.parameters(), "lr": opt.lr_g}]
    optimizerG = optim.Adam(parameter_list, lr=opt.lr_g, betas=(opt.beta1, opt.beta2))

    # define learning rate schedules
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD, milestones=[0.8 * opt.niter],
                                                      gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG, milestones=[0.8 * opt.niter],
                                                      gamma=opt.gamma)

    ############################
    # calculate noise_amp
    ###########################
    if depth == 0:
        noise_amp.append(opt.noise_amp_init)
    else:
        noise_amp.append(opt.noise_amp_init)

    ###########################
    # define losses
    ###########################
    content_loss = nn.SmoothL1Loss()
    adversarial_loss = nn.BCELoss()

    real_label = 1.
    fake_label = 0.
    # start training
    for iter_idx in range(opt.niter):
        for batch_id, (data, _) in enumerate(train_loader):
            real = data[depth + 1].to(opt.device)
            net_in = data[0].to(opt.device)

            ############################
            # (0) sample noise for unconditional generation
            ###########################
            noise = functions.sample_random_noise(depth + 1, shapes, opt)


            ############################
            # (1) Update D network: maximize D(x) + D(G(z))
            ###########################
            netD.zero_grad()

            # train with real
            output = netD(real)

            ones_label = torch.full(output.size(), real_label, dtype=torch.float, device=opt.device, requires_grad=False)
            zeros_label = torch.full(output.size(), fake_label, dtype=torch.float, device=opt.device, requires_grad=False)

            errD_real = adversarial_loss(output, ones_label)
            errD_real.backward()
            # D_x = output.mean().item()

            # train with fake
            fake = netG(noise, net_in, shapes, noise_amp)
            output = netD(fake.detach())

            errD_fake = adversarial_loss(output, zeros_label)
            errD_fake.backward()
            # D_G_z1 = output.mean().item()

            errD_total = errD_real + errD_fake
            # errD_total.backward()

            optimizerD.step()

            ############################
            # (2) Update G network: maximize D(G(z))
            ###########################
            netG.zero_grad()

            output = netD(fake)

            errG_f = adversarial_loss(output, ones_label)
            # errG_f.backward()

            D_G_z2 = output.mean().item()

            content_loss_f = content_loss(fake, real)

            errG_total = 1e-3 * errG_f + content_loss_f
            errG_total.backward()
            optimizerG.step()

            ############################
            # (3) Log Results
            ###########################
            batch2check = 10
            step_id = iter_idx * len(train_loader) + batch_id
            if batch_id % batch2check == 0 or iter_idx == (len(train_loader) - 1):
                writer.add_scalar('Stage{}/Loss/errD_real'.format(depth), errD_real.item(), step_id + 1)
                writer.add_scalar('Stage{}/Loss/errD_fake'.format(depth), errD_fake.item(), step_id + 1)
                writer.add_scalar('Stage{}/Loss/errD_total'.format(depth), errD_total.item(), step_id + 1)

                writer.add_scalar('Stage{}/Loss/errG_f'.format(depth), errG_f.item(), step_id + 1)
                writer.add_scalar('Stage{}/Loss/content_loss_f'.format(depth), content_loss_f.item(), step_id + 1)
                writer.add_scalar('Stage{}/Loss/errG_total'.format(depth), errG_total.item(), step_id + 1)

        schedulerD.step()
        schedulerG.step()

        ############################
        # (4) Validation
        ###########################
        # if iter_idx % 5 == 0 or iter_idx == (opt.niter - 1):
        val_nums = 20
        with torch.no_grad():
            for batch_id, (data, _) in enumerate(val_loader):

                step_id = iter_idx * val_nums + batch_id

                real = data[depth + 1].to(opt.device)

                noise = functions.sample_random_noise(depth + 1, shapes, opt)
                net_in = data[0].to(opt.device)

                fake = netG(noise, net_in, shapes, noise_amp)

                net_in = functions.denorm(net_in)
                imgs_lr_1 = torch.nn.functional.interpolate(net_in, size=[shapes[depth+1][2], shapes[depth+1][3]],
                                                            mode='bicubic', align_corners=True)

                # 保存图像到tensorboard
                fake = functions.denorm(fake)
                real = functions.denorm(real)

                imgs_lr = make_grid(imgs_lr_1, nrow=1, normalize=False)
                imgs_fake = make_grid(fake, nrow=1, normalize=False)
                imgs_real = make_grid(real, nrow=1, normalize=False)
                img_grid = torch.cat((imgs_lr, imgs_fake, imgs_real), -1)
                save_image(img_grid, "{}/stage{}_epoch{}_val_{}.jpg".format(opt.outf, depth, iter_idx, step_id),
                           normalize=False)

                writer.add_image("Stage_{}/Epoch_{}/Val_{}".format(depth, iter_idx, batch_id),
                                 img_grid, step_id)

                if batch_id == (val_nums - 1):
                    break

    functions.save_networks(netG, netD, opt)
    return noise_amp, netG, netD


def generate_samples(netG, opt, depth, noise_amp, writer, reals, iter, rec_image=None, n=10):
    # opt.out_ = functions.generate_dir2save(opt)
    # dir2save = '{}/gen_samples_stage_{}'.format(opt.out_, depth)
    reals_shapes = [r.shape for r in reals]
    all_images = []
    # try:
    #     os.makedirs(dir2save)
    # except OSError:
    #     pass
    with torch.no_grad():
        for idx in range(n):
            noise = functions.sample_random_noise(depth, reals_shapes, opt)
            sample = netG(noise, reals_shapes, noise_amp)
            all_images.append(sample)
            # functions.save_image('{}/gen_sample_{}.jpg'.format(dir2save, idx), sample.detach(), opt)
        # rec_image = netG(fixed_noise, reals_shapes, noise_amp)

        all_images = torch.cat(all_images, 0)
        all_images[0] = reals[depth].squeeze()
        if rec_image is not None:
            all_images[1] = rec_image
        grid = make_grid(all_images, nrow=min(5, n), normalize=True)
        writer.add_image('gen_images_{}'.format(depth), grid, iter)


def init_G(opt):
    # generator initialization:
    # netG = models.GrowingGenerator(opt).to(opt.device)
    netG = models.PIPNet(opt).to(opt.device)
    netG.apply(models.weights_init)
    # print(netG)

    return netG


def init_D(opt):
    # discriminator initialization:
    netD = models.Discriminator(opt).to(opt.device)
    netD.apply(models.weights_init)
    # print(netD)

    return netD
