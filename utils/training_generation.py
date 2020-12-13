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
# import utils.models as models
import utils.models_new as models
import utils.datasets as datasets
import utils.evaluations as evals
from utils.imresize import my_torch2uint8


def train(opt):
    print("Training model with the following parameters:")
    print("\t number of stages: {}".format(opt.train_stages))
    print("\t number of concurrently trained stages: {}".format(opt.train_depth))
    print("\t learning rate scaling: {}".format(opt.lr_scale))
    print("\t non-linearity: {}".format(opt.activation))

    # 加载数据集
    train_images, val_images, test_images = datasets.split_dataset(opt.input_name)
    train_loader = DataLoader(datasets.NWPU(train_images, opt),
                              batch_size=opt.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(datasets.NWPU(val_images, opt),
                            batch_size=1, shuffle=True, num_workers=2)
    test_loader = DataLoader(datasets.NWPU(test_images, opt),
                             batch_size=opt.batch_size, shuffle=False, num_workers=2)

    temp, _ = next(iter(train_loader))
    shapes = [temp[i].shape for i in range(len(temp))]
    print("Training on image pyramid: {}".format(shapes))
    del temp

    # real = functions.read_image(opt)
    # real = functions.adjust_scales2image(real, opt)
    # reals = functions.create_reals_pyramid(real, opt)
    # print("Training on image pyramid: {}".format([r.shape for r in reals]))
    # print("")

    generator = init_G(opt)
    # fixed_noise = []
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

    # reals_shapes = [real.shape for real in reals]
    # real = reals[depth]

    ############################
    # define z_opt for training on reconstruction
    ###########################
    # if depth == 0:
    #     if opt.train_mode == "generation" or opt.train_mode == "retarget":
    #         z_opt = reals[0]
    # else:
    #     if opt.train_mode == "generation" or opt.train_mode == "animation":
    #         z_opt = functions.generate_noise([opt.nf,
    #                                           reals_shapes[depth][2], reals_shapes[depth][3]],
    #                                          device=opt.device)
    # fixed_noise.append(z_opt.detach())

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
        # noise_amp.append(1)
        noise_amp.append(opt.noise_amp_init)
    else:
        # noise_amp.append(0)
        # z_reconstruction = netG(fixed_noise, reals_shapes, noise_amp)
        #
        # criterion = nn.MSELoss()
        # rec_loss = criterion(z_reconstruction, real)
        #
        # RMSE = torch.sqrt(rec_loss).detach()
        # _noise_amp = opt.noise_amp_init * RMSE
        noise_amp.append(opt.noise_amp_init)

    ###########################
    # define losses
    ###########################
    feature_extractor = functions.FeatureExtractor()
    feature_extractor = feature_extractor.to(opt.device)
    # content_loss = nn.MSELoss()
    content_loss = nn.SmoothL1Loss()
    adversarial_loss = nn.BCELoss()
    # adversarial_loss = nn.CrossEntropyLoss()
    l1_loss = nn.L1Loss()

    real_label = 1.
    fake_label = 0.
    # start training
    # _iter = tqdm(range(opt.niter))
    for iter_idx in range(opt.niter):

        # _iter.set_description('stage [{}/{}]:'.format(depth, opt.stop_scale))
        for batch_id, (data, _) in enumerate(train_loader):

            # reals = data.to(opt.device)
            # real = data[depth].to(opt.device)
            real = data[depth + 1].to(opt.device)
            net_in = data[0].to(opt.device)

            ############################
            # (0) sample noise for unconditional generation
            ###########################
            noise = functions.sample_random_noise(depth + 1, shapes, opt)
            # noise[0] = data[0].to(opt.device)

            ############################
            # (1) Update D network: maximize D(x) + D(G(z))
            ###########################
            # for j in range(opt.Dsteps):
            #     # train with real
            #     netD.zero_grad()
            #     output = netD(real)
            #     errD_real = -output.mean()
            #
            #     # train with fake
            #     if j == opt.Dsteps - 1:
            #         fake = netG(noise, reals_shapes, noise_amp)
            #     else:
            #         with torch.no_grad():
            #             fake = netG(noise, reals_shapes, noise_amp)
            #
            #     output = netD(fake.detach())
            #     errD_fake = output.mean()
            #
            #     gradient_penalty = functions.calc_gradient_penalty(netD, real, fake, opt.lambda_grad, opt.device)
            #     errD_total = errD_real + errD_fake + gradient_penalty
            #     errD_total.backward()
            #     optimizerD.step()
            # ----------------------------------- #

            netD.zero_grad()

            # train with real
            output = netD(real)

            ones_label = torch.full(output.size(), real_label, dtype=torch.float, device=opt.device, requires_grad=False)
            zeros_label = torch.full(output.size(), fake_label, dtype=torch.float, device=opt.device, requires_grad=False)

            errD_real = adversarial_loss(output, ones_label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            fake = netG(noise, net_in, shapes, noise_amp)
            output = netD(fake.detach())

            errD_fake = adversarial_loss(output, zeros_label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            errD_total = errD_real + errD_fake
            # errD_total.backward()

            optimizerD.step()

            ############################
            # (2) Update G network: maximize D(G(z))
            ###########################
            # output = netD(fake)
            # errG = -output.mean()
            #
            # if alpha != 0:
            #     loss = nn.MSELoss()
            #     rec = netG(fixed_noise, reals_shapes, noise_amp)
            #     rec_loss = alpha * loss(rec, real)
            # else:
            #     rec_loss = 0
            #
            # netG.zero_grad()
            # errG_total = errG + rec_loss
            # errG_total.backward()
            #
            # for _ in range(opt.Gsteps):
            #     optimizerG.step()
            # -----------------------------------

            netG.zero_grad()

            output = netD(fake)

            errG_f = adversarial_loss(output, ones_label)
            # errG_f.backward()

            D_G_z2 = output.mean().item()

            content_loss_f = content_loss(fake, real)
            fake_feature_loss = l1_loss(feature_extractor(fake), feature_extractor(real).detach())

            errG_total = 1e-3 * errG_f + content_loss_f #+ 1e-2 * fake_feature_loss
            # errG_total = content_loss_f #+ 1e-1 * fake_feature_loss
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
                writer.add_scalar('Stage{}/Loss/fake_feature_loss'.format(depth), fake_feature_loss.item(), step_id + 1)
                writer.add_scalar('Stage{}/Loss/errG_total'.format(depth), errG_total.item(), step_id + 1)

                # saved_real = my_torch2uint8(real)
                # saved_fake = my_torch2uint8(fake)
                # gen_mse = evals.mse(saved_real, saved_fake)
                # gen_psnr = evals.psnr(saved_real, saved_fake)
                # gen_mssim = evals.m_ssim(saved_real, saved_fake)

                # writer.add_scalar('Stage{}/Image Quality/gen_mse'.format(depth), gen_mse, step_id + 1)
                # writer.add_scalar('Stage{}/Image Quality/gen_psnr'.format(depth), gen_psnr, step_id + 1)
                # writer.add_scalar('Stage{}/Image Quality/gen_ssim'.format(depth), gen_mssim, step_id + 1)

                print('[%d/%d][%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (depth+1, opt.train_stages - 1, iter_idx, opt.niter, batch_id, len(train_loader),
                         errD_total.item(), errG_total.item(), D_x, D_G_z1, D_G_z2))

            # break

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

                # 计算指标
                fake_1 = my_torch2uint8(fake)
                real_1 = my_torch2uint8(real)
                img2save = np.squeeze(np.concatenate((fake_1, real_1), 2))

                net_in = functions.denorm(net_in)
                imgs_lr_1 = torch.nn.functional.interpolate(net_in, size=[shapes[depth+1][2], shapes[depth+1][3]],
                                                            mode='bicubic', align_corners=True)
                interp_1 = my_torch2uint8(imgs_lr_1)

                val_psnr = evals.psnr(real_1, fake_1)
                val_ssim = evals.m_ssim(real_1, fake_1)
                # interp_psnr = evals.psnr(real_1, interp_1)
                # interp_ssim = evals.m_ssim(real_1, interp_1)

                # val_psnr = nn.MSELoss()(fake, real).item()

                cv2.imwrite("{}/real_S{}E{}V{}_{}_{}.jpg".format(
                    opt.outf, depth, iter_idx, step_id, val_psnr, val_ssim), img2save)

                writer.add_scalar('Stage{}/Val Image Quality/val_psnr'.format(depth), val_psnr, step_id + 1)
                writer.add_scalar('Stage{}/Val Image Quality/val_ssim'.format(depth), val_ssim, step_id + 1)
                # writer.add_scalar('Stage{}/Val Image Quality/interp_psnr'.format(depth), interp_psnr, step_id + 1)
                # writer.add_scalar('Stage{}/Val Image Quality/interp_ssim'.format(depth), interp_ssim, step_id + 1)

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

        #         break
        # break

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
