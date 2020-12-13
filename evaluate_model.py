import os
import torch

from torch.utils.data import DataLoader
import cv2
from tqdm import tqdm
from torchvision.utils import make_grid, save_image

from utils.config import get_arguments
import utils.functions as functions
from utils.evaluations import *
import utils.datasets as datasets
from utils.imresize import my_torch2uint8
import utils.evaluations as evals

import skimage.metrics as metric


def make_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--model_dir', help='input image name', default='/home/henry/RS_ConSinGAN/TrainedModels/NWPU/2020-10-22_15:26_1')
    parser.add_argument('--gpu', type=int, help='which GPU', default='0')
    parser.add_argument('--num_samples', type=int, help='which GPU', default=10)

    opt = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    _gpu = opt.gpu
    # _naive_img = opt.naive_img
    __model_dir = opt.model_dir
    opt = functions.load_config(opt)
    opt.gpu = _gpu
    # opt.naive_img = _naive_img
    opt.model_dir = __model_dir

    if torch.cuda.is_available():
        # torch.cuda.set_device(opt.gpu)
        opt.device = "cuda:{}".format(opt.gpu)

    dir2save = os.path.join(opt.model_dir, "Evaluation")
    make_dir(dir2save)

    print("Loading models...")
    netG = torch.load('%s/G.pth' % opt.model_dir, map_location="cuda:{}".format(torch.cuda.current_device()))
    # fixed_noise = torch.load('%s/fixed_noise.pth' % opt.model_dir,
    #                          map_location="cuda:{}".format(torch.cuda.current_device()))
    # reals = torch.load('%s/reals.pth' % opt.model_dir, map_location="cuda:{}".format(torch.cuda.current_device()))
    noise_amp = torch.load('%s/noise_amp.pth' % opt.model_dir,
                           map_location="cuda:{}".format(torch.cuda.current_device()))
    # reals_shapes = [r.shape for r in reals]

    train_images, val_images, test_images = datasets.split_dataset(opt.input_name)
    val_loader = DataLoader(datasets.NWPU(val_images, opt),
                            batch_size=1, shuffle=True, num_workers=1)

    temp, _ = next(iter(val_loader))
    shapes = [temp[i].shape for i in range(len(temp))]
    print("Testing on image pyramid: {}".format(shapes))
    del temp

    if opt.train_mode == "generation" or opt.train_mode == "retarget":
        with torch.no_grad():
            # for batch_id, (data, img_name) in enumerate(val_loader):
            for batch_id, (data, img_name) in enumerate(tqdm(val_loader)):

                real = data[-1].to(opt.device)
                img_name = img_name[0].split('/')[-1]

                noise = functions.sample_random_noise(len(shapes)-1, shapes, opt)
                net_in = data[0].to(opt.device)

                fake = netG(noise, net_in, shapes, noise_amp)

                # 计算指标
                fake_1 = my_torch2uint8(fake)
                real_1 = my_torch2uint8(real)
                imgs_lr_1 = torch.nn.functional.interpolate(net_in, size=[shapes[-1][2], shapes[-1][3]],
                                                            mode='bicubic', align_corners=True)
                interp_1 = my_torch2uint8(imgs_lr_1)

                val_psnr = evals.psnr(real_1, fake_1)
                val_ssim = evals.m_ssim(real_1, fake_1)
                interp_psnr = evals.psnr(real_1, interp_1)
                interp_ssim = evals.m_ssim(real_1, interp_1)

                img2save = np.squeeze(fake_1) #.transpose(2, 0, 1)
                img2save_real = np.squeeze(real_1)
                img2save_inter = np.squeeze(interp_1)

                if val_psnr >= 28.:
                    cv2.imwrite(f"{dir2save}/{img_name.split('.')[0]}_{val_psnr:.4f}_{val_ssim:.4f}_fake.jpg",
                                img2save)

                    cv2.imwrite(f"{dir2save}/{img_name.split('.')[0]}_real.jpg",
                                img2save_real)

                    cv2.imwrite(f"{dir2save}/{img_name.split('.')[0]}_{interp_psnr:.4f}_{interp_ssim:.4f}_inter.jpg",
                                img2save_inter)

                    # img2save = functions.denorm(fake)
                    # fake2save = make_grid(img2save, nrow=1, padding=0, normalize=False)
                    # real2save = make_grid(functions.denorm(real), nrow=1, padding=0, normalize=False)
                    # inter2save = make_grid(functions.denorm(imgs_lr_1), nrow=1, padding=0, normalize=False)
                    # save_image(fake2save, f"{dir2save}/{val_psnr:.4f}_{val_ssim:.4f}_{img_name.split('.')[0]}_fake.jpg",
                    #            normalize=False)
                    # save_image(real2save, f"{dir2save}/{img_name.split('.')[0]}_real.jpg", normalize=False)
                    # save_image(inter2save, f"{dir2save}/{interp_psnr:.4f}_{interp_ssim:.4f}_{img_name.split('.')[0]}_inter.jpg",
                    #            normalize=False)



                # break

    print("Done. Results saved at: {}".format(dir2save))
