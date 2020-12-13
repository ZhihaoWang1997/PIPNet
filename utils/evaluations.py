# -*- coding: utf-8 -*- #
# Author: Henry

import numpy as np
from skimage import metrics


def mse(img1, img2):
    out = metrics.mean_squared_error(img1, img2)
    return out


def psnr(img1, img2):
    # out = measure.compare_psnr(img1, img2, data_range=65535)
    out = metrics.peak_signal_noise_ratio(img1, img2)
    return out


def mpsnr(img1, img2):
    n_bands = img1.shape[2]
    p = [metrics.peak_signal_noise_ratio(img1[:, :, k], img2[:, :, k],
                                         dynamic_range=np.max(img1[:, :, k])) for k in range(n_bands)]
    return np.mean(p)


def m_ssim(img1, img2):

    out = 0.
    for i in range(img1.shape[0]):
        image_1, image_2 = img1[i], img2[i]
        out += metrics.structural_similarity(image_1, image_2, multichannel=True)
    out /= img1.shape[0]
    return out


if __name__ == '__main__':

    a = np.random.randn(16, 32, 32, 3)
    b = a
    c = a + 0.1
    print(m_ssim(a, c))
