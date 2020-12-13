# -*- coding: utf-8 -*- #
# Author: Henry
# Date:   2020/9/23

import os
import math

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from skimage import io as img

import utils.functions as functions
import utils.imresize as imresize


def split_dataset(dataset_path, train=0.8, val=0.1, test=0.1):
    """
    划分训练集和测试集
    """
    train_list = list()
    val_list = list()
    test_list = list()

    # folders : [/NWPU-RESISC45/airplanes  ...]
    folders = [os.path.join(dataset_path, temp) for temp in os.listdir(dataset_path)]
    for folder in folders:
        # image_list : [/NWPU-RESISC45/airplanes/airplanes_001.jpg  ...]
        image_list = [os.path.join(folder, temp) for temp in os.listdir(folder)]

        n_total = len(image_list)
        n_train = int(n_total * train)
        n_val = int(n_total * val)

        # train_list.extend(image_list[:100])
        # val_list.extend(image_list[n_train: n_train + 50])
        # test_list.extend(image_list[n_train + n_val:50])

        train_list.extend(image_list[:n_train])
        val_list.extend(image_list[n_train: n_train + n_val])
        test_list.extend(image_list[n_train + n_val:])

    return train_list, val_list, test_list


class NWPU(Dataset):
    def __init__(self, image_list, opt):
        super().__init__()
        self.image_list = image_list
        self.opt = opt

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):

        image = functions.my_read_image(self.image_list[item], self.opt)
        image = functions.adjust_scales2image(image, self.opt)
        images = functions.create_reals_pyramid(image, self.opt)
        images = [temp.squeeze(0) for temp in images]
        return images, self.image_list[item]


if __name__ == '__main__':
    train_images, val_images, test_images = split_dataset("/home/henry/Datasets/NWPU-RESISC45")
    for i in range(10):
        a, b, c = split_dataset("/home/henry/Datasets/NWPU-RESISC45")
        print((train_images == a) and (val_images == b) and (test_images == c))
