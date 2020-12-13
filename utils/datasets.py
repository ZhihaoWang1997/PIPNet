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
