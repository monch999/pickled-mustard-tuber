# -*- coding = utf-8 -*-
# @Time : 2021/10/5 20:29
# @Author : 自在清风
# @File : dataset.py
# @software: PyCharm
import argparse
import pickle
import glob
from abc import ABC
from PIL import Image
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

"""data load program"""


class UNetDataset(Dataset, ABC):
    def __init__(self, image_list=None, mask_list=None, phase='train'):
        super(UNetDataset, self).__init__()
        self.phase = phase
        # read images
        if phase != 'test':
            assert mask_list, 'mask list must given when training'
            self.mask_file_list = mask_list
            self.img_file_list = [path.replace('mask', 'image') for path in mask_list]
            assert len(self.img_file_list) == len(self.mask_file_list), 'the count of image and mask not equal'
        elif phase == 'test':
            assert image_list, 'image list must given when testing'
            self.img_file_list = image_list

    def __getitem__(self, idx):
        img_name = self.img_file_list[idx]
        img = cv2.imread(img_name, 1)
        img = img.transpose((2, 0, 1))
        # print(img.shape)
        # img = np.expand_dims(img, 0)

        mask_name = self.mask_file_list[idx]
        mask = cv2.imread(mask_name, 0).astype(int)
        mask[mask <= 128] = 0
        mask[mask > 128] = 1
        mask = np.expand_dims(mask, 0)

        # assert (np.array(img.shape[1:]) == np.array(mask.shape[1:])).all(), (img.shape[1:], mask.shape[1:])

        return torch.from_numpy(img).to(torch.float32), torch.LongTensor(mask)

    def __len__(self):
        return len(self.img_file_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create dataset')  # 创建解析器
    parser.add_argument('--images', '-1', help='image dir', default='')  # 添加参数
    parser.add_argument('--masks', '-m', help='mask dir', default='')
    parser.add_argument('--target', help='target dir', default='')
    args = parser.parse_args()  # 解析

    if not args.target:
        mask_list = []
        mask_path = glob.glob(r'data/train/fusion_data/mask/*.png')  # 返回目录下所有文件

        dataset = UNetDataset(mask_list=mask_path, phase='train')
        data_loader = DataLoader(
            dataset, batch_size=30, shuffle=True, num_workers=8, pin_memory=False
        )
        print(len(dataset))
        count = 0.0
        pos = 0.0
        for i, (data, target) in enumerate(data_loader):
            print('i', i)
            print(data.shape)
            print(target.shape)
            # count += np.prod(data.size())
            # print(data.size())
            # pos += (data == 1).sum()
        # print(pos / count)
