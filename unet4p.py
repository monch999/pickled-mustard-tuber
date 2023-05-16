# -*- coding = utf-8 -*-
# @Time : 2021/11/11 14:54
# @Author : 自在清风
# @File : unet4p.py
# @software: PyCharm
import math

import torch
from torch import nn
import torch.nn.functional as f
import time


def tow_layer_conv(in_dim, out_dim):
    layer = [nn.Conv2d(in_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
             nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
             nn.Conv2d(out_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
             nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True)]
    return nn.Sequential(*layer)


def tow_layer_conv_with_pool(in_dim, out_dim):
    layer = [nn.MaxPool2d(kernel_size=2, stride=2),
             nn.Conv2d(in_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
             nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
             nn.Conv2d(out_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
             nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True)]
    return nn.Sequential(*layer)


def three_layer_conv_with_pool(in_dim, middle_dim, out_dim):
    layer = [nn.MaxPool2d(kernel_size=2, stride=2),
             nn.Conv2d(in_dim, middle_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
             nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
             nn.Conv2d(middle_dim, middle_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
             nn.BatchNorm2d(middle_dim), nn.ReLU(inplace=True),
             nn.Conv2d(middle_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
             nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True)]
    return nn.Sequential(*layer)


def three_layer_conv(in_dim, middle_dim, out_dim):
    layer = [
        nn.Conv2d(in_dim, middle_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
        nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
        nn.Conv2d(middle_dim, middle_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
        nn.BatchNorm2d(middle_dim), nn.ReLU(inplace=True),
        nn.Conv2d(middle_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
        nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True)]
    return nn.Sequential(*layer)


def convTranspose(in_dim, out_dim):
    layer = [nn.ConvTranspose2d(in_dim, out_dim, kernel_size=(2, 2), stride=(2, 2)),
             nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
             ]
    return nn.Sequential(*layer)


def conv3_3(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
                         nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True))


def conv1_1(in_dim, out_dim):
    return nn.Conv2d(in_dim, out_dim, kernel_size=(1, 1))


def tow_conv_without_RELU(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
                         nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
                         nn.Conv2d(out_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
                         nn.BatchNorm2d(out_dim))


def three_conv_without_RELU(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
                         nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
                         nn.Conv2d(out_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
                         nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
                         nn.Conv2d(out_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
                         nn.BatchNorm2d(out_dim))


class Unet4P(nn.Module):
    """based on the backbone of VGG16"""

    def __init__(self, color_dim=1, num_classes=2):
        super(Unet4P, self).__init__()
        self.conv_x11 = tow_layer_conv(color_dim, 16)
        self.conv_x12 = tow_layer_conv_with_pool(16, 32)
        self.conv_x13 = three_layer_conv_with_pool(32, 64, 64)
        self.conv_x14 = three_layer_conv_with_pool(64, 128, 128)
        self.conv_x15 = three_layer_conv_with_pool(128, 256, 256)

        self.conv_x21 = tow_layer_conv(32, 16)
        self.conv_x22 = tow_layer_conv(64, 32)
        self.conv_x23 = three_layer_conv(128, 64, 64)
        self.conv_x24 = three_layer_conv(256, 128, 128)

        self.conv_x31 = tow_layer_conv(48, 16)
        self.conv_x32 = tow_layer_conv(96, 32)
        self.conv_x33 = three_layer_conv(192, 64, 64)

        self.conv_x41 = tow_layer_conv(64, 16)
        self.conv_x42 = tow_layer_conv(96, 32)

        self.conv_x51 = tow_layer_conv(80, 16)

        self.up_conv_x21 = convTranspose(32, 16)
        self.up_conv_x22 = convTranspose(64, 32)
        self.up_conv_x23 = convTranspose(128, 64)
        self.up_conv_x24 = convTranspose(256, 128)

        self.conv = conv1_1(16, num_classes)
        self.conv13 = conv3_3(64, 16)
        self.conv14 = conv3_3(128, 32)
        self.conv15 = conv3_3(256, 64)
        self.conv14_41 = conv3_3(128, 16)
        self.conv15_51 = conv3_3(256, 16)

    def forward(self, x):
        x11 = self.conv_x11(x)
        x12 = self.conv_x12(x11)
        x21_1 = self.up_conv_x21(x12)
        x21 = torch.cat((x11, x21_1), 1)
        x21 = self.conv_x21(x21)
        out21 = self.conv(x21)

        x13 = self.conv_x13(x12)
        x22_1 = self.up_conv_x22(x13)
        x22 = torch.cat((x12, x22_1), 1)
        x22 = self.conv_x22(x22)
        x31_1 = self.up_conv_x21(x22)
        x31_2 = f.interpolate(x13, scale_factor=4, mode='bilinear', align_corners=True)
        x31_2 = self.conv13(x31_2)

        x31 = torch.cat([x31_1, x31_2, x21], 1)
        x31 = self.conv_x31(x31)
        out31 = self.conv(x31)

        x14 = self.conv_x14(x13)
        x23_1 = self.up_conv_x23(x14)
        x23 = torch.cat([x13, x23_1], 1)
        x23 = self.conv_x23(x23)
        x32_1 = self.up_conv_x22(x23)
        x32_2 = f.interpolate(x14, scale_factor=4, mode='bilinear', align_corners=True)
        x32_2 = self.conv14(x32_2)
        x32 = torch.cat([x32_1, x22, x32_2], 1)
        x32 = self.conv_x32(x32)
        x41_1 = self.up_conv_x21(x32)
        x41_2 = f.interpolate(x23, scale_factor=4, mode='bilinear', align_corners=True)
        x41_2 = self.conv13(x41_2)
        x41_3 = f.interpolate(x14, scale_factor=8, mode='bilinear', align_corners=True)
        x41_3 = self.conv14_41(x41_3)
        x41 = torch.cat([x41_1, x41_2, x31, x41_3], 1)
        x41 = self.conv_x41(x41)
        out41 = self.conv(x41)

        x15 = self.conv_x15(x14)
        x24_1 = self.up_conv_x24(x15)
        x24 = torch.cat([x24_1, x14], 1)
        x24 = self.conv_x24(x24)
        x33_1 = self.up_conv_x23(x24)
        x33_2 = f.interpolate(x15, scale_factor=4, mode='bilinear', align_corners=True)
        x33_2 = self.conv15(x33_2)
        x33 = torch.cat([x33_1, x23, x33_2], 1)
        x33 = self.conv_x33(x33)
        x42_1 = self.up_conv_x22(x33)
        x42_2 = f.interpolate(x24, scale_factor=4, mode='bilinear', align_corners=True)
        x42_2 = self.conv14(x42_2)
        x42 = torch.cat([x42_1, x32, x42_2], 1)
        x42 = self.conv_x42(x42)
        x51_1 = self.up_conv_x21(x42)
        x51_2 = f.interpolate(x33, scale_factor=4, mode='bilinear', align_corners=True)
        x51_2 = self.conv13(x51_2)
        x51_3 = f.interpolate(x24, scale_factor=8, mode='bilinear', align_corners=True)
        x51_3 = self.conv14_41(x51_3)
        x51_4 = f.interpolate(x15, scale_factor=16, mode='bilinear', align_corners=True)
        x51_4 = self.conv15_51(x51_4)
        x51 = torch.cat([x51_1, x51_2, x51_3, x51_4, x41], 1)
        x51 = self.conv_x51(x51)
        out51 = self.conv(x51)

        return out51 + out41 + out31 + out21


if __name__ == '__main__':
    unet_2d = Unet4P(3, 2)
    x = torch.rand(4, 3, 256, 256)
    x = torch.autograd.Variable(x)
    print(x.size())
    start_time = time.time()
    y = unet_2d(x)
    end_time = time.time()
    print('-' * 50)
    print('run time: %.2f' % (end_time - start_time))
    print(y.size())
