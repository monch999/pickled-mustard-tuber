# -*-coding = utf-8 -*-
# @time : 2022/10/6 12:23
# @Author: 自在清风
# @File ： Fusion.py
# @Software ：PyCharm
import os
import time

import numpy as np
from sklearn.decomposition import PCA
import cv2 as cv
import tqdm

from TIF_read_write import read_img, write_img


def getListFiles(path, suffix: 'str' = 'raw'):
    """
    Obtain all files in the file directory (including files in sub folders)
    :param path: File Path
    :param suffix: file suffix
    :return: Absolute path, file name
    """
    assert os.path.isdir(path), '%s not exist,' % path
    ret = []
    names = []
    for root, dirs, files in os.walk(path):
        # print(root)
        for filespath in files:
            # print(filespath.rpartition('.')[-1])
            if filespath.rpartition('.')[-1] == suffix:
                ret.append(os.path.join(root, filespath))
                names.append(filespath.rpartition('.')[0])
    return ret, names


def sigmoid(x):
    x_exp = np.exp(-x)
    return 1 / (1 + x_exp)


def get_data(img_path, suffix: 'str' = 'raw', bands=9):
    envi_data = ['dat', 'raw']
    data_list = []
    if suffix in envi_data:
        im_proj, im_geotrans, data, w, h = read_img(img_path)
        for band in range(bands):
            d = np.ravel(data[band])  # Expanding a band of data in an array into one dimension
            data_list.append(d)
        return im_proj, im_geotrans, np.array(data_list).transpose((1, 0)), h, w
    else:
        data = cv.imread(img_path, 1)
        return data


def guideFilter(g, I, winSize, eps):
    mean_I = cv.boxFilter(I, ddepth=-1, ksize=winSize, normalize=1)  # Mean smoothing of I
    mean_g = cv.boxFilter(g, ddepth=-1, ksize=winSize, normalize=1)  # Mean smoothing of g

    mean_gg = cv.boxFilter(g * g, ddepth=-1, ksize=winSize, normalize=1)  # Mean smoothing of I*I
    mean_Ig = cv.boxFilter(I * g, ddepth=-1, ksize=winSize, normalize=1)  # Mean smoothing of I*g

    var_g = mean_gg - mean_g * mean_g  # variance
    cov_Ip = mean_Ig - mean_I * mean_g  # covariance

    a = cov_Ip / (var_g + eps)  # Correlation factor a
    b = mean_I - a * mean_g  # 相Correlation factor b

    mean_a = cv.boxFilter(a, ddepth=-1, ksize=winSize, normalize=1)  # Perform mean smoothing on a
    mean_b = cv.boxFilter(b, ddepth=-1, ksize=winSize, normalize=1)  # Perform mean smoothing on b

    out = mean_a * g + mean_b
    return out


def wmap(img1, img2):
    # Weight mapping
    data = (img1 == np.maximum(img1, img2)) * 1.0
    return data


def guider_fusion(img1, img2):
    # base layer
    base1 = cv.boxFilter(img1, -1, (31, 31), normalize=1)
    base2 = cv.boxFilter(img2, -1, (31, 31), normalize=1)

    # detail layer
    detail1 = img1 - base1
    detail2 = img2 - base2

    detail1[detail1 < 0] = 0

    # Laplace filtering
    h1 = abs(cv.Laplacian(img1, -1))
    h2 = abs(cv.Laplacian(img2, -1))

    # Gaussian filter
    s1 = cv.GaussianBlur(h1, ksize=(11, 11), sigmaX=5, sigmaY=5)
    s2 = cv.GaussianBlur(h2, ksize=(11, 11), sigmaX=5, sigmaY=5)

    # Obtain weight matrix
    p1 = wmap(s1, s2)
    p2 = wmap(s2, s1)

    eps1 = 0.01
    eps2 = 0.001

    wb1 = guideFilter(p1, img1, (8, 8), eps1)
    wb2 = guideFilter(p2, img2, (8, 8), eps1)

    wd1 = guideFilter(p1, img1, (4, 4), eps2)
    wd2 = guideFilter(p2, img2, (4, 4), eps2)

    wb1 = sigmoid(wb1)
    wb2 = sigmoid(wb2)
    wd1 = sigmoid(wd1)
    wd2 = sigmoid(wd2)

    B = base1 * wb1 + base2 * wb2
    D = detail1 * wd1 + detail2 * wd2

    im = B + D
    im[im < 0] = 0
    return im


def TwoPercentLinear(image, max_out=255, min_out=0):
    b, g, r = cv.split(image)  # Separate three bands

    def gray_process(gray, maxout=max_out, minout=min_out):
        high_value = np.percentile(gray, 98)  # Obtain the corresponding grayscale at 98% of the histogram
        low_value = np.percentile(gray, 2)
        truncated_gray = np.clip(gray, a_min=low_value, a_max=high_value)
        processed_gray = ((truncated_gray - low_value) / (high_value - low_value)) * (maxout - minout)  # Linear stretching
        return processed_gray

    r_p = gray_process(r)
    g_p = gray_process(g)
    b_p = gray_process(b)
    result = np.dstack((b_p, g_p, r_p))
    return np.uint8(result)


def fusion(img1_path, img2_path='', img_save=''):
    if not os.path.exists(img_save):
        os.mkdir(img_save)

    img1_path, img1_name = getListFiles(img1_path, suffix='png')
    img2_path, _ = getListFiles(img2_path, suffix='png')
    for i in range(len(img1_path)):
        data = cv.imread(img2_path[i], 1) / 255.0
        b, g, r = np.array(np.dsplit(data, 3)).squeeze()

        data1 = cv.imread(img1_path[i], 1) / 255.0
        data_fusion1, data_fusion2, data_fusion3 = np.array(np.dsplit(data1, 3)).squeeze()


        # Guided Fusion
        rs_data = np.stack((data_fusion1, data_fusion2, data_fusion3)).transpose((1, 2, 0))
        rgb_data = np.stack((r, g, b)).transpose((1, 2, 0))
        data_fusion = guider_fusion(rs_data, rgb_data)

        # print(data_fusion.shape)
        img = TwoPercentLinear(data_fusion)
        img = cv.resize(img, (256, 256))
        img_road = os.path.join(img_save, img1_name[i] + '.png')
        cv.imwrite(img_road, img)


if __name__ == '__main__':
    start = time.time()
    fusion(img1_path=r'data/ms_843',  # MS images
           img2_path=r'data/hd',  # hd images
           img_save=r'data/fusion')  # save fusion images
    end = time.time()
    print('run:%.2fs' % (end - start))

    print('\r')
    print('done'.center(50, '*'))
