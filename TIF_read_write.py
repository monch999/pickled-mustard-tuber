# -*- coding = utf-8 -*-
# @Time : 2021/8/26 15:55
# @Author : 自在清风
# @File : TIF_read_write.py
# @software: PyCharm

import os
import numpy as np
from osgeo import gdal


# os.environ['PROJ_LIB'] = r'D:\Anaconda3\envs\cloneTF21\Library\share\proj'


def read_img(file_name):
    """
    读取遥感数据文件
    :param file_name: 文件路径
    :return: im_proj, im_geotrans, im_data, im_width, im_height
    """
    dataset = gdal.Open(file_name)  # open file

    im_width = dataset.RasterXSize  # columns
    im_height = dataset.RasterYSize  # rows

    im_geotrans = dataset.GetGeoTransform()  # Affine Matrix
    im_proj = dataset.GetProjection()  # Map projection information
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # Write the data as an array
    im_data[np.isnan(im_data)] = 0

    del dataset  # delete dataset
    return im_proj, im_geotrans, im_data, im_width, im_height


# write image to "tif"
def write_img(file_name, im_proj, im_geotrans, im_data):
    # Determine the data type of raster data
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # Interpreting array dimensions
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    # create file
    driver = gdal.GetDriverByName("ENVI")  # The data type must be present.
    dataset = driver.Create(file_name, im_width, im_height, im_bands, datatype)

    dataset.SetGeoTransform(im_geotrans)  # Write affine transformation Parameters
    dataset.SetProjection(im_proj)  # Write Projection

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  # Write array data
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    del dataset
