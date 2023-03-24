"""
作者：didi
日期：2023年03月13日
内容：多波段图像分离
"""

import numpy as np
import os
from osgeo import gdal


# 保存tif文件函数
def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if dataset != None:
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i+1).WriteArray(im_data[i])
    del dataset


def bandsclip(path1, path2):

    dataset_img = gdal.Open(path1)

    num_bands = dataset_img.RasterCount
    print("波段数为：{}\n".format(num_bands))
    height = dataset_img.RasterYSize
    width = dataset_img.RasterXSize
    print('Image size is: {r} rows x {c} columns\n'.format(r=height, c=width))

    proj = dataset_img.GetProjection()
    print('Image projection:')
    print(proj + "\n")

    geotrans = dataset_img.GetGeoTransform()
    print('Image geo-transform:{gt}\n'.format(gt=geotrans))

    img = dataset_img.ReadAsArray(0, 0, width, height)

    img_out = []
    for i in range(img.shape[0]):
        img_out = np.array(img[i, ::])
        writeTiff(img_out, geotrans, proj, path2+'_'+str(i)+'.tif')  # 输出波段的名称命名格式可以修改，结合传递的path2参数


if __name__ == "__main__":
    os.chdir(r'AQ600_data')

    path1 = r'AQ600_0166_5boduan.tif'  # 要分离波段的原始图像数据名称
    path2 = r'AQ600_0166'      # 分离的各波段结果图像部分名称
    bandsclip(path1, path2)   # 调用上面定义的波段分离函数
    print('Bandsclip END!')

