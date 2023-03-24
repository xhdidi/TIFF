"""
作者：didi
日期：2023年03月13日
内容：单波段图像组合
"""
import numpy as np
from osgeo import gdal
from skimage._shared.utils import safe_as_int


class IMAGE:
    def read_attribute(self, filename):
        dataset = gdal.Open(filename, gdal.GA_ReadOnly)

        # 查看波段数
        num_bands = dataset.RasterCount
        print("波段数为：{}\n".format(num_bands))

        # 查看行列数
        rows = dataset.RasterYSize
        cols = dataset.RasterXSize
        print('Image size is: {r} rows x {c} columns\n'.format(r=rows, c=cols))

        # 查看描述和元数据
        desc = dataset.GetDescription()  # GetDescription方法
        metadata = dataset.GetMetadata()  # GetMetadata方法
        print('Raster description: {desc}'.format(desc=desc))
        print('Raster metadata:')
        print(metadata)
        print('\n')

        # 查看打开这个影像的驱动
        driver = dataset.GetDriver()  # GetDriver方法
        print('Raster driver: {d}\n'.format(d=driver.ShortName))

        # 查看投影信息
        proj = dataset.GetProjection()
        print('Image projection:')
        print(proj + "\n")

        # 查看geo-transform
        gt = dataset.GetGeoTransform()
        print('Image geo-transform:{gt}\n'.format(gt=gt))

    def read_img(self, filename):
        dataset = gdal.Open(filename, gdal.GA_ReadOnly)

        num_bands = dataset.RasterCount  # 波段数
        im_height = dataset.RasterYSize  # 栅格矩阵的行数
        im_width = dataset.RasterXSize  # 栅格矩阵的列数

        im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵，左上角像素的大地坐标和像素分辨率
        im_proj = dataset.GetProjection()  # 地图投影信息，字符串表示
        im_data = dataset.ReadAsArray(0, 0, im_width, im_height)

        del dataset  # 关闭对象dataset，释放内存
        return im_proj, im_geotrans, im_data, im_width, im_height, num_bands

    # 遥感影像的存储
    # 写GeoTiff文件
    def write_img(self, filename, im_proj, im_geotrans, im_data):
        # 判断栅格数据的数据类型
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        # 判读数组维数
        if len(im_data.shape) == 3:
            # 注意数据的存储波段顺序：im_bands, im_height, im_width
            im_bands, im_height, im_width = im_data.shape
        else:
            im_bands, (im_height, im_width) = 1, im_data.shape

        # 创建文件时 driver = gdal.GetDriverByName("GTiff")，数据类型必须要指定，因为要计算需要多大内存空间。
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影

        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

        del dataset


if __name__ == "__main__":

    run = IMAGE()

    # 查看图片属性
    # run.read_attribute('AQ600_data/AQ600_0166_450nm.tif')

    # 第一步
    proj, geotrans, data1, row1, column1, num_bands = run.read_img('AQ600_data/AQ600_0166_450nm.tif')
    _, _, data2, row2, column2, _ = run.read_img('AQ600_data/AQ600_0166_555nm.tif')
    _, _, data3, row3, column3, _ = run.read_img('AQ600_data/AQ600_0166_660nm.tif')
    _, _, data4, row4, column4, _ = run.read_img('AQ600_data/AQ600_0166_720nm.tif')
    _, _, data5, row5, column5, _ = run.read_img('AQ600_data/AQ600_0166_840nm.tif')

    # 第二步:将上述读取的3个波段放到一个数组中
    data = np.array((data1, data2, data3, data4, data5), dtype=data1.dtype)  # 按序将3个波段像元值放入
    # 这个是应对要进行波段组合的图像原本就是多波段的特点，因此我们可以用data[0,::],data[1,::],data[2,::]等表示，前面的0，1，2是图像波段号的索引
    # data = np.array((data1[0,::],data1[1,::],data2), dtype=data1.dtype)

    # 第三步
    run.write_img(r'AQ600_data/AQ600_0166_5boduan.tif', proj, geotrans, data)  # 写为3波段数据,假彩色，nir,red,green
