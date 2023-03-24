"""
作者：didi
日期：2023年03月13日
参考链接：
    https://blog.csdn.net/u010329292/article/details/128343521?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-3-128343521-blog-109674479.pc_relevant_landingrelevant&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-3-128343521-blog-109674479.pc_relevant_landingrelevant&utm_relevant_index=6
"""

from osgeo import gdal


def read_attribute(dataset):

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


if __name__ == "__main__":

    filepath = r'AQ600_data/AQ600_0166_450nm.tif'
    raster = gdal.Open(filepath)
    # read_attribute(raster)

    band = raster.GetRasterBand(1)
    type(band)
    gdal.GetDataTypeName(band.DataType)

    # Compute statistics
    if band.GetMinimum() is None or band.GetMaximum() is None:
        band.ComputeStatistics(0)
        print("Statistics computed.")

    # Fetch metadata for the band
    band.GetMetadata()

    # Print only selected metadata:
    print("[ NO DATA VALUE ] = ", band.GetNoDataValue())  # none
    print("[ MIN ] = ", band.GetMinimum())
    print("[ MAX ] = ", band.GetMaximum())

