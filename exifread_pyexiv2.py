"""
作者：didi
日期：2023年03月22日
"""
from pyexiv2 import Image


if __name__ == '__main__':
    device = 'M3M'
    file = 'DJI_20230308100235_0002_MS_G.TIF'
    fileName = f'{device}_data/{file}'
    save_file_xmp = f'{device}_data/out/pyexiv2_xmp_{file.split(".")[0]}.txt'
    save_file_exif = f'{device}_data/out/pyexiv2_exif_{file.split(".")[0]}.txt'

    img = Image(fileName)

    xmp = img.read_xmp()
    with open(save_file_xmp, 'w') as f:
        for key in xmp.keys():
            f.write(f'Key: {key}, Value: {xmp[key]}\n')

    exif = img.read_exif()
    with open(save_file_exif, 'w') as f:
        for key in exif.keys():
            f.write(f'Key: {key}, Value: {exif[key]}\n')


