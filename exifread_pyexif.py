"""
作者：didi
日期：2023年03月22日
"""

import pyexif


if __name__ == '__main__':

    file = 'DJI_20230308100235_0002_MS_RE.TIF'
    device = 'M3M'
    fileName = f'{device}_data/{file}'
    save_file = f'{device}_data/out/pyexif_{file.split(".")[0]}.txt'

    img = pyexif.ExifEditor(fileName)
    tags = img.getDictTags()

    with open(save_file, 'w') as f:
        for key in tags.keys():
            f.write(f'Key: {key}, Value: {tags[key]}\n')
