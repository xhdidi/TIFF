"""
作者：didi
日期：2023年03月21日
"""
import exifread
import json
import urllib.request


def getLatOrLng(refKey, tudeKey):
    ''' 获取经纬度 '''
    if refKey not in tags:
        return None
    ref = tags[refKey].printable
    LatOrLng = tags[tudeKey].printable[1:-1].replace(" ", "").replace("/", ",").split(",")
    print(LatOrLng)
    LatOrLng = float(LatOrLng[0]) + float(LatOrLng[1]) / 60 + float(LatOrLng[2]) / 3600
    if refKey == 'GPS GPSLatitudeRef' and tags[refKey].printable != "N":
        LatOrLng = LatOrLng * (-1)
    if refKey == 'GPS GPSLongitudeRef' and tags[refKey].printable != "E":
        LatOrLng = LatOrLng * (-1)
    return LatOrLng


# 调用百度地图API通过经纬度获取位置
def getlocation(lat, lng):
    ''' exif里面的经纬度是度分秒形式传入的，需要转换为小数形式，然后根据传入的经纬度,调用百度接口去查询详细地址 '''
    url = "http://api.map.baidu.com/reverse_geocoding/v3/?ak=百度API的AK信息，需要自己去申请&output=json&coordtype=wgs84ll&language_auto=1&extensions_town=true&location=" + lat + "," + lng
    req = urllib.request.urlopen(url)
    res = req.read().decode("utf-8")
    print(res)
    str1 = json.loads(res)
    jsonResult = str1.get('result')
    formatted_address = jsonResult.get('formatted_address')
    return formatted_address


if __name__ == '__main__':

    device = 'M3M'  # M3M or AQ600
    filename = 'DJI_20230308100235_0002_MS_RE.TIF'
    save_file = f'{device}_data/out/exifread_{filename.split(".")[0]}.txt'
    file = f'{device}_data/{filename}'
    f = open(file, 'rb')

    tags = exifread.process_file(f)
    with open(save_file, 'w') as f:
        for tag in tags.keys():
            print("Key: {0}, value {1}".format(tag, tags[tag]))
            f.write(f"Key: {tag}, value {tags[tag]} \n")

    lat = getLatOrLng('GPS GPSLatitudeRef', 'GPS GPSLatitude')    # 纬度
    lng = getLatOrLng('GPS GPSLongitudeRef', 'GPS GPSLongitude')  # 经度
    print('纬度:{0} 经度：{1}'.format(lat, lng))


