# tiff工程
## 数据集来源
AQ600和DJI M3M文件夹分别存储AQ600设备和DJI M3M设备拍摄的原始数据，其中out文件夹下存储各程序输出
## py文件介绍
- main_1.py：多个单波段图像组合成为一个多波段文件
- main_2.py：多波段文件按波段分离
- main_3.py：利用gdal库打开tif文件并读取简单参数（波段数、投影信息等）
- visualize_1.py：单波段图像可视化
- exifread_exifread.py：利用exifread库读取EXIF元信息（DJI设备图像信息读取不全）
- exifread_pyexif.py：利用pyexif库读取EXIF元信息（有FOV数据，但需要安装 [exiftool](https://exiftool.org/install.html) ）
- exifread_pyexiv2.py：利用pyexiv2库读取EXIF元信息（缺少FOV数据）
- 其他tiff文件元数据查看方式：[EXIF信息查看器](https://exif.tuchong.com/)
- Radiometric_calibrations.py：参照 [P4M图像处理手册](Attachments/P4M图像处理指南.pdf) 进行辐射定标，缺少 EXIF 的 Black
Level 项，故不进行黑电平校正![NIR辐射定标流程](Attachments/NIR辐射定标.png)