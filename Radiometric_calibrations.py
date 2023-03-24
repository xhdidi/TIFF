"""
作者：didi
日期：2023年03月22日
参照DJI P4M图像处理指南：https://dl.djicdn.com/downloads/p4-multispectral/20200717/P4_Multispectral_Image_Processing_Guide_CHS.pdf
"""
import pandas as pd
from pyexiv2 import Image
from osgeo import gdal
import math
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import PIL
import earthpy.spatial as es
from mpl_toolkits.axes_grid1 import make_axes_locatable


class TIF_Image():
    def __init__(self, filename):
        self.filename = filename
        self.pyexiv2 = Image(self.filename)  # <class 'pyexiv2.core.Image'>
        self.xmp = self.pyexiv2.read_xmp()
        self.exif = self.pyexiv2.read_exif()

        self.W, self.L = int(self.exif['Exif.Image.ImageWidth']), int(self.exif['Exif.Image.ImageLength'])

        t_raster = gdal.Open(filename)
        self.image = t_raster.ReadAsArray()  # <class 'numpy.ndarray'>

    def physical_position_calibration(self):
        '''相机物理位置误差校准'''
        pos_x = int(np.around(eval(self.xmp['Xmp.drone-dji.RelativeOpticalCenterX']), decimals=0, out=None))
        pos_y = int(np.around(eval(self.xmp['Xmp.drone-dji.RelativeOpticalCenterY']), decimals=0, out=None))
        # pos_x<0, pos_y>0
        image_pad = np.pad(self.image, ((0, pos_y), (-pos_x, 0)), 'constant', constant_values=(0, 0))
        self.image = image_pad[pos_y:self.L+pos_y, :self.W]

    def exposure_time_calibration_handcraft(self):
        '''相机曝光时间误差校准：平滑+特征点对齐，在此只做高斯平滑，手动滤波'''
        # TODO: 特征点对齐
        kernel_size = 3
        sigma = 1.3
        pad = kernel_size // 2

        out = np.zeros((self.L+pad*2, self.W+pad*2), dtype=np.float)
        out[pad: pad+self.L, pad: pad+self.W] = self.image.copy().astype(np.float)
        tmp = out.copy()

        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float)
        for x in range(-pad, -pad + kernel_size):
            for y in range(-pad, -pad + kernel_size):
                kernel[y + pad, x + pad] = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
        kernel /= (2 * np.pi * sigma * sigma)
        kernel /= kernel.sum()

        for y in range(self.L):
            for x in range(self.W):
                out[pad + y, pad + x] = np.sum(kernel * tmp[y: y + kernel_size, x: x + kernel_size])
        self.image = out[pad: pad + self.L, pad: pad + self.W].astype(np.uint8)

    def exposure_time_calibration_cv(self):
        '''相机曝光时间误差校准：平滑+特征点对齐，在此只做高斯平滑，使用cv2.GaussianBlur()'''
        # TODO: 特征点对齐
        self.image = cv2.GaussianBlur(self.image, (3,3), 0)

    def cam_correction(self):
        # 暗角补偿
        center_x = eval(self.xmp['Xmp.drone-dji.CalibratedOpticalCenterX'])
        center_y = eval(self.xmp['Xmp.drone-dji.CalibratedOpticalCenterY'])
        k0, k1, k2, k3, k4, k5 = self.xmp['Xmp.drone-dji.VignettingData'].split(', ')
        k0, k1, k2, k3, k4, k5 = eval(k0), eval(k1), eval(k2), eval(k3), eval(k4), eval(k5)

        max_distance = math.sqrt(
            max((vertex[0] - center_x) ** 2 + (vertex[1] - center_y) ** 2 for vertex in
                [[0, 0], [0, W], [L, 0], [L, W]]))
        for x in range(0, L):
            for y in range(0, W):
                distance = math.sqrt(pow(x - center_x, 2) + pow(y - center_y, 2))
                r = distance / max_distance
                k = k5 * r**6 + k4 * r**5 + k3 * r**4 + k2 * r**3 + k1 * r**2 + k0 * r + 1.0
                self.image[x, y] *= k

        # 畸变校准
        fx, fy, cx, cy, k1, k2, p1, p2, p3 = self.xmp['Xmp.drone-dji.DewarpData'].split(';')[-1].split(',')

        distcoeffs = np.float32([eval(k1), eval(k2), eval(p1), eval(p2), eval(p3)])
        cam_matrix = np.zeros((3, 3))
        cam_matrix[0, 0] = eval(fx)
        cam_matrix[1, 1] = eval(fy)
        cam_matrix[0, 2] = eval(cx) + center_x
        cam_matrix[1, 2] = eval(cy) + center_y
        cam_matrix[2, 2] = 1

        self.image = cv2.undistort(self.image, cam_matrix, distcoeffs)


    def cam(self):
        bit = eval(self.exif['Exif.Image.BitsPerSample'])
        self.image /= pow(2.0, bit)
        # black_current = eval(self.xmp['Xmp.Camera.BlackCurrent']) / 3200
        # self.image -= black_current

        self.cam_correction()

        pcam = eval(self.xmp['Xmp.drone-dji.SensorGainAdjustment'])
        gain = eval(self.xmp['Xmp.drone-dji.SensorGain'])
        etime = eval(self.xmp['Xmp.drone-dji.ExposureTime'])
        denominator = eval(self.xmp['Xmp.drone-dji.Irradiance'])

        self.image *= pcam/(gain*etime/1e6)/denominator

    def calibration(self):
        '''总校准函数'''
        self.physical_position_calibration()  # 物理位置误差校准
        self.exposure_time_calibration_cv()   # 曝光时间误差校准

        self.cam()


def colorbar(mapobj, size="3%", pad=0.09):
    try:
        ax = mapobj.axes
    except AttributeError:
        raise AttributeError("The colorbar function requires a matplotlib axis object. "
                             "You have provided a {}.".format(type(mapobj)))
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    return fig.colorbar(mapobj, cax=cax)


def _plot_image(arr_im, cmap="Greys_r", title=None, title_set=None, extent=None, cbar=True, scale=False,
                vmin=None, vmax=None, ax=None, alpha=1, norm=None, colorbar_label_set=None, label_size=None, cbar_len=None):
    """
    Create a matplotlib figure with an image axis and associated extent.

    Parameters
    ----------
    arr_im : numpy array
        An n-dimensional numpy array to plot.
    cmap : str (default = "Greys_r")
        Colormap name for plots.
    title : str or list (optional)
        Title of one band or list of titles with one title per band.
    extent : tuple (optional)
        Bounding box that the data will fill: (minx, miny, maxx, maxy).
    cbar : Boolean (default = True)
        Turn off colorbar if needed.
    scale : Boolean (Default = False)
        Turn off bytescale scaling if needed.
    vmin : Int (Optional)
        Specify the vmin to scale imshow() plots.
    vmax : Int (Optional)
        Specify the vmax to scale imshow() plots.
    ax : Matplotlib axes object (Optional)
        Matplotlib axis object to plot image.
    alpha : float (optional)
        The alpha value for the plot. This will help adjust the transparency of
        the plot to the desired level.
    norm : matplotlib Normalize object (Optional)
        The normalized boundaries for custom values coloring. NOTE: For this
        argument to work, the scale argument MUST be set to false. Otherwise,
        the values will be scaled from 0-255.

    Returns
    ----------
    ax : matplotlib.axes object
        The axes object(s) associated with the plot.
    """

    if scale:
        arr_im = es.bytescale(arr_im)

    im = ax.imshow(
        arr_im,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        alpha=alpha,
        norm=norm,
    )
    if title:
        if title_set:
            ax.set_title(label=title, fontsize=title_set[0], fontweight=title_set[1])
        else:
            ax.set(title=title)
    if cbar:
        # 设置色带的刻度长度和label大小
        if colorbar_label_set:
            cbar = colorbar(im)
            cbar.ax.tick_params(length=cbar_len, labelsize=label_size)
        else:
            colorbar(im)
    ax.set(xticks=[], yticks=[])

    return ax


def plot_bands(arr, cmap="Greys_r", figsize=(12, 12), cols=3, title=None, title_set=None, extent=None, cbar=True,
               scale=False, vmin=None, vmax=None, ax=None, alpha=1, norm=None, save_or_not=False, save_path=None,
               dpi_out=None, bbox_inches_out=None, pad_inches_out=None, text_or_not=None, text_set=None,
               colorbar_label_set=None, label_size=None, cbar_len=None):
    """Plot each band in a numpy array in its own axis.

    Assumes band order (band, row, col).

    Parameters
    ----------
    arr : numpy array
        An n-dimensional numpy array to plot.
    cmap : str (default = "Greys_r")
        Colormap name for plots.
    figsize : tuple (default = (12, 12))
        Figure size in inches.
    cols : int (default = 3)
        Number of columns for plot grid.
    title : str or list (optional)
        Title of one band or list of titles with one title per band.
    extent : tuple (optional)
        Bounding box that the data will fill: (minx, miny, maxx, maxy).
    cbar : Boolean (default = True)
        Turn off colorbar if needed.
    scale : Boolean (Default = False)
        Turn off bytescale scaling if needed.
    vmin : Int (Optional)
        Specify the vmin to scale imshow() plots.
    vmax : Int (Optional)
        Specify the vmax to scale imshow() plots.
    alpha : float (optional)
        The alpha value for the plot. This will help adjust the transparency
        of the plot to the desired level.
    norm : matplotlib Normalize object (Optional)
        The normalized boundaries for custom values coloring. NOTE: For this
        argument to work, the scale argument MUST be set to false. Because
        of this, the function will automatically set scale to false,
        even if the user manually sets scale to true.

    Returns
    ----------
    ax or axs : matplotlib.axes._subplots.AxesSubplot object(s)
        The axes object(s) associated with the plot.

    Example
    -------
    .. plot::

        >>> import matplotlib.pyplot as plt
        >>> import earthpy.plot as ep
        >>> from earthpy.io import path_to_example
        >>> import rasterio as rio
        >>> titles = ['Red', 'Green', 'Blue']
        >>> with rio.open(path_to_example('rmnp-rgb.tif')) as src:
        ...     ep.plot_bands(src.read(),
        ...                   title=titles,
        ...                   figsize=(8, 3))
        array([<AxesSubplot:title={'center':'Red'}>...
    """
    show = False
    try:
        arr.ndim
    except AttributeError:
        raise AttributeError("Input arr should be a numpy array")
    if norm:
        scale = False
    if title:
        if isinstance(title, str):
            title = [title]

        # A 2-dim array should only be passed one title
        if arr.ndim == 2 and len(title) > 1:
            raise ValueError(
                "plot_bands expects one title for a single band array. You have provided more than one title."
            )
        # A 3 dim array should have the same number of titles as dims
        if arr.ndim > 2:
            if len(title) != arr.shape[0]:
                raise ValueError(
                    "plot_bands expects the number of plot titles to equal the number of array raster layers."
                )

    # If the array is 3 dimensional setup grid plotting
    if arr.ndim > 2 and arr.shape[0] > 1:

        # Calculate the total rows that will be required to plot each band
        plot_rows = int(np.ceil(arr.shape[0] / cols))
        total_layers = arr.shape[0]

        # Plot all bands
        fig, axs = plt.subplots(plot_rows, cols, figsize=figsize)
        axs_ravel = axs.ravel()
        for ax, i in zip(axs_ravel, range(total_layers)):
            band = i + 1

            arr_im = arr[i]

            if title:
                the_title = title[i]
            else:
                the_title = "Band {}".format(band)

            _plot_image(
                arr_im,
                cmap=cmap,
                cbar=cbar,
                scale=scale,
                vmin=vmin,
                vmax=vmax,
                extent=extent,
                title=the_title,
                title_set=title_set,
                ax=ax,
                alpha=alpha,
                norm=norm,
                colorbar_label_set=colorbar_label_set,
                label_size=label_size,
                cbar_len=cbar_len,
            )
        # This loop clears out the plots for axes which are empty
        # A matplotlib axis grid is always uniform with x cols and x rows
        # eg: an 8 band plot with 3 cols will always be 3 x 3
        for ax in axs_ravel[total_layers:]:
            ax.set_axis_off()
            ax.set(xticks=[], yticks=[])
        plt.tight_layout()
        if text_or_not:
            plt.text(text_set[0], text_set[1], text_set[2], ha='center', va='center', fontsize=text_set[3],
                     fontweight='bold', transform=ax.transAxes)
        # save figure
        if save_or_not:
            plt.savefig(save_path, dpi=dpi_out, bbox_inches=bbox_inches_out, pad_inches=pad_inches_out)
            print('save successfully')
        plt.show()
        return axs

    elif arr.ndim == 2 or arr.shape[0] == 1:
        # If it's a 2 dimensional array with a 3rd dimension
        arr = np.squeeze(arr)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            show = True

        if title:
            title = title[0]

        _plot_image(arr, cmap=cmap, scale=scale, cbar=cbar, vmin=vmin, vmax=vmax, extent=extent,
                    title=title, title_set=title_set, ax=ax, alpha=alpha, norm=norm,
                    colorbar_label_set=colorbar_label_set, label_size=label_size, cbar_len=cbar_len)

        if text_or_not:
            plt.text(text_set[0], text_set[1], text_set[2], ha='center', va='center', fontsize=text_set[3],
                     fontweight=text_set[4], transform=ax.transAxes)
        if show:
            # save figure
            if save_or_not:
                plt.savefig(save_path, dpi=dpi_out, bbox_inches=bbox_inches_out, pad_inches=pad_inches_out)
            plt.show()
        return ax


def show(image, savefile):
    min_val = np.nanmin(image)
    max_val = np.nanmax(image)
    plot_title = savefile.split('/')[-1].split('.')[0]
    plot_bands(image, title=plot_title, title_set=[25, 'bold'], cmap="seismic",  # color
               cols=3, figsize=(12, 12), extent=None, cbar=True, scale=False,
               vmin=min_val, vmax=max_val, ax=None, alpha=1, norm=None, save_or_not=True, save_path=savefile,
               dpi_out=300, bbox_inches_out="tight", pad_inches_out=0.1, text_or_not=True,
               text_set=[0.75, 0.95, "T(°C)", 20, 'bold'], colorbar_label_set=True, label_size=20, cbar_len=2)


if __name__ == '__main__':
    NIR_file = 'M3M_data/DJI_20230308100235_0002_MS_NIR.TIF'
    RED_file = 'M3M_data/DJI_20230308100235_0002_MS_R.TIF'
    RE_file = 'M3M_data/DJI_20230308100235_0002_MS_RE.TIF'
    G_file = 'M3M_data/DJI_20230308100235_0002_MS_G.TIF'

    img = Image(NIR_file)
    xmp = img.read_xmp()
    exif = img.read_exif()

    t_raster = gdal.Open(NIR_file)
    image_ori = t_raster.ReadAsArray()
    # cv2.imwrite('M3M_data/out/NIR_ori.png', image_ori)

    image_norm = image_ori.astype(np.float)
    bit = eval(exif['Exif.Image.BitsPerSample'])
    image_norm /= pow(2.0, bit)
    # black_current = eval(xmp['Xmp.Camera.BlackCurrent']) / 3200
    # image_norm -= black_current
    # cv2.imwrite('M3M_data/out/NIR_norm.png', image_norm)

    # 暗角补偿
    center_x = eval(xmp['Xmp.drone-dji.CalibratedOpticalCenterX'])
    center_y = eval(xmp['Xmp.drone-dji.CalibratedOpticalCenterY'])
    k0, k1, k2, k3, k4, k5 = xmp['Xmp.drone-dji.VignettingData'].split(', ')
    k0, k1, k2, k3, k4, k5 = eval(k0), eval(k1), eval(k2), eval(k3), eval(k4), eval(k5)
    W, L = int(exif['Exif.Image.ImageWidth']), int(exif['Exif.Image.ImageLength'])

    # vignetting = np.zeros_like(image_ori)
    image_vig = np.zeros_like(image_norm)
    max_distance = math.sqrt(
        max((vertex[0] - center_x) ** 2 + (vertex[1] - center_y) ** 2 for vertex in [[0, 0], [0, W], [L, 0], [L, W]]))
    for x in range(0, L):
        for y in range(0, W):
            distance = math.sqrt(pow(x - center_x, 2) + pow(y - center_y, 2))
            r = distance / max_distance
            k = k5 * pow(r, 6) + k4 * pow(r, 5) + k3 * pow(r, 4) + k2 * pow(r, 3) + k1 * pow(r, 2) + k0 * pow(r, 1) + 1.0
            # vignetting[x, y] = k
            image_vig[x, y] = k * image_norm[x, y]
    # cv2.imshow('Vignetting Image', image_vig)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite('M3M_data/out/NIR_vig.png', image_vig)

    # 畸变校准
    fx, fy, cx, cy, k1, k2, p1, p2, p3 = xmp['Xmp.drone-dji.DewarpData'].split(';')[-1].split(',')
    distcoeffs = np.float32([eval(k1), eval(k2), eval(p1), eval(p2), eval(p3)])
    cam_matrix = np.zeros((3, 3))
    cam_matrix[0, 0] = eval(fx)
    cam_matrix[1, 1] = eval(fy)
    cam_matrix[0, 2] = eval(cx) + center_x
    cam_matrix[1, 2] = eval(cy) + center_y
    cam_matrix[2, 2] = 1

    image_dewarp = cv2.undistort(image_vig, cam_matrix, distcoeffs)
    # cv2.imshow('Indistort Image', image_dewarp)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite('M3M_data/out/NIR_Indistort.png', image_dewarp)

    pcam = eval(xmp['Xmp.drone-dji.SensorGainAdjustment'])
    gain = eval(xmp['Xmp.drone-dji.SensorGain'])
    etime = eval(xmp['Xmp.drone-dji.ExposureTime'])
    denominator = eval(xmp['Xmp.drone-dji.Irradiance'])

    image_final = image_vig * pcam / (gain * etime / 1e6) / denominator
    # cv2.imshow('Final Image', image_final)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite('M3M_data/out/NIR_Final.png', image_final)

    # data_ori = pd.DataFrame(image_ori)
    # data_norm = pd.DataFrame(image_norm)
    # data_vig = pd.DataFrame(image_vig)
    # data_dewarp = pd.DataFrame(image_dewarp)
    # data_final = pd.DataFrame(image_final)
    # writer = pd.ExcelWriter('M3M_data/out/NIR_total.xlsx')
    # data_ori.to_excel(writer, 'ori', float_format='%.5f')
    # data_norm.to_excel(writer, 'norm', float_format='%.5f')
    # data_vig.to_excel(writer, 'vig', float_format='%.5f')
    # data_dewarp.to_excel(writer, 'dewarp', float_format='%.5f')
    # data_final.to_excel(writer, 'final', float_format='%.5f')
    # writer.save()

    show(image_norm, 'M3M_data/out/NIR_norm.png')
    show(image_vig, 'M3M_data/out/NIR_vig.png')
    show(image_dewarp, 'M3M_data/out/NIR_dewarp.png')
    show(image_final, 'M3M_data/out/NIR_final.png')

    # result = PIL.Image.new('F', (W*5+500, L+100))
    # result.paste(image_ori, (50, 50, 50 + W, 50 + L))
    # result.paste(image, (150 + W, 50, 150 + 2 * W, 50 + L))
    # result.paste(image_vig, (250 + 2 * W, 50, 250 + 3 * W, 50 + L))
    # result.paste(image_dewarp, (350 + 3 * W, 50, 350 + 4 * W, 50 + L))
    # result.paste(image_final, (450 + 4 * W, 50, 4500 + 5 * W, 50 + L))
    # result.save("M3M_data/out/NIR_total.jpg")
    # plt.imshow(result)
    # plt.show()

    # print(type(exif['Exif.Image.BitsPerSample']))

    # print(xmp['Xmp.Camera.VignettingPolynomial'])
    # k0, k1, k2, k3, k4, k5 = xmp['Xmp.Camera.VignettingPolynomial']
    # print(type(k0))
    #
    # print(xmp['Xmp.drone-dji.DewarpData'])
    # print(xmp['Xmp.drone-dji.DewarpData'].split(';')[1])
    # fx, fy, cx, cy, k1, k2, p1, p2, p3 = xmp['Xmp.drone-dji.DewarpData'].split(';')[-1].split(',')
    # print(type(fx))

    # image = cv2.GaussianBlur(image, (3, 3), 0)
    # cv2.imshow('Gaussian Image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # W, L = int(exif['Exif.Image.ImageWidth']), int(exif['Exif.Image.ImageLength'])
    # pos_x = int(np.around(eval(xmp['Xmp.drone-dji.RelativeOpticalCenterX']), decimals=0, out=None))
    # pos_y = int(np.around(eval(xmp['Xmp.drone-dji.RelativeOpticalCenterY']), decimals=0, out=None))
    # print(W, L, pos_x, pos_y, image.shape)
    # image = np.pad(image, ((0, pos_y), (-pos_x, 0)), 'constant', constant_values=(0, 0))
    # plt.imshow(image)
    # plt.show()
    # plt.imsave('M3M_data/out/pad_DJI_20230308100235_0002_MS_R.png', image)
    # image = image[pos_y:L+pos_y, :W]
    # plt.imshow(image)
    # plt.show()
    # plt.imsave('M3M_data/out/crop_DJI_20230308100235_0002_MS_R.png', image)


