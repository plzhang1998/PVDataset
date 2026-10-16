# -*- coding: utf-8 -*-
import glob
import os
import numpy as np
import pandas as pd
from osgeo import gdal
from tqdm import tqdm

gdal.UseExceptions()  # 抛出异常而不是返回错误码
gdal.PushErrorHandler('CPLQuietErrorHandler')  # 可选：静默非致命警告

def crop_image(image_file, sample_image_file,
               xoff, yoff, xsize, ysize,
               compress_image, scale, resample_mode,
               outsize_x, outsize_y, ot="Byte"):
    """
    使用 gdal.Translate API 替代 os.system, 大幅提升速度并支持错误捕获
    """
    # 构建选项列表
    options = []

    # 基础参数
    options.append('-srcwin')
    options.extend([str(xoff), str(yoff), str(xsize), str(ysize)])

    # 数据类型
    options.append('-ot')
    options.append(ot)

    # 重采样与输出大小 (仅当 scale != 1 时)
    if scale != 1:
        options.append('-r')
        options.append(resample_mode)
        options.append('-outsize')
        options.extend([str(outsize_x), str(outsize_y)])

    # 创建选项 (CO)
    co_opts = ['TILED=YES']
    if compress_image:
        co_opts.append('COMPRESS=DEFLATE')
    else:
        # 默认给一个轻量压缩，防止文件过大
        co_opts.append('COMPRESS=LZW')

    for co in co_opts:
        options.append('-co')
        options.append(co)

    try:
        # 如果输入是文件路径字符串，gdal.Open 会自动处理；如果是 Dataset 对象也可以
        # 这里为了兼容您的原逻辑，传入路径字符串
        gdal.Translate(sample_image_file, image_file, options=options)
        return True
    except Exception as e:
        print(f"❌ 裁剪失败: {sample_image_file}, 错误: {e}")
        return False

def make_sample(input_dir, output_dir, 
                sample_width, sample_height, sample_stride, scale,
                crop_all, none_value, make_csv, csv_mode, compress_image, data_type,
                name1, name2, name3, name4):
    """
    将一个目录内，以name1, name2, name3, name4为名的文件夹内的以data_type为类型的文件按照一定大小裁剪至output_dir
    input_dir, 输入路径
    output_dir, 输出路径
    sample_width, 裁切大小x
    sample_height, 裁切大小y
    sample_stride, 裁剪的步长
    scale, 将即将要裁剪的切片按照一定比例重采样至新的大小
    crop_all, 是否裁剪所有影像，因为有的部分会全是无效值
    none_value, 没有数据的地方的值
    make_csv, 是否制作csv记录影像信息
    csv_mode, w	打开一个文件只用于写入。a 打开一个文件用于追加。a+	打开一个文件用于读写。
    compress_image, 压缩文件
    data_type, 文件的后缀 tiff tif
    name1, name2, name3, name4
    """
    outsize_x = sample_width
    outsize_y = sample_height
    sample_width = int(scale * sample_width)
    sample_height = int(scale * sample_height)
    sample_stride = int(scale * sample_stride)

    if name1 != '':
        input_name1_dir = os.path.join(input_dir, name1)
        output_name1_dir = os.path.join(output_dir, name1)
        if not os.path.exists(output_name1_dir):
            os.makedirs(output_name1_dir)
    else:
        print('name1 is None')
        return -1
    if name2 != '':
        input_name2_dir = os.path.join(input_dir, name2)
        output_name2_dir = os.path.join(output_dir, name2)
        if not os.path.exists(output_name2_dir):
            os.makedirs(output_name2_dir)
    if name3 != '':
        input_name3_dir = os.path.join(input_dir, name3)
        output_name3_dir = os.path.join(output_dir, name3)
        if not os.path.exists(output_name3_dir):
            os.makedirs(output_name3_dir)
    if name4 != '':
        input_name4_dir = os.path.join(input_dir, name4)
        output_name4_dir = os.path.join(output_dir, name4)
        if not os.path.exists(output_name4_dir):
            os.makedirs(output_name4_dir)

    if make_csv:
        info_file = open(output_dir + '/indices.csv', csv_mode)
        info_file.write('name,height,width,positive_ratio,mean_value\n')

    image_files = glob.glob(input_name1_dir + '/*' + data_type)
    total_images = len(image_files)
    
    # 先计算总样本数
    total_samples = 0
    sample_info_list = []
    for idx, image_file in enumerate(image_files):
        ds_temp = gdal.Open(image_file)
        if ds_temp is None:
            continue
        xoffs_temp = list(range(0, ds_temp.RasterXSize - sample_width, sample_stride))
        xoffs_temp.append(ds_temp.RasterXSize - sample_width)
        yoffs_temp = list(range(0, ds_temp.RasterYSize - sample_height, sample_stride))
        yoffs_temp.append(ds_temp.RasterYSize - sample_height)
        if ds_temp.RasterXSize >= sample_width and ds_temp.RasterYSize >= sample_height:
            total_samples += len(xoffs_temp) * len(yoffs_temp)
            sample_info_list.append((image_file, len(xoffs_temp) * len(yoffs_temp)))
        del ds_temp
    
    global_sample_idx = 0
    
    for image_file in image_files:
        image_name = os.path.basename(image_file)
        if name2 != '':
            name2_file = input_name2_dir + "/" + image_name
        if name3 != '':
            name3_file = input_name3_dir + "/" + image_name
        if name4 != '':
            name4_file = input_name4_dir + "/" + image_name

        ds = gdal.Open(image_file)
        xoffs = list(range(0, ds.RasterXSize - sample_width, sample_stride))
        xoffs.append(ds.RasterXSize - sample_width)
        yoffs = list(range(0, ds.RasterYSize - sample_height, sample_stride))
        yoffs.append(ds.RasterYSize - sample_height)

        if ds.RasterXSize < sample_width or ds.RasterYSize < sample_height:
            continue
        
        for i, xoff in enumerate(xoffs):
            for j, yoff in enumerate(yoffs):
                xsize = min(sample_width, ds.RasterXSize - xoff)
                ysize = min(sample_height, ds.RasterYSize - yoff)
                sample_image_name = os.path.splitext(image_name)[0] + '_' + str(
                    i).zfill(4) + '_' + str(j).zfill(4) + data_type
                sample_image_file = output_name1_dir + "/" + sample_image_name

                image_data = ds.ReadAsArray(xoff, yoff, xsize=xsize, ysize=ysize)
                empty_num = np.sum(image_data == none_value)

                # image
                if crop_all or empty_num < image_data.size * 0.6:
                    global_sample_idx += 1
                    progress_percent = (global_sample_idx / total_samples) * 100
                    print(f'\r[{global_sample_idx}/{total_samples}] make sample: {sample_image_name} ({progress_percent:.1f}%)', end='', flush=True)
                    
                    # name1
                    crop_image(image_file, sample_image_file,
                                xoff, yoff, xsize, ysize,
                                compress_image, scale, 'nearest',
                                outsize_x, outsize_y)
                    if make_csv:
                        mean_value = np.mean(np.reshape(image_data, (image_data.shape[0], -1)))
                    # name2
                    if name2 != '':
                        sample_name2_file = output_name2_dir + "/" + sample_image_name
                        crop_image(name2_file, sample_name2_file,
                                    xoff, yoff, xsize, ysize,
                                    compress_image, scale, 'nearest',
                                    outsize_x, outsize_y)
                        if make_csv:
                            label_ds = gdal.Open(sample_name2_file)
                            labeldata = label_ds.ReadAsArray()
                            positive_ratio = np.sum(labeldata[labeldata > 0]) / float(outsize_x * outsize_y)
                            info_file.write(
                                '%s,%d,%d,%.3f,%d\n' % (sample_image_name, ysize, xsize, positive_ratio, mean_value))
                    # name3
                    if name3 != '':
                        sample_name3_file = output_name3_dir + "/" + sample_image_name
                        crop_image(name3_file, sample_name3_file,
                                    xoff, yoff, xsize, ysize,
                                    False, scale, 'nearest',
                                    outsize_x, outsize_y)
                    # name4
                    if name4 != '':
                        sample_name4_file = output_name4_dir + "/" + sample_image_name
                        crop_image(name4_file, sample_name4_file,
                                    xoff, yoff, xsize, ysize,
                                    compress_image, scale, 'nearst',
                                    outsize_x, outsize_y)
    if make_csv:
        info_file.close()

def remove_aux(output_dir):
    for file in glob.glob(output_dir + '/*.aux.xml'):
        os.remove(file)

if __name__ == '__main__':
    import argparse
    from pathlib import Path
    p = argparse.ArgumentParser(description='Crop images using the original window rule (scale=1).')
    p.add_argument('--input-dir', required=True, help='Station directory containing images/')
    p.add_argument('--output-dir', required=True, help='New station crop directory')
    p.add_argument('--patch-size', type=int, default=512)
    p.add_argument('--stride', type=int, default=448)
    p.add_argument('--extension', default='.tif', help='Exact input extension, e.g. .tif or .JPG')
    p.add_argument('--crop-all', action='store_true', help='Disable original >=60 percent zero-value exclusion')
    a = p.parse_args()
    if a.patch_size <= 0 or a.stride <= 0:
        p.error('patch-size and stride must be positive')
    if not Path(a.input_dir, 'images').is_dir():
        p.error('input-dir must contain images/')
    if Path(a.output_dir).exists():
        p.error('output-dir must not exist; refusing to overwrite crops')
    make_sample(a.input_dir, a.output_dir, a.patch_size, a.patch_size, a.stride, 1,
                a.crop_all, 0, False, 'w', True, a.extension, 'images', '', '', '')
   
