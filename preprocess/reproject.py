"""
物方矢量投影到像方（convert_shp.py 的优化版，保留原版不覆盖）。
沿用原共线方程、旋转矩阵、函数名及末尾手动配置方式。
相机姿态角默认弧度；XYZ 必须同坐标系/同高程基准；f 与 u 单位一致。
支持 PolygonZ/MultiPolygonZ 及孔洞，逐顶点使用真实 Z，不用平均高程代替。
注意：这里只检查投影及影像范围，不进行三维遮挡检测或自动高程纠正。
不保存逐影像/逐实例详细报告；每个电站正常完成后只写一次 projection_summary.json。
依赖 numpy、numba、GDAL（osgeo）。
"""
import csv
import json
import os
import time
from math import cos, sin

import numba as nb
import numpy as np

try:
    from osgeo import gdal, ogr
except ImportError:
    # 允许在未安装 GDAL 的环境中单独测试投影公式。
    gdal = ogr = None


@nb.jit(nopython=True)
def Ground2Photo(f, X, Y, Z, XS, YS, ZS, a1, a2, a3, b1, b2, b3, c1, c2, c3):
    """地面点反算至相片点，沿用原公式；批量流程会先检查分母。"""
    x = -f * (a1 * (X - XS) + b1 * (Y - YS) + c1 * (Z - ZS)) / (a3 * (X - XS) + b3 * (Y - YS) + c3 * (Z - ZS))
    y = -f * (a2 * (X - XS) + b2 * (Y - YS) + c2 * (Z - ZS)) / (a3 * (X - XS) + b3 * (Y - YS) + c3 * (Z - ZS))
    return x, y


@nb.jit(nopython=True)
def Photo2Ground(x, y, f, Z, XS, YS, ZS, a1, a2, a3, b1, b2, b3, c1, c2, c3):
    """相片点正算至指定高程平面，保留供已有代码调用。"""
    XG = (Z - ZS) * (a1 * x + a2 * y - a3 * f) / (c1 * x + c2 * y - c3 * f) + XS
    YG = (Z - ZS) * (b1 * x + b2 * y - b3 * f) / (c1 * x + c2 * y - c3 * f) + YS
    return XG, YG


def readCSV(csv_path):
    """照片名 -> [Xs, Ys, Zs, Phi, Omega, Kappa]，兼容原字典接口。"""
    photo_dict = {}
    with open(csv_path, newline='', encoding='utf-8-sig') as csv_file:
        for row_number, row in enumerate(csv.reader(csv_file), 1):
            if not row or not any(v.strip() for v in row):
                continue
            if row[0].strip().casefold() in ('image name', 'image_name', 'filename'):
                continue
            if len(row) < 7:
                raise ValueError(f'POS 第 {row_number} 行不足 7 列')
            name = os.path.basename(row[0].strip().replace('\\', '/'))
            try:
                values = [float(v) for v in row[1:7]]
            except ValueError as exc:
                raise ValueError(f'POS 第 {row_number} 行数值错误：{row}') from exc
            if not np.isfinite(values).all():
                raise ValueError(f'POS 第 {row_number} 行存在非有限数值')
            if name in photo_dict:
                raise ValueError(f'POS 中照片名重复：{name}')
            photo_dict[name] = values
    return photo_dict


def rotation_matrix(phi, omega, kappa):
    # 与原代码 a1..c3 完全一致；列分别对应相片 x/y/z 轴。
    a1 = cos(phi) * cos(kappa) - sin(phi) * sin(omega) * sin(kappa)
    a2 = -cos(phi) * sin(kappa) - sin(phi) * sin(omega) * cos(kappa)
    a3 = -sin(phi) * cos(omega)
    b1 = cos(omega) * sin(kappa)
    b2 = cos(omega) * cos(kappa)
    b3 = -sin(omega)
    c1 = sin(phi) * cos(kappa) + cos(phi) * sin(omega) * sin(kappa)
    c2 = -sin(phi) * sin(kappa) + cos(phi) * sin(omega) * cos(kappa)
    c3 = cos(phi) * cos(omega)
    return np.array([[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]])


def project_points(points, pose, f, u, u0, v0, flip_y=True,
                   angle_unit='radian', min_depth=1e-6):
    """一次投影一张影像所需的所有顶点；返回像素坐标及顶点有效标记。

    沿用原相机约定：光轴指向相片坐标 -Z，前方点的原公式分母为负。
    min_depth 是物方长度单位，不是像素；跨越相机平面的实例整块跳过。
    """
    pose = np.asarray(pose, dtype=np.float64)
    angles = pose[3:6]
    if angle_unit == 'degree':
        angles = np.deg2rad(angles)
    elif angle_unit != 'radian':
        raise ValueError('angle_unit 只能是 radian 或 degree')
    camera_points = (np.asarray(points) - pose[:3]) @ rotation_matrix(*angles)
    depth = -camera_points[:, 2]
    valid = np.isfinite(camera_points).all(axis=1) & (depth > min_depth)
    pixel = np.full((len(points), 2), np.nan, dtype=np.float64)
    pixel[valid, 0] = u0 + f * camera_points[valid, 0] / (u * depth[valid])
    pixel[valid, 1] = v0 - f * camera_points[valid, 1] / (u * depth[valid])
    if flip_y:
        pixel[:, 1] *= -1
    valid &= np.isfinite(pixel).all(axis=1)
    return pixel, valid


def _require_gdal():
    if ogr is None:
        raise RuntimeError('当前 Python 环境没有 GDAL/osgeo。请使用原投影代码能正常运行的 GIS 环境。')
    gdal.UseExceptions()
    ogr.UseExceptions()


def _polygons(geometry):
    """遍历面及集合中的所有面，不把内环错误地当作独立实例。"""
    kind = ogr.GT_Flatten(geometry.GetGeometryType())
    if kind == ogr.wkbPolygon:
        yield geometry
    elif kind in (ogr.wkbMultiPolygon, ogr.wkbGeometryCollection):
        for i in range(geometry.GetGeometryCount()):
            yield from _polygons(geometry.GetGeometryRef(i))


def _pixel_boundary(im_width, im_height, flip_y):
    ymin, ymax = (-im_height, 0) if flip_y else (0, im_height)
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for x, y in ((0, ymin), (im_width, ymin), (im_width, ymax), (0, ymax), (0, ymin)):
        ring.AddPoint_2D(x, y)
    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometry(ring)
    return polygon


def _image_geometry(parts, pixels):
    multi = ogr.Geometry(ogr.wkbMultiPolygon)
    for rings in parts:
        polygon = ogr.Geometry(ogr.wkbPolygon)
        for start, end in rings:
            ring = ogr.Geometry(ogr.wkbLinearRing)
            for x, y in pixels[start:end]:
                ring.AddPoint_2D(float(x), float(y))
            ring.CloseRings()
            polygon.AddGeometry(ring)
        multi.AddGeometry(polygon)
    return multi


def GroundShp2PhotoShp(ground_shp_path, output_path, photo_dict, image_folder_path,
                       im_width, im_height, f, u, u0, v0, output_boundary=True,
                       flip_y=True, clip_to_image=True, angle_unit='radian',
                       min_depth=1e-6, check_image_size=True, max_images=None):
    """每幅影像输出一份 SHP；不覆盖现有文件，建议用新目录比较结果。

    新增可选参数放在原参数后，不影响原有位置参数调用。
    不做面积、长宽比、包含关系过滤；不做未验证的遮挡删除。
    返回简洁计数汇总，不再返回逐影像详情。统计内存不随已处理影像数增长。
    """
    _require_gdal()
    if not np.isfinite([im_width, im_height, f, u, u0, v0, min_depth]).all():
        raise ValueError('相机参数不能包含非有限数值')
    if min(im_width, im_height, f, u, min_depth) <= 0:
        raise ValueError('影像尺寸、f、u、min_depth 必须大于 0')
    if angle_unit not in ('radian', 'degree'):
        raise ValueError('angle_unit 只能是 radian 或 degree')
    if max_images is not None and (not isinstance(max_images, int) or max_images < 1):
        raise ValueError('max_images 必须为正整数或 None')
    lookup = {}
    for name, values in photo_dict.items():
        key = name.casefold()
        if key in lookup:
            raise ValueError(f'POS 存在大小写冲突：{name}')
        values = np.asarray(values[:6], dtype=float)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError(f'POS 参数不完整：{name}')
        lookup[key] = values
    folder = sorted(name for name in os.listdir(image_folder_path)
                    if os.path.splitext(name)[1].lower() in ('.jpg', '.jpeg', '.tif', '.tiff', '.png'))
    missing_pose_count = sum(name.casefold() not in lookup for name in folder)
    folder = [name for name in folder if name.casefold() in lookup]
    if max_images is not None:
        folder = folder[:max_images]
    if not folder:
        raise ValueError('影像文件名与 POS 没有匹配项，请检查文件名及扩展名')
    stems = [os.path.splitext(name)[0].casefold() for name in folder]
    if len(set(stems)) != len(stems):
        raise ValueError('存在同名不同扩展名影像，输出 SHP 名称会冲突')
    os.makedirs(output_path, exist_ok=True)
    summary_path = os.path.join(output_path, 'projection_summary.json')
    # SHP 是多文件格式：任何旧输出都不覆盖，以免误改人工修订结果。
    for name in folder:
        base = os.path.join(output_path, os.path.splitext(name)[0])
        if any(os.path.exists(base + ext) for ext in ('.shp', '.dbf', '.shx')):
            raise FileExistsError(f'输出已存在：{base}，请更换 output_path')
    if os.path.exists(summary_path):
        raise FileExistsError(f'统计已存在：{summary_path}，请更换 output_path')

    ground_shp = ogr.Open(ground_shp_path, 0)
    if ground_shp is None:
        raise FileNotFoundError(ground_shp_path)
    ground_layer = ground_shp.GetLayer(0)
    definition = ground_layer.GetLayerDefn()
    fields = []
    for i in range(definition.GetFieldCount()):
        source_field = definition.GetFieldDefn(i)
        field = ogr.FieldDefn(source_field.GetName(), source_field.GetType())
        field.SetWidth(source_field.GetWidth())
        field.SetPrecision(source_field.GetPrecision())
        fields.append(field)
    has_id = any(field.GetName().casefold() == 'id' for field in fields)
    features, vertices = [], []
    skipped_source_counts = {'empty_geometry': 0, 'unsupported_or_nonfinite_geometry': 0}
    # 只读一次 SHP；缓存每个实例的所有面/环及顶点，避免逐影像反复读取。
    for feature in ground_layer:
        geom = feature.GetGeometryRef()
        fid = feature.GetFID()
        if geom is None or geom.IsEmpty():
            skipped_source_counts['empty_geometry'] += 1
            continue
        if geom.GetCoordinateDimension() < 3:
            raise ValueError(f'FID={fid} 没有 Z 坐标，不能默认为高程 0')
        feature_vertices, parts = [], []
        for polygon in _polygons(geom):
            rings = []
            for i in range(polygon.GetGeometryCount()):
                ring = polygon.GetGeometryRef(i)
                points = [ring.GetPoint(j)[:3] for j in range(ring.GetPointCount())]
                if len(points) < 4 or points[0] != points[-1]:
                    raise ValueError(f'FID={fid} 的环不闭合或顶点不足')
                start = len(vertices) + len(feature_vertices)
                feature_vertices.extend(points)
                rings.append((start, start + len(points)))
            if rings:
                parts.append(rings)
        if not feature_vertices or not np.isfinite(feature_vertices).all():
            skipped_source_counts['unsupported_or_nonfinite_geometry'] += 1
            continue
        start = len(vertices)
        vertices.extend(feature_vertices)
        values = []
        for i, field in enumerate(fields):
            value = feature.GetField(i)
            if value is not None and field.GetType() in (ogr.OFTDate, ogr.OFTTime, ogr.OFTDateTime):
                value = tuple(feature.GetFieldAsDateTime(i))
            values.append(value)
        features.append({'fid': fid, 'fields': values,
                         'parts': parts, 'start': start, 'end': len(vertices)})
    ground_shp = None
    if not features:
        raise ValueError('物方 SHP 没有可投影的三维面')
    vertices = np.asarray(vertices, dtype=np.float64)
    print(f'已缓存 {len(features)} 个实例、{len(vertices)} 个顶点；'
          f'高程范围 {vertices[:, 2].min():.3f}～{vertices[:, 2].max():.3f}', flush=True)
    print(f'待处理 {len(folder)} 张影像；缺少 POS 的影像 {missing_pose_count} 张', flush=True)
    boundary = _pixel_boundary(im_width, im_height, flip_y)
    xmin, xmax, ymin, ymax = boundary.GetEnvelope()
    driver = ogr.GetDriverByName('ESRI Shapefile')
    # 仅固定数量的计数器，不保存 issues/FID/逐影像记录。
    totals = {'written': 0, 'behind_or_near_camera': 0, 'outside': 0,
              'invalid_geometry': 0, 'clipped': 0}
    summary = {'images_processed': 0, 'missing_pose_images': missing_pose_count,
               'source_instances': len(features), 'source_skipped': skipped_source_counts,
               'projection_counts': totals}
    began = time.perf_counter()
    for number, file in enumerate(folder, 1):
        if check_image_size:
            image_ds = gdal.Open(os.path.join(image_folder_path, file))
            if image_ds is None:
                raise RuntimeError(f'无法打开影像：{file}')
            actual_size = (image_ds.RasterXSize, image_ds.RasterYSize)
            image_ds = None
            if actual_size != (im_width, im_height):
                raise ValueError(f'{file} 实际尺寸 {actual_size} 与配置不符；请同时检查内参')
        pose = lookup[file.casefold()]
        pixels, valid = project_points(vertices, pose, f, u, u0, v0,
                                      flip_y, angle_unit, min_depth)
        filename = os.path.splitext(file)[0]
        path = os.path.join(output_path, filename + '.shp')
        info = dict.fromkeys(totals, 0)
        ds = driver.CreateDataSource(path)
        try:
            layer = ds.CreateLayer(filename, None, ogr.wkbPolygon, options=['ENCODING=UTF-8'])
            for field in fields:
                layer.CreateField(field)
            if not has_id:
                layer.CreateField(ogr.FieldDefn('ID', ogr.OFTInteger64))
            for item in features:
                start, end = item['start'], item['end']
                if not valid[start:end].all():
                    info['behind_or_near_camera'] += 1
                    continue
                points = pixels[start:end]
                lo, hi = points.min(axis=0), points.max(axis=0)
                if hi[0] <= xmin or lo[0] >= xmax or hi[1] <= ymin or lo[1] >= ymax:
                    info['outside'] += 1
                    continue
                projected = _image_geometry(item['parts'], pixels)
                if not projected.IsValid():
                    info['invalid_geometry'] += 1
                    continue
                # 真正的多边形求交，不是逐顶点 clamp，不制造边界上的畸形面。
                if clip_to_image and (lo[0] < xmin or hi[0] > xmax or lo[1] < ymin or hi[1] > ymax):
                    projected = projected.Intersection(boundary)
                    info['clipped'] += 1
                output_geometry = ogr.Geometry(ogr.wkbMultiPolygon)
                for polygon in _polygons(projected):
                    if not polygon.IsEmpty() and polygon.GetArea() > 0:
                        output_geometry.AddGeometry(polygon)
                if output_geometry.IsEmpty():
                    info['outside'] += 1
                    continue
                out_feature = ogr.Feature(layer.GetLayerDefn())
                out_feature.SetGeometry(output_geometry)
                for i, value in enumerate(item['fields']):
                    if value is not None:
                        if fields[i].GetType() in (ogr.OFTDate, ogr.OFTTime, ogr.OFTDateTime):
                            out_feature.SetField(i, *value)
                        else:
                            out_feature.SetField(i, value)
                if not has_id:
                    out_feature.SetField('ID', item['fid'])
                layer.CreateFeature(out_feature)
                out_feature = None
                info['written'] += 1
            layer.SyncToDisk()
            layer = None
        finally:
            ds = None
        if output_boundary:
            # 独立子目录，绝不混进光伏实例；边界也使用同样的 y 方向。
            boundary_dir = os.path.join(output_path, '_image_boundary')
            os.makedirs(boundary_dir, exist_ok=True)
            boundary_path = os.path.join(boundary_dir, filename + '.shp')
            if os.path.exists(boundary_path):
                raise FileExistsError(boundary_path)
            bds = driver.CreateDataSource(boundary_path)
            blayer = bds.CreateLayer(filename, None, ogr.wkbPolygon)
            bf = ogr.Feature(blayer.GetLayerDefn())
            bf.SetGeometry(boundary)
            blayer.CreateFeature(bf)
            bf = blayer = bds = None
        for key in totals:
            totals[key] += info[key]
        summary['images_processed'] += 1
        print(f'[{number}/{len(folder)}] {filename}：输出 {info["written"]}，'
              f'范围外 {info["outside"]}，相机后方/近零分母 {info["behind_or_near_camera"]}，'
              f'无效几何 {info["invalid_geometry"]}，裁剪 {info["clipped"]}', flush=True)
    summary['elapsed_seconds'] = round(time.perf_counter() - began, 2)
    # 全部影像处理完成后只写一次，不进行逐图追加或历史报告重写。
    with open(summary_path, 'x', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(f'完成，用时 {summary["elapsed_seconds"]:.1f}s；简单统计：{summary_path}', flush=True)
    print(f'异常汇总：缺少 POS {missing_pose_count} 张，'
          f'源几何跳过 {sum(skipped_source_counts.values())} 个，'
          f'投影无效几何 {totals["invalid_geometry"]} 次。', flush=True)
    return summary


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Project 3D polygons using author-provided camera parameters.')
    p.add_argument('--ground-shp', required=True)
    p.add_argument('--pose-csv', required=True)
    p.add_argument('--image-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--width', type=int, required=True)
    p.add_argument('--height', type=int, required=True)
    p.add_argument('--focal', type=float, required=True, help='f, same unit as pixel-size')
    p.add_argument('--pixel-size', type=float, required=True, help='u; use 1 for pixel focal length')
    p.add_argument('--cx', type=float, required=True)
    p.add_argument('--cy', type=float, required=True)
    p.add_argument('--angle-unit', choices=['radian', 'degree'], default='radian')
    p.add_argument('--flip-y', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--max-images', type=int)
    a = p.parse_args()
    if min(a.width, a.height, a.focal, a.pixel_size) <= 0:
        p.error('Image dimensions, focal length and pixel size must be positive')
    _require_gdal()
    GroundShp2PhotoShp(a.ground_shp, a.output_dir, readCSV(a.pose_csv), a.image_dir,
                      a.width, a.height, a.focal, a.pixel_size, a.cx, a.cy, False,
                      flip_y=a.flip_y, clip_to_image=True, angle_unit=a.angle_unit,
                      min_depth=1e-6, check_image_size=True, max_images=a.max_images)
