"""SHP / 二值 mask 转 COCO

SHP：一个 Feature 对应一个实例，多部件不拆 ID；真正求交裁剪。
SHP 按无内环数据处理，统一输出浮点多边形顶点，iscrowd 为 0。
area/bbox 统一使用 COCO 栅格化掩码；保留多边形轮廓，不按面积阈值过滤。
默认过滤栅格 bbox 宽或高小于 5 像素的实例，宽、高恰好为 5 时保留。
二值 mask：每个连通域为一个实例，仅适合本身没有实例粘连的数据。
依赖 numpy、shapely>=2、pycocotools、tqdm、Pillow；SHP 另需 GDAL/osgeo，
二值 mask 连通域提取另需 opencv-python。不修改源数据或已有输出。
"""
import json
import os
import tempfile
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from shapely import force_2d, from_wkb, make_valid
from shapely.affinity import scale
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from shapely.validation import explain_validity
from tqdm import tqdm

try:
    from osgeo import gdal, ogr
except ImportError:
    gdal = ogr = None


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


def calculate_polygon_area(points):
    """保留原辅助函数接口，计算简单多边形的连续几何面积。"""
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return 0.0
    return float(abs(np.dot(points[:, 0], np.roll(points[:, 1], -1)) -
                     np.dot(points[:, 1], np.roll(points[:, 0], -1))) / 2)


def _polygons(geometry):
    if geometry.geom_type == 'Polygon':
        if not geometry.is_empty and geometry.area > 0:
            yield geometry
    elif geometry.geom_type in ('MultiPolygon', 'GeometryCollection'):
        for part in geometry.geoms:
            yield from _polygons(part)


def clip_polygon_to_image(points, img_width, img_height):
    """真实求交；返回 Polygon/MultiPolygon（不同于旧版单一顶点列表）。

    凹面裁剪可能形成多个部件，因此不能仅返回一个环，也不能逐点 clamp。
    """
    geometry = Polygon(points)
    if not geometry.is_valid:
        raise ValueError(explain_validity(geometry))
    return geometry.intersection(box(0, 0, img_width, img_height))


def _roi_to_full_rle(roi, x, y, width, height):
    """直接将局部前景游程偏移至整幅坐标，不逐实例扫描整幅大图。"""
    roi = np.asarray(roi, dtype=np.uint8)
    h, w = roi.shape
    if x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError('ROI 超出影像范围')
    # COCO 按列（Fortran order）编码，在每一列的首尾补背景获取前景游程。
    changes = np.diff(np.pad(roi.astype(np.int8), ((1, 1), (0, 0))), axis=0).ravel(order='F')
    first, last = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    if not len(first):
        counts = [height * width]
    else:
        starts = (x + first // (h + 1)) * height + y + first % (h + 1)
        ends = (x + last // (h + 1)) * height + y + last % (h + 1)
        # ROI 覆盖整列时，跨列相接的游程需要合并。
        split = starts[1:] != ends[:-1]
        starts = starts[np.r_[True, split]]
        ends = ends[np.r_[split, True]]
        counts, previous = [], 0
        for start, end in zip(starts, ends):
            counts.extend([int(start) - previous, int(end - start)])
            previous = int(end)
        if previous < height * width:
            counts.append(height * width - previous)
    return mask_utils.frPyObjects({'size': [height, width], 'counts': counts}, height, width)


def geometry_to_coco(geometry, width, height, flip_y=False,
                     min_bbox_size=5, invalid_policy='error'):
    """返回 (标注字段或 None, 质检信息)，不分配 image/annotation ID。

    统一输出 polygon；area/bbox 使用该 polygon 的 COCO 栅格化掩码统计。
    RLE 仅用于计数和外接框，不解码整幅像素数组，也不替换多边形顶点。
    """
    if invalid_policy not in ('error', 'repair', 'skip'):
        raise ValueError('invalid_policy 必须为 error、repair 或 skip')
    if width <= 0 or height <= 0 or not np.isfinite(min_bbox_size) or min_bbox_size < 0:
        raise ValueError('尺寸必须为正数，min_bbox_size 必须为非负有限值')
    info = {'clipped': False, 'repaired': False}
    if geometry is None or geometry.is_empty:
        return None, dict(info, reason='empty_geometry')
    geometry = force_2d(geometry)
    if flip_y:
        geometry = scale(geometry, xfact=1, yfact=-1, origin=(0, 0))
    if not geometry.is_valid:
        info['invalid_reason'] = explain_validity(geometry)
        if invalid_policy == 'error':
            raise ValueError('无效几何：' + info['invalid_reason'])
        if invalid_policy == 'skip':
            return None, dict(info, reason='invalid_geometry')
        geometry = make_valid(geometry)
        info['repaired'] = True
    polygons = list(_polygons(geometry))
    if not polygons:
        return None, dict(info, reason='no_polygon')
    geometry = unary_union(polygons) if len(polygons) > 1 else polygons[0]
    xmin, ymin, xmax, ymax = geometry.bounds
    if xmax <= 0 or ymax <= 0 or xmin >= width or ymin >= height:
        return None, dict(info, reason='outside_image')
    if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
        geometry = geometry.intersection(box(0, 0, width, height))
        info['clipped'] = True
    polygons = list(_polygons(geometry))
    if not polygons:
        return None, dict(info, reason='outside_image')
    segmentation = [np.asarray(poly.exterior.coords, dtype=float)[:-1, :2].ravel().tolist()
                    for poly in polygons]
    # 仅合并同一源 Feature 内的栅格部件，不合并不同 Feature。
    rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    area = float(mask_utils.area(rle))
    bbox = mask_utils.toBbox(rle).astype(float).tolist()
    info['representation'] = 'polygon'
    if area <= 0 or bbox[2] <= 0 or bbox[3] <= 0:
        return None, dict(info, reason='zero_pixel_mask', area=area, bbox=bbox)
    if bbox[2] < min_bbox_size or bbox[3] < min_bbox_size:
        return None, dict(info, reason='below_min_bbox_size', area=area, bbox=bbox)
    return {'segmentation': segmentation, 'bbox': bbox, 'area': area, 'iscrowd': 0}, info


def _index_files(directory, extensions):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    indexed = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in extensions:
            key = path.stem.casefold()
            if key in indexed:
                raise ValueError(f'存在同名文件，无法唯一匹配：{indexed[key]} / {path}')
            indexed[key] = path
    return indexed


def _image_size(path):
    # 只读文件头，不解码整张无人机影像。
    if gdal is not None:
        ds = gdal.Open(str(path))
        if ds is None:
            raise RuntimeError(f'无法打开影像：{path}')
        size = ds.RasterXSize, ds.RasterYSize
        ds = None
        return size
    with Image.open(path) as image:
        return image.size


def _atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, allow_nan=False, indent=2)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _prepare(label_dir, img_dir, out_json, extensions, category_name):
    labels, images = _index_files(label_dir, extensions), _index_files(img_dir, IMAGE_EXTENSIONS)
    if not labels:
        raise ValueError(f'没有找到标注文件：{label_dir}')
    missing_images = [str(path) for key, path in labels.items() if key not in images]
    if missing_images:
        raise ValueError(f'{len(missing_images)} 份标注未匹配影像，例如 {missing_images[0]}')
    report_path = None if out_json is None else str(Path(out_json).with_suffix('.report.json'))
    for path in (out_json, report_path):
        if path is not None and Path(path).exists():
            raise FileExistsError(f'不覆盖已有文件，请更换输出文件名：{path}')
    unmatched = [path.name for key, path in images.items() if key not in labels]
    if unmatched:
        warnings.warn(f'{len(unmatched)} 张影像没有对应标注文件，未作为空标注加入；详见报告。')
    coco = {'info': {'description': 'UAV PV instance annotations'}, 'licenses': [],
            'images': [], 'annotations': [],
            'categories': [{'id': 1, 'name': category_name, 'supercategory': 'object'}]}
    report = {'label_dir': str(label_dir), 'img_dir': str(img_dir),
              'images_without_label_file': unmatched, 'counts': Counter(), 'issues': []}
    return labels, images, coco, report, report_path


def _finish(coco, report, out_json, report_path):
    report['images'] = len(coco['images'])
    report['annotations'] = len(coco['annotations'])
    if out_json is not None:
        _atomic_json(out_json, coco)
        _atomic_json(report_path, report)
    print(f"完成：{report['images']} 张影像，{report['annotations']} 个实例。")
    print('质检统计：', dict(report['counts']))
    if out_json is not None:
        print(f'COCO：{out_json}\n报告：{report_path}')
    return coco


def shp2coco(shp_dir, img_dir, out_json, category_name='pv_panel', min_bbox_size=5,
             flip_y=False, invalid_policy='error'):
    """flip_y=True 用于将 GIS 的负 Y 恢复为 COCO 正 Y。

    无效面默认报错；repair 显式修复并记录；skip 显式跳过并记录。
    一个 SHP Feature 对应一个 COCO 实例，保留 source_fid/source_id 便于追溯。
    原 min_area 参数已替换为 min_bbox_size，单位为像素，不再执行面积过滤。
    """
    if ogr is None:
        raise RuntimeError('SHP 读取需要 GDAL/osgeo，请使用原 SHP 转换代码的 GIS 环境。')
    if not np.isfinite(min_bbox_size) or min_bbox_size < 0:
        raise ValueError('min_bbox_size 必须为非负有限值')
    ogr.UseExceptions()
    labels, images, coco, report, report_path = _prepare(shp_dir, img_dir, out_json, {'.shp'}, category_name)
    report.update(flip_y=flip_y, min_bbox_size=min_bbox_size, invalid_policy=invalid_policy,
                  area_bbox_definition='COCO rasterized per-object mask',
                  filter_rule='rasterized bbox width < min_bbox_size OR height < min_bbox_size',
                  instance_definition='one SHP Feature per instance')
    for image_id, (key, shp_file) in enumerate(tqdm(labels.items(), desc='转换 Shapefile'), 1):
        image_path = images[key]
        width, height = _image_size(image_path)
        ds = ogr.Open(str(shp_file), 0)
        if ds is None:
            raise RuntimeError(f'无法打开 SHP：{shp_file}')
        try:
            layer = ds.GetLayer(0)
            id_index = layer.GetLayerDefn().GetFieldIndex('ID')
            coco['images'].append({'id': image_id, 'file_name': image_path.name,
                                   'width': width, 'height': height})
            for feature in layer:
                report['counts']['source_features'] += 1
                geom = feature.GetGeometryRef()
                geometry = None if geom is None else from_wkb(bytes(geom.ExportToWkb()))
                try:
                    annotation, info = geometry_to_coco(geometry, width, height, flip_y,
                                                         min_bbox_size, invalid_policy)
                except Exception as exc:
                    raise RuntimeError(f'{shp_file.name}, FID={feature.GetFID()} 转换失败：{exc}') from exc
                if info['clipped']:
                    report['counts']['clipped'] += 1
                if info['repaired']:
                    report['counts']['repaired'] += 1
                if annotation is None or info['repaired']:
                    report['issues'].append({'file': shp_file.name, 'fid': feature.GetFID(), **info})
                if annotation is None:
                    report['counts'][info['reason']] += 1
                    continue
                annotation.update(id=len(coco['annotations']) + 1, image_id=image_id, category_id=1,
                                  source_fid=feature.GetFID())
                if id_index >= 0:
                    annotation['source_id'] = feature.GetField(id_index)
                coco['annotations'].append(annotation)
                report['counts'][info['representation']] += 1
        finally:
            ds = None
    return _finish(coco, report, out_json, report_path)


def mask2coco(mask_dir, img_dir, out_json, category_name='pv_panel', min_bbox_size=5,
              connectivity=4, foreground_value=1):
    """二值 mask 按连通域生成 RLE，保留每一个前景像素。

    默认维持原版 0/1、4 邻接；0/255 输入请显式设 foreground_value=255。
    不能将相接的多个组件重新识别为独立实例，真实实例优先使用 shp2coco。
    """
    import cv2
    if connectivity not in (4, 8) or foreground_value not in (1, 255):
        raise ValueError('connectivity 只能为 4/8，foreground_value 只能为 1/255')
    if not np.isfinite(min_bbox_size) or min_bbox_size < 0:
        raise ValueError('min_bbox_size 必须为非负有限值')
    warnings.warn('二值 mask 只能按连通域推定实例，无法恢复粘连组件的真实实例 ID。')
    labels, images, coco, report, report_path = _prepare(mask_dir, img_dir, out_json, IMAGE_EXTENSIONS, category_name)
    report.update(min_bbox_size=min_bbox_size, connectivity=connectivity, foreground_value=foreground_value,
                  area_bbox_definition='COCO rasterized per-object mask',
                  filter_rule='rasterized bbox width < min_bbox_size OR height < min_bbox_size',
                  instance_definition='connected component, not verified physical instance')
    for image_id, (key, path) in enumerate(tqdm(labels.items(), desc='转换二值 Mask'), 1):
        width, height = _image_size(images[key])
        if gdal is not None:
            ds = gdal.Open(str(path))
            if ds is None or ds.RasterCount != 1:
                raise ValueError(f'需要单波段 mask：{path}')
            mask = ds.GetRasterBand(1).ReadAsArray()
            ds = None
        else:
            with Image.open(path) as source:
                mask = np.asarray(source)
        if mask.shape != (height, width):
            raise ValueError(f'{path} mask 尺寸/通道 {mask.shape} 与影像 {(height, width)} 不符')
        if not np.all((mask == 0) | (mask == foreground_value)):
            raise ValueError(f'{path} 含 0/{foreground_value} 以外的值；不自动阈值化')
        binary = (mask == foreground_value).astype(np.uint8)
        n, component, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=connectivity)
        coco['images'].append({'id': image_id, 'file_name': images[key].name, 'width': width, 'height': height})
        # 从小窗游程直接生成整幅 RLE，避免每个实例扫描 height*width 像素。
        report['counts']['components'] += n - 1
        for label_id in range(1, n):
            x, y, w, h, area = map(int, stats[label_id])
            if w < min_bbox_size or h < min_bbox_size:
                report['counts']['below_min_bbox_size'] += 1
                continue
            roi = (component[y:y+h, x:x+w] == label_id)
            rle = _roi_to_full_rle(roi, x, y, width, height)
            assert int(mask_utils.area(rle)) == area
            bbox = mask_utils.toBbox(rle).astype(float).tolist()
            rle['counts'] = rle['counts'].decode('ascii')
            coco['annotations'].append({'id': len(coco['annotations']) + 1, 'image_id': image_id,
                                        'category_id': 1, 'segmentation': rle, 'area': area,
                                        'bbox': bbox, 'iscrowd': 0})
            report['counts']['rle'] += 1
    return _finish(coco, report, out_json, report_path)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Convert per-image pixel-space SHP polygons to COCO.')
    p.add_argument('--shp-dir', required=True)
    p.add_argument('--image-dir', required=True)
    p.add_argument('--out-json', required=True)
    p.add_argument('--category-name', default='pv_panel')
    p.add_argument('--min-bbox-size', type=float, default=5)
    p.add_argument('--flip-y', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--invalid-policy', choices=['error', 'skip'], default='error')
    a = p.parse_args()
    shp2coco(a.shp_dir, a.image_dir, a.out_json, category_name=a.category_name,
             min_bbox_size=a.min_bbox_size, flip_y=a.flip_y, invalid_policy=a.invalid_policy)
