"""像方 SHP -> 每张影像一个 ISAT JSON（不复制或缩放影像）。

依赖：GDAL、Shapely >= 2、Pillow、tqdm。
输入必须是已投影到影像的像素坐标，不支持直接输入地理坐标。
默认将 QGIS 用的负 y 坐标翻回图像坐标；已有正 y 请关闭 FLIP_Y。
一个 SHP 要素（或指定 ID 字段值）为一个实例，多部件保持相同 group。
不做面积、长宽比或置信度过滤，不修复无效几何，不覆盖已有输出。
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

from PIL import Image
from shapely import affinity
from shapely.geometry import shape, box
from tqdm import tqdm

# -------- 配置区：也可直接修改这里运行 --------
ROOT = Path('./data/test/030')
SHP_DIR = ROOT / 'shp'       # 也可指定一张影像的 .shp 文件
IMAGE_DIR = ROOT / 'images'
OUTPUT_DIR = ROOT / 'ISAT'
CATEGORY_NAME = 'pv_panel'
FLIP_Y = True              # QGIS: y <= 0 -> 图像: y >= 0
ID_FIELD = 'ID'            # None: 每个要素独立；'ID': 同 ID 多部件为同一实例
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.tif', '.tiff', '.png', '.bmp'}


def image_index(image_dir):
    index = {}
    for p in sorted(Path(image_dir).iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            if p.stem.casefold() in index:
                raise ValueError(f'同名影像扩展名冲突，无法唯一匹配：{p}')
            index[p.stem.casefold()] = p
    if not index:
        raise ValueError(f'未找到影像：{image_dir}')
    return index


def polygon_parts(geom):
    if geom.is_empty:
        return []
    if geom.geom_type == 'Polygon':
        if len(geom.interiors):
            raise ValueError('输入含内环，不符合本数据集的无孔洞约定，请人工检查')
        return [geom] if geom.area > 0 else []
    if geom.geom_type in ('MultiPolygon', 'GeometryCollection'):
        return [p for g in geom.geoms for p in polygon_parts(g)]
    return []  # 裁剪产生的线/点不构成实例面积


def write_json_new(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def shp2isat(shp_dir, image_dir, output_dir, category_name='pv', flip_y=True,
             id_field='ID'):
    from osgeo import ogr
    ogr.UseExceptions()
    shp_dir, image_dir, output_dir = map(Path, (shp_dir, image_dir, output_dir))
    if not isinstance(category_name, str) or not category_name.strip() or category_name == '__background__':
        raise ValueError('请设置有效的前景类别名称')
    if output_dir.exists():
        raise FileExistsError(f'不覆盖已有标注目录（可能含人工修改）：{output_dir}')
    images = image_index(image_dir)
    sources = [shp_dir] if shp_dir.is_file() else sorted(shp_dir.glob('*'))
    sources = [p for p in sources if p.suffix.lower() == '.shp']
    shps = {}
    for p in sources:
        if p.stem.casefold() in shps:
            raise ValueError(f'重复 SHP 文件名：{p}')
        shps[p.stem.casefold()] = p
    if not shps:
        raise ValueError(f'未找到 SHP：{shp_dir}')
    if shp_dir.is_file():
        keys = list(shps)
    else:
        keys = sorted(images)
        missing = set(images) - set(shps)
        extra = set(shps) - set(images)
        if missing or extra:
            raise ValueError(f'SHP 与影像不一一对应。缺少 SHP: {sorted(missing)[:8]}；缺少影像: {sorted(extra)[:8]}')
    for key in keys:
        if key not in images:
            raise ValueError(f'未找到影像：{shps[key]}')
    output_dir.mkdir(parents=True)
    counts, files = Counter(), []
    for key in tqdm(keys, desc='SHP -> ISAT', unit='image'):
        image_path, shp_path = images[key], shps[key]
        with Image.open(image_path) as im:
            width, height = im.size
            depth = len(im.getbands()) if len(im.getbands()) > 1 else 0
            if im.getexif().get(274, 1) != 1:
                raise ValueError(f'影像带非默认 EXIF 方向，请先统一像素方向与标签：{image_path}')
        ds = ogr.Open(str(shp_path), 0)
        if ds is None:
            raise ValueError(f'无法打开 SHP：{shp_path}')
        layer = ds.GetLayer(0)
        if id_field and id_field not in [f.GetName() for f in layer.schema]:
            raise ValueError(f'{shp_path.name} 没有字段 {id_field!r}；可将 ID_FIELD 改为 None')
        extent = layer.GetExtent() if layer.GetFeatureCount() else None
        if extent:
            if flip_y and extent[2] >= 0 and extent[3] > 0:
                raise ValueError(f'{shp_path.name} 的 y 为正，疑似已反转，请设置 flip_y=False')
            if not flip_y and extent[3] <= 0 and extent[2] < 0:
                raise ValueError(f'{shp_path.name} 的 y 为负，请设置 flip_y=True')
        objects, groups = [], {}
        local = Counter()
        for feat in layer:
            fid = feat.GetFID()
            source_id = feat.GetField(id_field) if id_field else fid
            if source_id is None or str(source_id).strip() == '':
                raise ValueError(f'{shp_path.name} FID={fid}: 实例 ID 为空，请检查是否混入影像边界要素')
            group_key = (type(source_id).__name__, str(source_id))
            group = groups.setdefault(group_key, len(groups) + 1)
            ogr_geom = feat.GetGeometryRef()
            if ogr_geom is None:
                raise ValueError(f'{shp_path.name} FID={fid}: 空几何')
            geom = shape(json.loads(ogr_geom.ExportToJson()))
            if geom.geom_type not in ('Polygon', 'MultiPolygon') or not geom.is_valid:
                raise ValueError(f'{shp_path.name} FID={fid}: 几何无效或不是面，请先检查')
            polygon_parts(geom)  # 验证无内环
            if not all(math.isfinite(v) for v in geom.bounds):
                raise ValueError(f'{shp_path.name} FID={fid}: 非有限坐标')
            if flip_y:
                geom = affinity.scale(geom, xfact=1, yfact=-1, origin=(0, 0))
            clipped = geom.intersection(box(0, 0, width, height))
            local['source_features'] += 1
            if not clipped.equals(geom):
                local['clipped_features'] += 1
            parts = polygon_parts(clipped)
            if not parts:
                local['outside_or_degenerate'] += 1
            for part_index, part in enumerate(parts):
                points = [[float(p[0]), float(p[1])] for p in list(part.exterior.coords)[:-1]]
                objects.append(dict(category=category_name, group=group, segmentation=points,
                                    area=float(part.area), bbox=list(part.bounds), layer=len(objects)+1,
                                    iscrowd=False, shape_type='polygon',
                                    note=f'source_id={source_id}; shp_fid={fid}; part={part_index}'))
        ds = None
        local['instances'] = len({o['group'] for o in objects})
        local['objects'] = len(objects)
        data = dict(info=dict(description='ISAT', folder=str(image_dir.resolve()),
                              name=image_path.name, width=width, height=height, depth=depth,
                              note=f'source_shp={shp_path.name}; flip_y={flip_y}'), objects=objects)
        write_json_new(output_dir / (image_path.stem + '.json'), data)
        counts.update(local)
        files.append(dict(image=image_path.name, **local))
    write_json_new(output_dir / 'isat.yaml', {'label': [
        {'name': '__background__', 'color': '#000000'}, {'name': category_name, 'color': '#ff0000'}]})
    report = dict(images=len(keys), flip_y=flip_y, id_field=id_field, **counts, files=files)
    (output_dir / '_metadata').mkdir()
    write_json_new(output_dir / '_metadata' / 'shp2isat_report.json', report)
    print(f'完成：{len(keys)} 张影像，{counts["instances"]} 个实例，{counts["objects"]} 个轮廓。')
    print(f'裁剪 {counts["clipped_features"]} 个要素；完全在图外/退化 {counts["outside_or_degenerate"]} 个。')
    print(f'ISAT 标签目录：{output_dir}；影像仍使用：{image_dir}')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shp-dir', required=True)
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--category-name', default=CATEGORY_NAME)
    parser.add_argument('--id-field', default=ID_FIELD, help='实例身份字段；传入空字符串时按 FID 分组')
    parser.add_argument('--flip-y', action=argparse.BooleanOptionalAction, default=FLIP_Y)
    args = parser.parse_args()
    shp2isat(args.shp_dir, args.image_dir, args.output_dir, args.category_name,
             args.flip_y, args.id_field or None)
