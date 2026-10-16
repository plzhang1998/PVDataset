"""人工审核后的 ISAT JSON -> 一个 COCO instances.json。

与 shp2isat.py 放在同一目录。依赖 Shapely >= 2、Pillow、tqdm、pycocotools。
每个 ISAT object（多边形）独立对应一个实例，不按 group 合并或去重。
group 仅作为来源信息保存在 isat_group 字段，复制粘贴导致 group 相同不影响实例划分。
根据修改后的轮廓重新计算 area 和 bbox，不使用 ISAT 中可能过期的缓存字段。
area/bbox 来自 COCO 栅格化掩码；默认过滤 bbox 宽或高小于 5 像素的实例。
普通实例仍保留多边形轮廓，不按面积阈值过滤；宽、高恰好为 5 时保留。
正常实例输出 COCO polygon；iscrowd=1 输出 COCO RLE。空 objects 保留为负样本。
默认必须所有影像都有对应 JSON，缺失 JSON 不会自动当成负样本。
栅格化后没有任何前景像素的实例默认逐项记录并跳过；可用 --zero-pixel-policy error 停止。
自交、退化或坐标无效的多边形跳过，结束时统一提醒，详细记录写入 report。
输出 COCO 和 report 使用两空格缩进的多行 JSON。
"""
import argparse
import json
import math
from pathlib import Path

from PIL import Image
from shapely.geometry import Polygon, box
from shapely.validation import explain_validity
from pycocotools import mask as mask_utils
from tqdm import tqdm

from shp2isat import image_index, polygon_parts

# -------- 配置区：人工审查完成后修改这里 --------
ROOT = Path('./data/test/022')
ISAT_DIR = ROOT / 'ISAT_coco'
IMAGE_DIR = ROOT / 'images'
OUT_JSON = ROOT / 'instances.json'   
CATEGORIES = ['pv_panel']                         # 必须与 ISAT 类别名称一致
MIN_BBOX_SIZE = 5                                # 栅格 bbox 宽、高均须 >= 此值


def write_json_new(path, data):
    """另存可读的多行 JSON，不覆盖已有文件，不改变 SHP 转换脚本的输出。"""
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write('\n')


def polygon_parts_with_context(geom, image_path, label_path, stage,
                               category, group, object_indices, notes):
    """保留无孔洞检查，出错时补充可在 ISAT 中定位的来源和像素位置。"""
    try:
        return polygon_parts(geom)
    except ValueError as exc:
        locations = []
        pending = [geom]
        while pending:
            current = pending.pop()
            if current.geom_type == 'Polygon':
                for ring in current.interiors:
                    hole = Polygon(ring)
                    point = hole.representative_point()
                    bounds = ', '.join(f'{v:.2f}' for v in hole.bounds)
                    locations.append(
                        f'  空洞 {len(locations) + 1}: 内部参考点 '
                        f'(x={point.x:.2f}, y={point.y:.2f})，'
                        f'范围 [xmin, ymin, xmax, ymax]=[{bounds}]')
            elif current.geom_type in ('MultiPolygon', 'GeometryCollection'):
                pending.extend(reversed(list(current.geoms)))
        message = [
            str(exc),
            f'检查阶段：{stage}',
            f'影像：{image_path}',
            f'ISAT JSON：{label_path}',
            f'类别：{category}；group：{group!r}',
            f'对象序号（从 1 开始）：{[i + 1 for i in object_indices]}',
            f'objects 数组下标（从 0 开始）：{list(object_indices)}',
            f'来源备注 note：{notes!r}',
        ]
        if locations:
            message.append('空洞位置（原图像素坐标，左上角为原点，y 向下）：')
            message.extend(locations)
        raise ValueError('\n'.join(message)) from exc


def isat2coco(isat_dir, image_dir, out_json, categories=('pv',), zero_pixel_policy='skip',
              min_bbox_size=MIN_BBOX_SIZE):
    isat_dir, image_dir, out_json = map(Path, (isat_dir, image_dir, out_json))
    report_path = out_json.with_suffix('.report.json')
    if zero_pixel_policy not in ('error', 'skip'):
        raise ValueError('zero_pixel_policy 只能为 error 或 skip')
    if not math.isfinite(min_bbox_size) or min_bbox_size < 0:
        raise ValueError('min_bbox_size 必须为非负有限值')
    if out_json.exists() or report_path.exists():
        raise FileExistsError(f'输出或统计文件已存在，请另设文件名：{out_json}')
    if not categories or len(set(categories)) != len(categories) or any(
            not isinstance(c, str) or not c.strip() or c == '__background__' for c in categories):
        raise ValueError('categories 需为不重复的前景类别名称')
    cat_ids = {name: i + 1 for i, name in enumerate(categories)}
    images = image_index(image_dir)
    labels = {}
    for p in sorted(isat_dir.iterdir()):
        if p.is_file() and p.suffix.lower() == '.json':
            key = p.stem.casefold()
            if key in labels:
                raise ValueError(f'重复标签名称：{p}')
            labels[key] = p
    missing, extra = set(images) - set(labels), set(labels) - set(images)
    if missing or extra:
        raise ValueError(f'影像与 JSON 不一一对应，缺少 JSON: {sorted(missing)[:8]}；多余 JSON: {sorted(extra)[:8]}')
    coco = dict(info={'description': 'Reviewed UAV PV instance annotations'}, licenses=[], images=[],
                annotations=[], categories=[dict(id=i, name=n, supercategory='object') for n, i in cat_ids.items()])
    report = dict(images=len(images), instances=0, objects=0, clipped_objects=0, empty_images=0,
                  input_objects=0, skipped_invalid_polygons=[], invalid_only_images=[],
                  area_bbox_definition='COCO rasterized per-object mask',
                  instance_definition='one ISAT object per instance; group is metadata only',
                  min_bbox_size=min_bbox_size,
                  filter_rule='rasterized bbox width < min_bbox_size OR height < min_bbox_size',
                  zero_pixel_instances=[], skipped_small_bboxes=[], files=[])
    for image_id, key in enumerate(tqdm(sorted(images), desc='ISAT -> COCO', unit='image'), 1):
        image_path, label_path = images[key], labels[key]
        data = json.loads(label_path.read_text(encoding='utf-8-sig'))
        info = data.get('info', {})
        if info.get('description') != 'ISAT' or not isinstance(data.get('objects'), list):
            raise ValueError(f'不是 ISAT 标注：{label_path}')
        if Path(info.get('name', '')).stem.casefold() != key:
            raise ValueError(f'标签 info.name 与文件名不匹配：{label_path}')
        with Image.open(image_path) as im:
            width, height = im.size
            if im.getexif().get(274, 1) != 1:
                raise ValueError(f'请先处理 EXIF 方向：{image_path}')
        if (info.get('width'), info.get('height')) != (width, height):
            raise ValueError(f'标签与影像尺寸不符：{label_path}')
        instances = []
        invalid_start = len(report['skipped_invalid_polygons'])
        small_start = len(report['skipped_small_bboxes'])
        report['input_objects'] += len(data['objects'])
        for obj_index, obj in enumerate(data['objects']):
            name = obj.get('category')
            if name not in cat_ids:
                raise ValueError(f'{label_path.name} object={obj_index}: 未知类别 {name!r}，请修改 CATEGORIES 或标注')
            if obj.get('shape_type', 'polygon') != 'polygon':
                raise ValueError(f'{label_path.name} object={obj_index}: 仅支持 polygon')
            points = obj.get('segmentation', [])
            reason = None
            if not isinstance(points, (list, tuple)) or len(points) < 3:
                reason = '退化多边形：顶点不足 3 个'
            elif any(not isinstance(p, (list, tuple)) or len(p) != 2 or not all(
                    isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                    for v in p) for p in points):
                reason = '多边形坐标无效（需有限数值的二维坐标）'
            else:
                geom = Polygon(points)
                if not geom.is_valid:
                    reason = f'自交/退化多边形：{explain_validity(geom)}'
                elif geom.is_empty or geom.area <= 0:
                    reason = '退化多边形：面积为零'
            if reason:
                report['skipped_invalid_polygons'].append(dict(image=image_path.name,
                    label_file=label_path.name, object_index=obj_index, category=name,
                    group=obj.get('group'), reason=reason, note=obj.get('note', '')))
                continue  # 只跳过当前轮廓，其他有效轮廓仍独立保留
            clipped = geom.intersection(box(0, 0, width, height))
            if not clipped.equals(geom):
                report['clipped_objects'] += 1
            parts = polygon_parts_with_context(
                clipped, image_path, label_path, '单个对象裁剪后',
                name, obj.get('group'), [obj_index], [obj.get('note', '')])
            if not parts:
                raise ValueError(f'{label_path.name} object={obj_index}: 完全在影像外，请人工确认并删除')
            group = obj.get('group')
            if group is not None and (isinstance(group, bool) or not isinstance(group, int) or group < 0):
                raise ValueError(f'{label_path.name}: group 必须为非负整数或 null')
            crowd = obj.get('iscrowd', False)
            if not isinstance(crowd, (bool, int)) or crowd not in (0, 1):
                raise ValueError(f'{label_path.name}: iscrowd 必须为布尔值或 0/1')
            instances.append(dict(category=name, parts=parts, crowd=int(crowd),
                                  notes=[obj.get('note', '')], group=group,
                                  object_index=obj_index))
            report['objects'] += 1
        coco['images'].append(dict(id=image_id, file_name=image_path.name, width=width, height=height))
        before_count = len(coco['annotations'])
        for entry in instances:
            # 仅栅格化当前对象；若影像边界裁剪把它分成多段，仍属于这个源实例。
            # 不与其他 object 合并，即使 group 相同、相交或轮廓完全重合。
            parts = entry['parts']
            segmentation = [[float(v) for xy in list(p.exterior.coords)[:-1] for v in xy[:2]] for p in parts]
            rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
            area = float(mask_utils.area(rle))
            if area <= 0:
                if zero_pixel_policy == 'error':
                    raise ValueError(f'{label_path.name} object={entry["object_index"]}: 像素面积为零，请人工检查')
                report['zero_pixel_instances'].append(dict(image=image_path.name,
                    category=entry['category'], group=entry['group'],
                    object_index=entry['object_index'], source_notes=entry['notes'],
                    vector_area=sum(p.area for p in parts)))
                continue  # 无任何前景像素，不能成为可评估实例；逐项记录，不按面积阈值删除
            bbox = [float(v) for v in mask_utils.toBbox(rle)]
            if bbox[2] < min_bbox_size or bbox[3] < min_bbox_size:
                report['skipped_small_bboxes'].append(dict(image=image_path.name,
                    label_file=label_path.name, object_index=entry['object_index'],
                    category=entry['category'], group=entry['group'],
                    bbox=bbox, area=area, source_notes=entry['notes']))
                continue
            if entry['crowd']:
                segmentation = dict(size=rle['size'], counts=rle['counts'].decode('ascii'))
            coco['annotations'].append(dict(id=len(coco['annotations'])+1, image_id=image_id,
                category_id=cat_ids[entry['category']], segmentation=segmentation, area=area,
                bbox=bbox, iscrowd=entry['crowd'], isat_group=entry['group'],
                isat_object_index=entry['object_index'], source_notes=entry['notes']))
        retained = len(coco['annotations']) - before_count
        report['empty_images'] += retained == 0
        skipped = len(report['skipped_invalid_polygons']) - invalid_start
        if data['objects'] and skipped == len(data['objects']):
            report['invalid_only_images'].append(image_path.name)
        report['files'].append(dict(image=image_path.name, objects=len(data['objects']),
                                   skipped_invalid_polygons=skipped,
                                   skipped_small_bboxes=len(report['skipped_small_bboxes']) - small_start,
                                   input_instances=len(instances), instances=retained))
    report['instances'] = len(coco['annotations'])
    report['skipped_invalid_polygon_count'] = len(report['skipped_invalid_polygons'])
    report['invalid_polygon_image_count'] = len({p['image'] for p in report['skipped_invalid_polygons']})
    report['skipped_small_bbox_count'] = len(report['skipped_small_bboxes'])
    out_json.parent.mkdir(parents=True, exist_ok=True)
    write_json_new(out_json, coco)
    write_json_new(report_path, report)
    print(f'完成：{len(images)} 张影像，{report["instances"]} 个实例，{report["empty_images"]} 张空标注影像。')
    print(f'COCO：{out_json}\n统计：{report_path}')
    print(f'汇总提醒：跳过自交/退化或坐标无效多边形 {report["skipped_invalid_polygon_count"]} 个'
          f'（涉及 {report["invalid_polygon_image_count"]} 张影像）；'
          f'另跳过零像素实例 {len(report["zero_pixel_instances"])} 个。'
          f'框宽或高小于 {min_bbox_size:g} 像素过滤 {report["skipped_small_bbox_count"]} 个。'
          f'其中 {len(report["invalid_only_images"])} 张影像因所有轮廓无效而输出空标注。'
          '详情见 report。')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--isat-dir', required=True)
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--out-json', required=True)
    parser.add_argument('--categories', nargs='+', default=CATEGORIES)
    parser.add_argument('--zero-pixel-policy', choices=['skip', 'error'], default='skip',
                        help='无前景像素的实例：skip 逐项记录并跳过；error 报错停止')
    parser.add_argument('--min-bbox-size', type=float, default=MIN_BBOX_SIZE,
                        help='按栅格 bbox 过滤：宽或高小于此值时跳过；默认 5 像素，等于阈值保留')
    args = parser.parse_args()
    isat2coco(args.isat_dir, args.image_dir, args.out_json, args.categories,
              args.zero_pixel_policy, args.min_bbox_size)
