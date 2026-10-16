# -*- coding: utf-8 -*-
"""只裁切 COCO 标签，复用已经存在的影像切片，不读取或改写原图。

窗口与 crop.py(scale=1) 一致：512x512、步长448、末窗口贴边，文件名为
原图名_列序号_行序号.tif。按每张原图的 COCO width/height 计算窗口。
仅为 patch_image_dir 中实际存在的切片生成记录；缺失、额外文件汇总提醒。
真实多边形求交；同一源 annotation 的多部件仍是一个实例，不按 group 合并。
保留 polygon 顶点，area/bbox 用 pycocotools 栅格化结果计算；宽或高 <5 过滤。
RLE 输入保持 RLE 输出。自交/退化轮廓默认跳过并汇总，不静默修复；不支持孔洞。
按原图建标注索引，按空间索引查相交候选，逐条写出 annotation，降低内存占用。
输出为缩进 JSON，拒绝覆盖已有 COCO/report；失败时不留下半成品 COCO。
依赖 numpy、shapely>=2、pycocotools、Pillow、tqdm。末尾配置后直接运行。
"""
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from shapely.affinity import translate
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.validation import explain_validity
from tqdm import tqdm


def window_offsets(length, patch_size=512, stride=448):
    """与 crop.py 完全一致；末窗口不是简单的序号乘步长。"""
    if length < patch_size:
        return []
    return list(range(0, length-patch_size, stride)) + [length-patch_size]


def polygon_parts(geometry):
    if geometry.geom_type == 'Polygon':
        if not geometry.is_empty and geometry.area > 0:
            if geometry.interiors:
                raise ValueError('多边形含孔洞，不能用外环填满代替，请检查源实例')
            yield geometry
    elif geometry.geom_type in ('MultiPolygon', 'GeometryCollection'):
        for part in geometry.geoms:
            yield from polygon_parts(part)


def clip_polygon_to_patch(polygon, xoff, yoff, patch_width, patch_height):
    """真实求交，返回局部坐标 Polygon/MultiPolygon，不再返回压边顶点列表。"""
    geometry = Polygon(polygon)
    if not geometry.is_valid:
        raise ValueError(explain_validity(geometry))
    return translate(geometry.intersection(box(xoff, yoff, xoff+patch_width,
                                               yoff+patch_height)), -xoff, -yoff)


def prepare_annotation(ann, width, height):
    segmentation = ann.get('segmentation')
    if isinstance(segmentation, dict):
        if segmentation.get('size') != [height, width]:
            raise ValueError('RLE size 与原图尺寸不一致')
        rle = dict(segmentation)
        if isinstance(rle.get('counts'), list):
            rle = mask_utils.frPyObjects(rle, height, width)
        elif isinstance(rle.get('counts'), str):
            rle['counts'] = rle['counts'].encode('ascii')
        bbox = mask_utils.toBbox(rle).astype(float)
        if float(mask_utils.area(rle)) <= 0:
            raise ValueError('原始 RLE 没有前景像素')
        return dict(ann=ann, rle=rle, geom=box(bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]))
    if not isinstance(segmentation, list) or not segmentation:
        raise ValueError('缺少 polygon/RLE segmentation')
    parts = []
    for ring in segmentation:
        if not isinstance(ring, list) or len(ring) < 6 or len(ring) % 2:
            raise ValueError('退化多边形或坐标数量无效')
        pts = np.asarray(ring, dtype=float).reshape(-1, 2)
        if not np.isfinite(pts).all():
            raise ValueError('坐标包含 NaN/Inf')
        poly = Polygon(pts)
        if not poly.is_valid or poly.is_empty or poly.area <= 0:
            raise ValueError('无效多边形：'+explain_validity(poly))
        parts.append(poly)
    # 只在同一个原始 annotation 内处理部件，不合并不同光伏板。
    geom = unary_union(parts) if len(parts) > 1 else parts[0]
    list(polygon_parts(geom))  # 明确检查孔洞，不偷偷填洞
    return dict(ann=ann, geom=geom)


def crop_annotation(prepared, x, y, size, min_bbox_size):
    if 'rle' in prepared:
        mask = mask_utils.decode(prepared['rle'])
        roi = np.asfortranarray(mask[y:y+size, x:x+size], dtype=np.uint8)
        rle = mask_utils.encode(roi)
        segmentation = dict(size=rle['size'], counts=rle['counts'].decode('ascii'))
    else:
        clipped = prepared['geom'].intersection(box(x, y, x+size, y+size))
        parts = list(polygon_parts(clipped))
        if not parts:
            return None, 'no_intersection'
        segmentation = [np.asarray(translate(p, -x, -y).exterior.coords)[:-1, :2].ravel().tolist()
                        for p in parts]
        rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, size, size))
    area = float(mask_utils.area(rle))
    bbox = mask_utils.toBbox(rle).astype(float).tolist()
    if area <= 0:
        return None, 'zero_pixel_mask'
    if bbox[2] < min_bbox_size or bbox[3] < min_bbox_size:
        return None, 'below_min_bbox_size'
    annotation = dict(prepared['ann'])
    annotation.update(segmentation=segmentation, area=area, bbox=bbox,
                      iscrowd=prepared['ann'].get('iscrowd', 0),
                      source_annotation_id=prepared['ann']['id'],
                      source_image_id=prepared['ann']['image_id'])
    return annotation, 'kept'


class AnnotationWriter:
    """流式保存 annotations，不在 RAM 中累积数百万条结果。"""
    def __init__(self, directory):
        self.stream = tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8',
            dir=directory, suffix='.annotations.tmp', delete=False)
        self.path = Path(self.stream.name)
        self.count = 0

    def append(self, ann):
        if self.count:
            self.stream.write(',\n')
        json.dump(ann, self.stream, ensure_ascii=False, allow_nan=False, indent=2)
        self.count += 1

    def finish(self, destination, metadata):
        temp_path = None
        try:
            self.stream.flush()
            self.stream.seek(0)
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                    dir=destination.parent, suffix='.coco.tmp', delete=False) as out:
                temp_path = Path(out.name)
                out.write('{\n')
                for key, value in metadata.items():
                    out.write('  '+json.dumps(key)+': ')
                    json.dump(value, out, ensure_ascii=False, allow_nan=False, indent=2)
                    out.write(',\n')
                out.write('  "annotations": [\n')
                shutil.copyfileobj(self.stream, out)
                out.write('\n  ]\n}\n')
            if destination.exists():
                raise FileExistsError(destination)
            temp_path.rename(destination)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    def close(self):
        self.stream.close()
        self.path.unlink(missing_ok=True)


def crop_coco_annotations(original_coco_path, output_coco_path,
                          patch_size=512, stride=448, image_width=None, image_height=None,
                          *, patch_image_dir=None, min_bbox_size=5,
                          invalid_policy='skip', check_patch_size=True):
    """只生成标签。原 min_area 参数取消，改用 min_bbox_size（默认5像素）。

    image_width/height 仅用于兼容旧调用的尺寸校验；默认读取每张 COCO image 尺寸。
    patch_image_dir 默认是 output_coco_path 同级 images 文件夹，必须真实存在。
    缺失窗口不写入 COCO，额外旧切片不自动纳入；两者都会在最终报告中提醒。
    """
    for name, value in (('patch_size', patch_size), ('stride', stride)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f'{name} 必须为正整数')
    if stride > patch_size:
        raise ValueError('stride 不能大于 patch_size，以免窗口之间留下空隙')
    if not math.isfinite(min_bbox_size) or min_bbox_size < 0:
        raise ValueError('min_bbox_size 必须为非负有限值')
    if invalid_policy not in ('skip', 'error'):
        raise ValueError('invalid_policy 必须为 skip 或 error')
    source, output = Path(original_coco_path), Path(output_coco_path)
    report_path = output.with_suffix('.report.json')
    if output.exists() or report_path.exists():
        raise FileExistsError(f'拒绝覆盖已有结果，请修改输出文件名：{output}')
    patch_dir = Path(patch_image_dir) if patch_image_dir is not None else output.parent/'images'
    if not patch_dir.is_dir():
        raise FileNotFoundError(f'找不到已有切片影像目录：{patch_dir}')
    patch_files = {}
    for p in patch_dir.iterdir():
        if p.is_file() and p.suffix.lower() in ('.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp'):
            key = p.stem.casefold()
            if key in patch_files:
                raise ValueError(f'切片重名，无法唯一匹配：{p.name}')
            patch_files[key] = p
    with source.open(encoding='utf-8-sig') as stream:
        original = json.load(stream)
    image_ids = {im['id'] for im in original['images']}
    if len(image_ids) != len(original['images']):
        raise ValueError('原始 COCO image id 重复')
    by_image = defaultdict(list)
    ann_ids = set()
    for ann in original['annotations']:
        if ann['image_id'] not in image_ids or ann['id'] in ann_ids:
            raise ValueError(f'annotation id={ann["id"]} 重复或引用不存在的影像')
        ann_ids.add(ann['id'])
        by_image[ann['image_id']].append(ann)
    category_ids = {c['id'] for c in original['categories']}
    del original['annotations'], ann_ids  # 后续逐图释放索引，不保留全站的第二份引用
    counts = Counter()
    report = dict(source=str(source), patch_image_dir=str(patch_dir),
                  patch_size=patch_size, stride=stride, min_bbox_size=min_bbox_size,
                  area_bbox_definition='COCO rasterized per-patch instance mask',
                  instance_definition='one source annotation per patch; no grouping across annotations',
                  filter_rule='bbox width < min_bbox_size OR height < min_bbox_size',
                  counts=counts, invalid_examples=[], missing_examples=[], per_image=[])
    metadata = {k:original[k] for k in ('info','licenses','categories') if k in original}
    metadata.setdefault('info', {'description':'Reviewed UAV PV cropped instance annotations'})
    metadata.setdefault('licenses', [])
    metadata['images'] = []
    used = set()
    source_stems = set()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = AnnotationWriter(output.parent)
    try:
        for im in tqdm(original['images'], desc='裁切 COCO（复用影像）', unit='image'):
            w, h = im['width'], im['height']
            if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (w,h)):
                raise ValueError(f'原图尺寸无效：{im["file_name"]}')
            if (image_width is not None and image_width != w) or (image_height is not None and image_height != h):
                raise ValueError(f'配置尺寸与 COCO 不符：{im["file_name"]}，COCO={w}x{h}')
            stem = Path(im['file_name']).stem
            if stem.casefold() in source_stems:
                raise ValueError(f'原图文件名重复：{stem}')
            source_stems.add(stem.casefold())
            prepared = []
            for ann in by_image.pop(im['id'], []):
                counts['source_annotations'] += 1
                if ann['category_id'] not in category_ids:
                    raise ValueError(f'未知类别：annotation id={ann["id"]}')
                try:
                    prepared.append(prepare_annotation(ann,w,h))
                except (ValueError, TypeError, KeyError) as exc:
                    counts['invalid_source_annotations'] += 1
                    issue = dict(image=im['file_name'], annotation_id=ann['id'], reason=str(exc))
                    if len(report['invalid_examples']) < 100:
                        report['invalid_examples'].append(issue)
                    if invalid_policy == 'error':
                        raise ValueError(str(issue)) from exc
            tree = STRtree([p['geom'] for p in prepared]) if prepared else None
            saved = missing = 0
            before = writer.count
            xs, ys = window_offsets(w,patch_size,stride), window_offsets(h,patch_size,stride)
            for i,x in enumerate(xs):
                for j,y in enumerate(ys):
                    counts['expected_patches'] += 1
                    key = f'{stem}_{i:04d}_{j:04d}'.casefold()
                    patch = patch_files.get(key)
                    if patch is None:
                        missing += 1
                        counts['missing_patches'] += 1
                        if len(report['missing_examples']) < 100:
                            report['missing_examples'].append(key+'.tif')
                        continue
                    if check_patch_size:
                        with Image.open(patch) as bitmap:
                            if bitmap.size != (patch_size,patch_size):
                                raise ValueError(f'切片尺寸不符：{patch}，实际 {bitmap.size}')
                    used.add(key)
                    saved += 1
                    image_id = len(metadata['images'])+1
                    patch_info = {k:v for k,v in im.items() if k in ('license','date_captured')}
                    patch_info.update(id=image_id, file_name=patch.name, width=patch_size, height=patch_size,
                                      source_image_id=im['id'], source_file_name=im['file_name'],
                                      x_offset=x, y_offset=y)
                    metadata['images'].append(patch_info)
                    count_before = writer.count
                    candidates = sorted(tree.query(box(x,y,x+patch_size,y+patch_size)).tolist()) if tree is not None else []
                    for index in candidates:
                        ann, reason = crop_annotation(prepared[index],x,y,patch_size,min_bbox_size)
                        counts[reason] += 1
                        if ann is not None:
                            ann.update(id=writer.count+1, image_id=image_id)
                            writer.append(ann)
                    counts['empty_patches'] += writer.count == count_before
            report['per_image'].append(dict(image=im['file_name'], expected_patches=len(xs)*len(ys),
                saved_patches=saved, missing_patches=missing, annotations=writer.count-before))
        counts['extra_patch_files'] = len(patch_files)-len(used)
        report['extra_examples'] = [patch_files[k].name for k in sorted(patch_files.keys()-used)[:100]]
        report.update(images=len(metadata['images']), annotations=writer.count,
                      source_images=len(original['images']))
        if not metadata['images']:
            raise ValueError('没有匹配到任何切片，未生成 COCO；请检查目录、命名、尺寸和步长')
        writer.finish(output,metadata)
        with report_path.open('x',encoding='utf-8') as stream:
            json.dump(report,stream,ensure_ascii=False,allow_nan=False,indent=2)
            stream.write('\n')
    finally:
        writer.close()
    print(f'完成：{report["images"]} 张已有切片，{report["annotations"]} 个实例。')
    print(f'汇总：缺失切片 {counts["missing_patches"]}，额外旧切片 {counts["extra_patch_files"]}，'
          f'无效源实例 {counts["invalid_source_annotations"]}，零像素 {counts["zero_pixel_mask"]}，'
          f'框宽/高 < {min_bbox_size:g} 过滤 {counts["below_min_bbox_size"]}。')
    print(f'COCO：{output}\n报告：{report_path}')
    return report


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Crop COCO annotations to existing image tiles.')
    p.add_argument('--input-json', required=True)
    p.add_argument('--output-json', required=True)
    p.add_argument('--patch-image-dir', required=True)
    p.add_argument('--patch-size', type=int, default=512)
    p.add_argument('--stride', type=int, default=448)
    p.add_argument('--min-bbox-size', type=float, default=5)
    p.add_argument('--invalid-policy', choices=['skip', 'error'], default='skip')
    a = p.parse_args()
    crop_coco_annotations(a.input_json, a.output_json, patch_size=a.patch_size,
                          stride=a.stride, patch_image_dir=a.patch_image_dir,
                          min_bbox_size=a.min_bbox_size, invalid_policy=a.invalid_policy,
                          check_patch_size=True)
