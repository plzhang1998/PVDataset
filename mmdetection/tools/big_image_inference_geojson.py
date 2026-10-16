#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run instance segmentation on one large image and save predictions as GeoJSON.

The script deliberately reuses the large-image inference functions from
``big_image_inference_eval.py``.  Therefore its sliding-window geometry,
cross-tile instance stitching, NMS, and containment filtering are identical to
the benchmark evaluation pipeline, while no annotations or COCO evaluation are
required.

Example:
    python tools/big_image_inference_geojson.py \
        --image ./Dataset/test/ShaoXing/005/images/example.JPG \
        --checkpoint ./out/ckpt-detect/mask_rcnn.pth \
        --out ./out/example_predictions.geojson

For large panels, add ``--tta --tta-scales 1.0 0.75 0.5``. Each scale runs
the same 512px sliding-window pipeline. Masks are restored with nearest-neighbor
resampling before cross-view mask NMS; coordinates and y=-y remain relative to
the original input. ``--tta-flip`` also evaluates a horizontal flip at each
scale. TTA does not train the model or alter its internal test configuration.
"""

import argparse
import json
import math
import os
import os.path as osp
import time

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from mmdet.apis import init_detector

from big_image_inference_eval import (
    PATCH_SIZE,
    STRIDE,
    _build_test_pipeline,
    _resolve_image_path,
    _rle_to_geojson_feature,
    _rle_for_pycoco,
    encode_mask,
    mask_nms,
    filter_contained_predictions,
    inference_single_image,
)


DEFAULT_CONFIG = (
    'configs/pv-detect/mask-rcnn_r50_fpn_1x_pv_panel.py')
DEFAULT_CHECKPOINT = './checkpoints/mask_rcnn.pth'


def _tta_views(image_shape, scales, horizontal_flip=False):
    """Validate every view before inference; do not silently pad small views."""
    height, width = image_shape[:2]
    views = []
    seen = set()
    for scale in scales:
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('TTA scales must be finite positive numbers.')
        sw, sh = round(width * scale), round(height * scale)
        if min(sw, sh) < PATCH_SIZE:
            raise ValueError(
                f'TTA scale {scale:g} produces {sw}x{sh}, smaller than '
                f'the {PATCH_SIZE}px window. Remove this scale.')
        if (sw, sh) in seen:
            continue
        seen.add((sw, sh))
        for flip in ([False, True] if horizontal_flip else [False]):
            views.append(dict(scale=float(scale), width=sw, height=sh,
                              horizontal_flip=flip))
    if not views:
        raise ValueError('At least one TTA scale is required.')
    return views


def _restore_tta_mask(rle, height, width, horizontal_flip=False):
    """Invert augmentation in raster space and rebuild an exclusive xyxy box."""
    binary = mask_utils.decode(_rle_for_pycoco(rle)).astype(np.uint8)
    if horizontal_flip:
        binary = cv2.flip(binary, 1)
    if binary.shape != (height, width):
        binary = cv2.resize(binary, (width, height),
                            interpolation=cv2.INTER_NEAREST)
    restored = encode_mask(binary)
    x, y, w, h = mask_utils.toBbox(_rle_for_pycoco(restored))
    if w <= 0 or h <= 0:
        return None
    return np.array([x, y, x + w, y + h], dtype=np.float32), restored


def inference_tta(model, image, test_pipeline, scales=(1.0, 0.75, 0.5),
                  horizontal_flip=False, tta_nms_thr=0.5, **inference_kwargs):
    """Independent per-view stitching, then original-size mask NMS.

    No masks are unioned across augmentations. The retained instance keeps its
    model score (no score averaging). Containment filtering runs once at the
    end so a fragment can be compared against complete masks from other views.
    """
    height, width = image.shape[:2]
    views = _tta_views(image.shape, scales, horizontal_flip)
    if not math.isfinite(tta_nms_thr) or not 0 <= tta_nms_thr <= 1:
        raise ValueError('tta_nms_thr must be in [0, 1].')
    kwargs = dict(inference_kwargs)
    containment_thr = kwargs.pop('containment_thr', 0.0)
    boxes, scores, rles = [], [], []
    for index, view in enumerate(views, start=1):
        size = (view['width'], view['height'])
        if size == (width, height):
            augmented = image
        else:
            interpolation = cv2.INTER_AREA if view['scale'] < 1 else cv2.INTER_LINEAR
            augmented = cv2.resize(image, size, interpolation=interpolation)
        if view['horizontal_flip']:
            augmented = cv2.flip(augmented, 1)
        print(f"  TTA [{index}/{len(views)}] scale={view['scale']:g}, "
              f"size={size[0]}x{size[1]}, flip={view['horizontal_flip']}", flush=True)
        predictions = inference_single_image(
            model, augmented, test_pipeline, containment_thr=0.0, **kwargs)
        view['prediction_count'] = len(predictions['scores'])
        view['scale_x'] = view['width'] / width
        view['scale_y'] = view['height'] / height
        restored_count = 0
        for score, rle in zip(predictions['scores'], predictions['masks_rle']):
            restored = _restore_tta_mask(rle, height, width, view['horizontal_flip'])
            if restored is None:
                continue
            bbox, original_rle = restored
            boxes.append(bbox)
            scores.append(float(score))
            rles.append(original_rle)
            restored_count += 1
        view['restored_count'] = restored_count
        print(f"    retained per view: {restored_count}", flush=True)
        del augmented, predictions
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32)
    before_nms = len(scores)
    keep = mask_nms(boxes, scores, rles, iou_threshold=tta_nms_thr)
    boxes, scores, rles = boxes[keep], scores[keep], [rles[i] for i in keep]
    after_nms = len(scores)
    boxes, scores, rles, removed = filter_contained_predictions(
        boxes, scores, rles, containment_threshold=containment_thr)
    print(f'  TTA merge: {before_nms} -> mask NMS {after_nms} '
          f'-> containment {len(scores)}', flush=True)
    return dict(bboxes=boxes, scores=scores, masks_rle=rles), dict(
        enabled=True, views=views, merge='mask_nms', nms_threshold=tta_nms_thr,
        score_policy='retain selected prediction score; no averaging',
        before_nms=before_nms, after_nms=after_nms,
        containment_removed=removed, final_count=len(scores))


def _atomic_json_dump(path, data):
    """Avoid leaving a partially written GeoJSON if the process is interrupted."""
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _prediction_feature(index, bbox_xyxy, score, mask_rle, simplify_epsilon):
    """Make one GeoJSON feature with GIS-overlay-friendly image coordinates."""
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    properties = {
        'id': index,
        'class_id': 1,
        'class_name': 'pv',
        'score': float(score),
        # Preserve source-image coordinates for direct numerical use.
        'bbox_xywh_image_px': [x1, y1, x2 - x1, y2 - y1],
        # GeoJSON geometry uses (x, -y); this bbox follows the same convention.
        'bbox_geojson_xyxy_px': [x1, -y2, x2, -y1],
    }
    return _rle_to_geojson_feature(
        mask_rle, properties, simplify_epsilon=simplify_epsilon)


def main(args):
    if args.batch_size < 1:
        parser.error('--batch-size must be >= 1')
    for name in ('score_thr', 'nms_thr', 'containment_thr', 'tta_nms_thr'):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 <= value <= 1:
            parser.error(f'--{name.replace("_", "-")} must be in [0, 1]')
    if args.tta_flip and not args.tta:
        parser.error('--tta-flip requires --tta')

    image_path = _resolve_image_path(args.image)
    if not osp.isfile(image_path):
        parser.error(f'输入影像不存在：{image_path}')
    if args.out is None:
        image_stem, _ = osp.splitext(image_path)
        args.out = image_stem + '_predictions.geojson'
    if not args.out.lower().endswith(('.geojson', '.json')):
        args.out += '.geojson'
    out_dir = osp.dirname(osp.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        parser.error(f'OpenCV 无法读取输入影像：{image_path}')
    image_height, image_width = image.shape[:2]
    if args.tta:
        try:
            _tta_views(image.shape, args.tta_scales, args.tta_flip)
        except ValueError as exc:
            parser.error(str(exc))

    print('[1/3] 加载模型...')
    print(f'  Config: {args.config}')
    print(f'  Checkpoint: {args.checkpoint}')
    model = init_detector(args.config, args.checkpoint, device=args.device)
    test_pipeline = _build_test_pipeline(model)

    print('[2/3] 大图滑窗推理...')
    print(f'  Image: {image_path} ({image_width}×{image_height})')
    print(f'  Patch: {PATCH_SIZE}×{PATCH_SIZE}, stride: {STRIDE}, batch size: {args.batch_size}')
    t0 = time.time()
    inference_kwargs = dict(
        batch_size=args.batch_size,
        score_thr=args.score_thr,
        nms_type=args.nms_type,
        nms_thr=args.nms_thr,
        containment_thr=args.containment_thr,
        verbose=args.verbose)
    if args.tta:
        predictions, tta_info = inference_tta(
            model, image, test_pipeline, scales=args.tta_scales,
            horizontal_flip=args.tta_flip, tta_nms_thr=args.tta_nms_thr,
            **inference_kwargs)
    else:
        predictions = inference_single_image(
            model, image_path, test_pipeline, **inference_kwargs)
        tta_info = dict(enabled=False)
    del image
    elapsed = time.time() - t0

    print('[3/3] 写入 GeoJSON...')
    features = []
    for index, (bbox, score, mask_rle) in enumerate(
            zip(predictions['bboxes'], predictions['scores'], predictions['masks_rle']), start=1):
        feature = _prediction_feature(
            index, bbox, score, mask_rle, args.simplify_epsilon)
        if feature is not None:
            features.append(feature)

    geojson = {
        'type': 'FeatureCollection',
        'name': osp.splitext(osp.basename(image_path))[0] + '_predictions',
        'properties': {
            'source_image': osp.abspath(image_path),
            'image_width_px': image_width,
            'image_height_px': image_height,
            'class_name': 'pv',
            'prediction_count': len(features),
            'score_threshold': args.score_thr,
            'nms_type': args.nms_type,
            'nms_threshold': args.nms_thr,
            'containment_threshold': args.containment_thr,
            'patch_size_px': PATCH_SIZE,
            'stride_px': STRIDE,
            'config': osp.abspath(args.config),
            'checkpoint': osp.abspath(args.checkpoint),
            'tta': tta_info,
            'coordinate_system': 'image pixel coordinates; GeoJSON geometry uses x=x_image, y=-y_image',
            'y_transform': 'y_geojson = -y_image',
        },
        'features': features,
    }
    _atomic_json_dump(args.out, geojson)
    print(f'  检测实例数: {len(features)}')
    print(f'  推理耗时: {elapsed:.1f}s')
    print(f'  GeoJSON 已保存: {args.out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='单张大图滑窗实例分割推理，并输出所有预测实例的 GeoJSON。')
    parser.add_argument('--image', required=True,
                        help='输入原始大图路径（JPG/PNG/TIF 等 OpenCV 可读取格式）。')
    parser.add_argument('--out', default=None,
                        help='输出 GeoJSON 路径；默认与输入影像同目录，文件名后加 _predictions.geojson。')
    parser.add_argument('--config',
                        default='configs/pv-detect/mask2former_r50_8xb2-lsj-50e_pv_panel.py',
                        help='MMDetection 模型配置文件。')
    parser.add_argument('--checkpoint', 
                        default='./checkpoints/mask2former.pth',
                        help='训练完成的模型权重文件。')
    parser.add_argument('--device', default='cuda:0', help='推理设备（默认：cuda:0）。')
    parser.add_argument('--batch-size', type=int, default=27,
                        help='单张大图中同时推理的 512×512 切片数（默认：27）。')
    parser.add_argument('--score-thr', type=float, default=0.5,
                        help='切片预测置信度阈值（默认：0.50）。')
    parser.add_argument('--nms-type', choices=['bbox', 'mask'], default='mask',
                        help='每个尺度内拼接后的去重方式（默认：mask）。')
    parser.add_argument('--nms-thr', type=float, default=0.5,
                        help='全图 NMS IoU 阈值（默认：0.5）。')
    parser.add_argument('--containment-thr', type=float, default=0.5,
                        help='删除被更大掩码覆盖的预测比例；设为 0 可关闭（默认：0.5）。')
    parser.add_argument('--simplify-epsilon', type=float, default=1.0,
                        help='GeoJSON 轮廓简化容差，单位为像素；0 表示不简化（默认：1.0）。')
    parser.add_argument('--verbose', action='store_true',
                        help='输出切片拼接、NMS 和包含过滤的诊断信息。')
    parser.add_argument('--tta', action='store_true',
                        help='启用整图多尺度 TTA；默认关闭。')
    parser.add_argument('--tta-scales', nargs='+', type=float,
                        default=[1.0, 0.75, 0.5], metavar='SCALE',
                        help='相对原图缩放比例，需配合 --tta；默认 1.0 0.75 0.5。')
    parser.add_argument('--tta-flip', action='store_true',
                        help='配合 --tta，为每个尺度额外推理水平翻转视图。')
    parser.add_argument('--tta-nms-thr', type=float, default=0.5,
                        help='各 TTA 视图恢复到原图后，掩码 IoU NMS 阈值（默认 0.5）。')
    args = parser.parse_args()
    main(args)
