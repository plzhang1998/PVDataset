#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
大图切片推理 + 评估脚本
用于在原始大图上评估实例分割模型性能

流程:
1. 将原始大图按训练时的参数裁切为 512x512 patches (stride=448)
2. 对每个 patch(batch) 运行模型推理
3. 将预测结果 (bbox/mask) 映射回大图坐标系
4. 对全图结果做 GreedyNMS 去除重叠区域的重复检测
5. 用 pycocotools 评估 COCO 格式指标 (bbox/segm mAP)

用法:
    python big_image_inference_eval.py \
        --config configs/mask_rcnn/mask-rcnn_r50_fpn_1x_pv_panel.py \
        --checkpoint /path/to/best.pth \
        --test-root Dataset/test \
        --out-dir results/mask_rcnn
"""

import argparse
import copy
import json
import os
import os.path as osp
import sys
import time
from collections import defaultdict
from functools import lru_cache

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector

# ==================== 常量配置 ====================
PATCH_SIZE = 512
STRIDE = 448

# A prediction is considered cut by a tile only when its bbox reaches this
# close to a non-image tile edge.  The value tolerates one or two pixels of
# mask/bbox quantisation error without treating ordinary nearby panels as a
# cut instance.
STITCH_EDGE_MARGIN = 4
STITCH_MIN_ORTHOGONAL_OVERLAP = 0.30
# A few leaked foreground pixels at a tile edge are not enough evidence that
# two panels are one instance.  A genuine cut instance appears in both tiles'
# 64px overlap area with substantial and spatially continuous mask support.
STITCH_MIN_MASK_OVERLAP_PIXELS = 64
STITCH_MIN_MASK_OVERLAP_RATIO = 0.50
STITCH_MIN_CROSS_BOUNDARY_SPAN = 8
# The containment-style overlap ratio above can still be 1.0 when a small
# false mask lies inside a large panel mask. Shared-region mask IoU rejects
# this case while retaining genuinely corresponding tile predictions.
STITCH_MIN_MASK_IOU = 0.20

# 模型配置映射 (根据 checkpoint 名称推断)
CONFIG_MAP = {
    'mask_rcnn': 'configs/pv-detect/mask-rcnn_r50_fpn_1x_pv_panel.py',
    'htc': 'configs/pv-detect/htc-without-semantic_r50_fpn_1x_pv_panel.py',
    'mask2former': 'configs/pv-detect/mask2former_r50_8xb2-lsj-50e_pv_panel.py',
    'solov2': 'configs/pv-detect/solov2_r50_fpn_1x_pv_panel.py',
}

# 测试集定义: city_name -> [(station_id, relative_path), ...]
TEST_SETS = {
    'NingBo': [
        ('013', 'NingBo/013'),
        ('022', 'NingBo/022'),
        ('024', 'NingBo/024'),
        ('028', 'NingBo/028'),
        ('030', 'NingBo/030'),
        ('032', 'NingBo/032'),
    ],
    'ShaoXing': [
        ('004', 'ShaoXing/004'),
        ('005', 'ShaoXing/005'),
        ('006', 'ShaoXing/006'),
    ],
    'ShenZhen': [
        ('031', 'ShenZhen/031'),
    ],
}

# ==================== 工具函数 ====================

def _sliding_offsets(length, patch_size=PATCH_SIZE, stride=STRIDE):
    """Return crop starts using exactly the same rule as preprocess/crop.py.

    The last crop is aligned to the right/bottom image edge instead of being
    a partial crop padded with zeros. This matches the training crops.
    """
    if length < patch_size:
        raise ValueError(
            f'Image side ({length}) is smaller than patch size ({patch_size}).')
    offsets = list(range(0, length - patch_size, stride))
    offsets.append(length - patch_size)
    return offsets


def slice_image(image):
    """将大图裁切为重叠的 patches

    Args:
        image: HxWxC numpy array

    Returns:
        patches: list of (patch_HxWxC, x_offset, y_offset)
    """
    H, W = image.shape[:2]
    x_offsets = _sliding_offsets(W)
    y_offsets = _sliding_offsets(H)
    return [
        (image[y:y + PATCH_SIZE, x:x + PATCH_SIZE], x, y)
        for y in y_offsets
        for x in x_offsets
    ]


def encode_mask(binary_mask):
    """将二值 mask 编码为 COCO RLE 格式

    Args:
        binary_mask: HxW numpy array (bool or uint8, 0/1)

    Returns:
        RLE dict with keys: size, counts
    """
    from pycocotools import mask as mask_utils
    mask = np.asarray(binary_mask, dtype=np.uint8)
    # Fortran order for column-major encoding (COCO standard)
    rle = mask_utils.encode(np.asfortranarray(mask))
    # bytes -> str for JSON serialization
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def _encode_local_mask(mask, x, y, image_height, image_width):
    """Encode a local binary mask at its global offset without a full canvas.

    Foreground runs are found separately per column (COCO Fortran order),
    shifted to global indices, and joined when contiguous. This preserves
    the exact pixels and canonical RLE, including image-edge cases.
    """
    from pycocotools import mask as mask_utils
    h, w = mask.shape
    if x < 0 or y < 0 or x + w > image_width or y + h > image_height:
        raise ValueError('Local mask is outside the image.')
    padded = np.zeros((w, h + 2), dtype=np.int8)
    padded[:, 1:-1] = np.asarray(mask.T, dtype=bool)
    changes = np.diff(padded, axis=1)
    sc, sr = np.nonzero(changes == 1)
    ec, er = np.nonzero(changes == -1)
    total = image_height * image_width
    if len(sc) == 0:
        counts = [total]
    else:
        starts = (sc.astype(np.int64) + x) * image_height + sr + y
        ends = (ec.astype(np.int64) + x) * image_height + er + y
        separated = starts[1:] != ends[:-1]
        starts = starts[np.r_[True, separated]]
        ends = ends[np.r_[separated, True]]
        counts_array = np.empty(2 * len(starts), dtype=np.int64)
        counts_array[0::2] = starts - np.r_[0, ends[:-1]]
        counts_array[1::2] = ends - starts
        counts = counts_array.tolist()
        if ends[-1] < total:
            counts.append(int(total - ends[-1]))
    rle = mask_utils.frPyObjects(
        dict(size=[image_height, image_width], counts=counts),
        image_height, image_width)
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def _rle_for_pycoco(rle):
    """Return an RLE with byte counts, as required by pycocotools."""
    rle = rle.copy()
    if isinstance(rle['counts'], str):
        rle['counts'] = rle['counts'].encode('utf-8')
    return rle


def greedy_nms(bboxes, scores, masks_rle, iou_threshold=0.5):
    """Greedy NMS: 按分数从高到低依次抑制重叠框

    Args:
        bboxes: (N, 4) array in [x1, y1, x2, y2] format
        scores: (N,) array
        masks_rle: list of RLE dicts
        iou_threshold: float

    Returns:
        keep: list of indices to keep
    """
    if len(bboxes) == 0:
        return []

    n = len(bboxes)
    order = scores.argsort()[::-1].tolist()
    suppressed = [False] * n

    for idx in range(n):
        i = order[idx]
        if suppressed[i]:
            continue
        for jdx in range(idx + 1, n):
            j = order[jdx]
            if suppressed[j]:
                continue
            # IoU 计算
            inter_x1 = max(bboxes[i, 0], bboxes[j, 0])
            inter_y1 = max(bboxes[i, 1], bboxes[j, 1])
            inter_x2 = min(bboxes[i, 2], bboxes[j, 2])
            inter_y2 = min(bboxes[i, 3], bboxes[j, 3])
            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h
            area_i = (bboxes[i, 2] - bboxes[i, 0]) * (bboxes[i, 3] - bboxes[i, 1])
            area_j = (bboxes[j, 2] - bboxes[j, 0]) * (bboxes[j, 3] - bboxes[j, 1])
            union = area_i + area_j - inter_area
            if union > 0 and inter_area / union > iou_threshold:
                suppressed[j] = True

    keep = [i for i in range(n) if not suppressed[i]]
    return keep


def mask_nms(bboxes, scores, masks_rle, iou_threshold=0.1):
    """Suppress only predictions whose masks overlap, not nearby objects."""
    if len(bboxes) == 0:
        return []

    from pycocotools import mask as mask_utils

    order = scores.argsort()[::-1].tolist()
    suppressed = np.zeros(len(bboxes), dtype=bool)
    rles = [_rle_for_pycoco(rle) for rle in masks_rle]

    for order_pos, i in enumerate(order):
        if suppressed[i]:
            continue

        # Bboxes only avoid needless mask comparisons. The final decision is
        # mask IoU, which is robust for tightly packed neighbouring panels.
        candidates = []
        for j in order[order_pos + 1:]:
            if suppressed[j]:
                continue
            inter_w = min(bboxes[i, 2], bboxes[j, 2]) - max(bboxes[i, 0], bboxes[j, 0])
            inter_h = min(bboxes[i, 3], bboxes[j, 3]) - max(bboxes[i, 1], bboxes[j, 1])
            if inter_w > 0 and inter_h > 0:
                candidates.append(j)

        if candidates:
            mask_ious = mask_utils.iou(
                [rles[i]], [rles[j] for j in candidates],
                [0] * len(candidates))[0]
            for j, mask_iou in zip(candidates, mask_ious):
                if mask_iou > iou_threshold:
                    suppressed[j] = True

    return [i for i in range(len(bboxes)) if not suppressed[i]]


def filter_contained_predictions(bboxes, scores, masks_rle,
                                 containment_threshold=0.0):
    """Remove a smaller mask when it is mostly contained by a larger one.

    This is intentionally asymmetric: a candidate is removed when at least
    ``containment_threshold`` of *its own* mask area is covered by an already
    retained larger mask. Area, not classification score, determines which
    duplicate to retain: tile fragments can occasionally have a higher score
    than the complete prediction that contains them.
    """
    if containment_threshold <= 0 or len(masks_rle) < 2:
        return bboxes, scores, masks_rle, 0
    if not 0 < containment_threshold <= 1:
        raise ValueError('containment_threshold must be in (0, 1].')

    from pycocotools import mask as mask_utils

    rles = [_rle_for_pycoco(rle) for rle in masks_rle]
    areas = np.asarray([float(mask_utils.area(rle)) for rle in rles])
    # Process larger masks first. For equal-area near-duplicates, prefer the
    # higher-score result but do not classify either as a "smaller" fragment.
    order = sorted(range(len(bboxes)),
                   key=lambda index: (areas[index], scores[index]), reverse=True)
    removed = np.zeros(len(bboxes), dtype=bool)

    for order_pos, index in enumerate(order):
        if areas[index] <= 0:
            removed[index] = True
            continue

        # Only compare with already retained strictly larger masks. Bounding-box
        # intersection is merely a fast pre-filter.
        previous = np.asarray(order[:order_pos], dtype=np.int64)
        previous = previous[(~removed[previous]) & (areas[previous] > areas[index])]
        if len(previous) == 0:
            continue
        inter_w = np.minimum(bboxes[index, 2], bboxes[previous, 2]) - \
            np.maximum(bboxes[index, 0], bboxes[previous, 0])
        inter_h = np.minimum(bboxes[index, 3], bboxes[previous, 3]) - \
            np.maximum(bboxes[index, 1], bboxes[previous, 1])
        candidates = previous[(inter_w > 0) & (inter_h > 0)].tolist()
        if not candidates:
            continue

        mask_ious = mask_utils.iou(
            [rles[index]], [rles[item] for item in candidates],
            [0] * len(candidates))[0]
        for other_index, mask_iou in zip(candidates, mask_ious):
            # IoU = I / (A + B - I)  ->  I = IoU * (A + B) / (1 + IoU)
            intersection = mask_iou * (areas[index] + areas[other_index]) / \
                (1.0 + mask_iou)
            if intersection / areas[index] >= containment_threshold:
                removed[index] = True
                break

    keep = np.flatnonzero(~removed)
    return (bboxes[keep], scores[keep], [masks_rle[index] for index in keep],
            int(removed.sum()))


def _mask_oriented_aspect_ratio(bbox, mask_rle):
    """Return long/short ratio of a single mask's minimum-area rectangle.

    A rotated rectangle is used instead of the axis-aligned bbox so normal
    photovoltaic panels remain valid even when the drone view is oblique.
    ``None`` means the mask is degenerate or disconnected and is retained by
    the caller rather than being rejected on an unreliable ratio.
    """
    mask = _decode_rle(mask_rle)
    height, width = mask.shape[:2]
    x1 = max(0, int(np.floor(bbox[0])))
    y1 = max(0, int(np.floor(bbox[1])))
    x2 = min(width, int(np.ceil(bbox[2])))
    y2 = min(height, int(np.ceil(bbox[3])))
    if x2 <= x1 or y2 <= y1:
        return None

    crop = mask[y1:y2, x1:x2]
    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 1:
        return None
    _, (side_a, side_b), _ = cv2.minAreaRect(contours[0])
    short_side = min(side_a, side_b)
    if short_side <= 0:
        return None
    return max(side_a, side_b) / short_side


def _decode_rle(rle):
    """Decode an in-memory RLE whose ``counts`` may have been made JSON-safe."""
    from pycocotools import mask as mask_utils
    rle = rle.copy()
    if isinstance(rle['counts'], str):
        rle['counts'] = rle['counts'].encode('utf-8')
    return mask_utils.decode(rle).astype(np.uint8)


def _edge_flags(local_bbox, edge_margin=STITCH_EDGE_MARGIN):
    """Return which sides of a 512px tile are touched by a detection."""
    x1, y1, x2, y2 = local_bbox
    return dict(
        left=x1 <= edge_margin,
        top=y1 <= edge_margin,
        right=x2 >= PATCH_SIZE - edge_margin,
        bottom=y2 >= PATCH_SIZE - edge_margin)


def _orthogonal_overlap_ratio(bbox_a, bbox_b, direction):
    """Overlap ratio on the axis orthogonal to a tile boundary."""
    if direction == 'horizontal':
        start = max(bbox_a[1], bbox_b[1])
        end = min(bbox_a[3], bbox_b[3])
        denom = min(bbox_a[3] - bbox_a[1], bbox_b[3] - bbox_b[1])
    else:
        start = max(bbox_a[0], bbox_b[0])
        end = min(bbox_a[2], bbox_b[2])
        denom = min(bbox_a[2] - bbox_a[0], bbox_b[2] - bbox_b[0])
    return max(0.0, end - start) / max(float(denom), 1e-6)


def _global_mask_overlap_metrics(record_a, record_b, direction,
                                 mask_a=None, mask_b=None):
    """Measure whether two masks are genuinely continuous across a tile edge.

    The overlap ratio is intersection divided by the smaller foreground area
    within the shared tile region. ``cross_boundary_span`` is the size of the
    common foreground in the direction perpendicular to the tile boundary.
    Both reject thin accidental contacts between neighbouring panels.
    """
    ax, ay = record_a['tile_x'], record_a['tile_y']
    bx, by = record_b['tile_x'], record_b['tile_y']
    x1, y1 = max(ax, bx), max(ay, by)
    x2 = min(ax + PATCH_SIZE, bx + PATCH_SIZE)
    y2 = min(ay + PATCH_SIZE, by + PATCH_SIZE)
    if x2 <= x1 or y2 <= y1:
        return 0, 0.0, 0, 0.0

    if mask_a is None:
        mask_a = _decode_rle(record_a['patch_rle'])
    if mask_b is None:
        mask_b = _decode_rle(record_b['patch_rle'])
    a_region = mask_a[y1 - ay:y2 - ay, x1 - ax:x2 - ax]
    b_region = mask_b[y1 - by:y2 - by, x1 - bx:x2 - bx]
    common = np.logical_and(a_region, b_region)
    intersection = int(common.sum())
    smaller_area = min(int(a_region.sum()), int(b_region.sum()))
    overlap_ratio = intersection / max(smaller_area, 1)
    if intersection == 0:
        return 0, overlap_ratio, 0, 0.0

    ys, xs = np.where(common)
    # Horizontal neighbours meet at a vertical edge, so continuity must span
    # several x pixels. Vertical neighbours require the analogous y span.
    cross_boundary_span = (int(xs.max() - xs.min() + 1)
                           if direction == 'horizontal'
                           else int(ys.max() - ys.min() + 1))
    union = int(a_region.sum()) + int(b_region.sum()) - intersection
    mask_iou = intersection / max(union, 1)
    return intersection, overlap_ratio, cross_boundary_span, mask_iou


def _stitch_match_score(record_a, record_b, direction, mask_a=None,
                        mask_b=None):
    """Return a stitch-match score, or ``None`` when the pair is unsafe.

    Only records from adjacent tiles are passed in.  Requiring both the
    opposite tile edges, mask continuity and shared-region IoU avoids merging
    distinct nearby photovoltaic panels. The returned IoU is used to enforce
    mutual-best one-to-one matching at each tile boundary.
    """
    if direction == 'horizontal':
        if not (record_a['edges']['right'] and record_b['edges']['left']):
            return None
    else:
        if not (record_a['edges']['bottom'] and record_b['edges']['top']):
            return None

    if _orthogonal_overlap_ratio(record_a['bbox'], record_b['bbox'], direction) < \
            STITCH_MIN_ORTHOGONAL_OVERLAP:
        return None

    intersection, overlap_ratio, cross_boundary_span, mask_iou = \
        _global_mask_overlap_metrics(record_a, record_b, direction,
                                     mask_a=mask_a, mask_b=mask_b)
    if (intersection < STITCH_MIN_MASK_OVERLAP_PIXELS or
            overlap_ratio < STITCH_MIN_MASK_OVERLAP_RATIO or
            cross_boundary_span < STITCH_MIN_CROSS_BOUNDARY_SPAN or
            mask_iou < STITCH_MIN_MASK_IOU):
        return None
    return mask_iou


def _should_stitch(record_a, record_b, direction):
    """Backward-compatible boolean wrapper around the strict match score."""
    return _stitch_match_score(record_a, record_b, direction) is not None


def _mutual_best_stitch_pairs(records, indices_a, indices_b, direction,
                              mask_getter=None):
    """Find one-to-one mutual-best stitch pairs across one tile boundary.

    A record can match at most one record in the directly adjacent tile. This
    prevents a broad, imperfect mask from joining multiple neighbouring panels
    and creating a transitive over-merged component.
    """
    best_for_a = {}
    best_for_b = {}
    for index_a in indices_a:
        for index_b in indices_b:
            mask_a = mask_getter(index_a) if mask_getter else None
            mask_b = mask_getter(index_b) if mask_getter else None
            score = _stitch_match_score(records[index_a], records[index_b],
                                        direction, mask_a=mask_a,
                                        mask_b=mask_b)
            if score is None:
                continue
            if (index_a not in best_for_a or
                    score > best_for_a[index_a][1]):
                best_for_a[index_a] = (index_b, score)
            if (index_b not in best_for_b or
                    score > best_for_b[index_b][1]):
                best_for_b[index_b] = (index_a, score)

    return [(index_a, index_b) for index_a, (index_b, _) in best_for_a.items()
            if best_for_b.get(index_b, (None,))[0] == index_a]


class _DisjointSet:
    """Small union-find implementation for grouping stitchable fragments."""

    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, item_a, item_b):
        root_a = self.find(item_a)
        root_b = self.find(item_b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def _merge_records_without_stitching(records, image_height, image_width):
    """Merge patch-level records to global canvas WITHOUT edge stitching.

    Each record becomes an independent prediction. Masks are placed on the
    global canvas individually, and a tight bbox is computed from each mask.
    """
    if not records:
        return np.zeros((0, 4), dtype=np.float32), np.array([]), [], 0

    bboxes, scores, masks_rle = [], [], []
    for record in records:
        patch_mask = _decode_rle(record['patch_rle'])
        y, x = record['tile_y'], record['tile_x']

        ys, xs = np.where(patch_mask)
        if len(xs) == 0:
            continue
        min_x = int(xs.min()) + x
        max_x = int(xs.max()) + x
        min_y = int(ys.min()) + y
        max_y = int(ys.max()) + y
        bboxes.append([float(min_x), float(min_y),
                       float(max_x + 1), float(max_y + 1)])
        scores.append(record['score'])
        masks_rle.append(_encode_local_mask(
            patch_mask[min_y-y:max_y-y+1, min_x-x:max_x-x+1],
            min_x, min_y, image_height, image_width))

    return (np.asarray(bboxes, dtype=np.float32),
            np.asarray(scores, dtype=np.float32), masks_rle, 0)


def stitch_boundary_instances(records, image_height, image_width):
    """Union predictions that represent one instance split across tile edges.

    Args:
        records: dictionaries holding a global bbox, a patch-local RLE mask,
            its score, and tile grid metadata.
        image_height, image_width: original large-image dimensions.

    Returns:
        bboxes, scores, masks_rle, num_stitched_groups
    """
    if not records:
        return np.zeros((0, 4), dtype=np.float32), np.array([]), [], 0

    records_by_tile = defaultdict(list)
    for index, record in enumerate(records):
        records_by_tile[(record['col'], record['row'])].append(index)

    # The same patch mask participates in comparisons with up to four
    # neighbour tiles. Caching recent decodes avoids repeated RLE decoding
    # while bounding the extra RAM to roughly 32 MiB (128 * 512 * 512 bytes).
    @lru_cache(maxsize=128)
    def decoded_mask(index):
        return _decode_rle(records[index]['patch_rle'])

    groups = _DisjointSet(len(records))
    for (col, row), left_or_top_indices in records_by_tile.items():
        # Horizontal neighbours: current tile is on the left.
        horizontal_pairs = _mutual_best_stitch_pairs(
            records, left_or_top_indices,
            records_by_tile.get((col + 1, row), []), 'horizontal',
            mask_getter=decoded_mask)
        for index_a, index_b in horizontal_pairs:
            groups.union(index_a, index_b)
        # Vertical neighbours: current tile is above.
        vertical_pairs = _mutual_best_stitch_pairs(
            records, left_or_top_indices,
            records_by_tile.get((col, row + 1), []), 'vertical',
            mask_getter=decoded_mask)
        for index_a, index_b in vertical_pairs:
            groups.union(index_a, index_b)

    components = defaultdict(list)
    for index in range(len(records)):
        components[groups.find(index)].append(index)

    merged_bboxes, merged_scores, merged_masks_rle = [], [], []
    stitched_groups = 0
    for indices in components.values():
        x0 = min(records[i]['tile_x'] for i in indices)
        y0 = min(records[i]['tile_y'] for i in indices)
        x1 = max(records[i]['tile_x'] + PATCH_SIZE for i in indices)
        y1 = max(records[i]['tile_y'] + PATCH_SIZE for i in indices)
        local_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        min_x = min_y = None
        max_x = max_y = None
        for index in indices:
            record = records[index]
            patch_mask = decoded_mask(index)
            y, x = record['tile_y'], record['tile_x']
            region = local_mask[y-y0:y-y0 + PATCH_SIZE, x-x0:x-x0 + PATCH_SIZE]
            np.maximum(region, patch_mask, out=region)

            # The tight global bbox can be accumulated from the 512x512 local
            # mask. This avoids scanning the whole multi-megapixel canvas once
            # more for every final instance.
            ys, xs = np.where(patch_mask)
            if len(xs):
                current_min_x, current_max_x = int(xs.min()) + x, int(xs.max()) + x
                current_min_y, current_max_y = int(ys.min()) + y, int(ys.max()) + y
                min_x = current_min_x if min_x is None else min(min_x, current_min_x)
                min_y = current_min_y if min_y is None else min(min_y, current_min_y)
                max_x = current_max_x if max_x is None else max(max_x, current_max_x)
                max_y = current_max_y if max_y is None else max(max_y, current_max_y)

        if min_x is None:
            continue
        merged_bboxes.append(
            [float(min_x), float(min_y), float(max_x + 1), float(max_y + 1)])
        merged_scores.append(max(records[index]['score'] for index in indices))
        merged_masks_rle.append(_encode_local_mask(
            local_mask[min_y-y0:max_y-y0+1, min_x-x0:max_x-x0+1],
            min_x, min_y, image_height, image_width))
        stitched_groups += len(indices) > 1

    return (np.asarray(merged_bboxes, dtype=np.float32),
            np.asarray(merged_scores, dtype=np.float32), merged_masks_rle,
            stitched_groups)


# ==================== 核心推理 ====================

def _resolve_image_path(image_path):
    """解决 JSON 中 file_name 与实际文件扩展名不匹配的问题

    例如: JSON 记录为 .JPG 但实际文件为 .tif
    """
    if osp.exists(image_path):
        return image_path
    # 尝试常见图像扩展名
    base, _ = osp.splitext(image_path)
    for ext in ['.tif', '.tiff', '.JPG', '.jpg', '.jpeg', '.png', '.bmp']:
        candidate = base + ext
        if osp.exists(candidate):
            return candidate
    return image_path  # 返回原路径, 让 cv2.imread 报错


def _build_test_pipeline(model):
    """从模型配置中预构建测试 pipeline (只构建一次)"""
    from mmengine.dataset import Compose
    cfg = model.cfg
    test_dataset_cfg = cfg.test_dataloader.dataset
    # ConcatDataset 没有直接的 pipeline, 从子数据集获取
    if hasattr(test_dataset_cfg, 'datasets'):
        pipeline_cfg = test_dataset_cfg.datasets[0].pipeline
    else:
        pipeline_cfg = test_dataset_cfg.pipeline
    pipeline_cfg = [dict(p) for p in pipeline_cfg]
    # 确保使用 LoadImageFromNDArray 处理 numpy 输入
    if pipeline_cfg[0]['type'] == 'LoadImageFromFile':
        pipeline_cfg[0]['type'] = 'mmdet.LoadImageFromNDArray'
    return Compose(pipeline_cfg)


def _batch_forward(model, test_pipeline, patch_list, batch_size=8):
    """批量推理 patches

    Args:
        model: mmdet model
        test_pipeline: 预构建的测试 pipeline
        patch_list: list of 512x512 numpy arrays
        batch_size: 每批处理的 patch 数量

    Returns:
        list of DetDataSample results
    """
    all_results = []
    for start in range(0, len(patch_list), batch_size):
        batch_patches = patch_list[start:start + batch_size]
        batch_inputs = []
        batch_data_samples = []
        for patch in batch_patches:
            data = dict(img=patch, img_id=0)
            data = test_pipeline(data)
            batch_inputs.append(data['inputs'])
            batch_data_samples.append(data['data_samples'])
        # 组成 batch
        data_ = dict(
            inputs=batch_inputs,
            data_samples=batch_data_samples
        )
        with torch.no_grad():
            results = model.test_step(data_)
        all_results.extend(results)
    return all_results


def inference_single_image(model, image_path, test_pipeline, batch_size=27,
                           score_thr=0.05, nms_thr=0.5,
                           nms_type='mask', containment_thr=0.0,
                           verbose=False, stitch=True):
    """对单张大图进行切片、拼接、掩膜/框 NMS。

    Args:
        model: mmdet model (from init_detector)
        image_path: str path, or a uint8 BGR ndarray (used by multi-scale TTA)
        test_pipeline: 预构建的测试 pipeline
        batch_size: batch 推理大小
        score_thr: float, confidence threshold
        nms_type: ``mask`` uses mask IoU; ``bbox`` uses bbox IoU.
        containment_thr: remove a smaller mask this fraction contained by a
            larger prediction; 0 disables this post-process.
        verbose: print per-image post-processing diagnostics.

    Returns:
        dict with keys: bboxes, masks_rle, scores
        - bboxes: (K, 4) ndarray in original image coordinates
        - masks_rle: list of K RLE dicts
        - scores: (K,) ndarray
    """
    if isinstance(image_path, np.ndarray):
        image = image_path
    else:
        image_path = _resolve_image_path(image_path)
        image = cv2.imread(image_path)
    if image is None:
        print(f"  [WARNING] Cannot read image: {image_path}")
        return dict(bboxes=np.zeros((0, 4)), masks_rle=[], scores=np.array([]))

    H, W = image.shape[:2]
    patches_info = slice_image(image)

    # 准备所有 patch (补边到 PATCH_SIZE)
    # crop.py always emits real, full-size crops. Keeping that geometry during
    # inference prevents a padded edge tile from changing the input domain.
    x_to_col = {offset: index for index, offset in enumerate(_sliding_offsets(W))}
    y_to_row = {offset: index for index, offset in enumerate(_sliding_offsets(H))}
    patch_images = [patch for patch, _, _ in patches_info]
    patch_meta = [
        (x_off, y_off, x_to_col[x_off], y_to_row[y_off])
        for _, x_off, y_off in patches_info
    ]

    # Batch 推理
    results = _batch_forward(model, test_pipeline, patch_images, batch_size)

    # 处理每个 patch 的推理结果
    records = []

    for result, (x_off, y_off, col, row) in zip(results, patch_meta):
        pred = result.pred_instances

        bboxes = pred.bboxes.cpu().numpy()
        scores = pred.scores.cpu().numpy()
        masks = pred.masks.cpu().numpy()  # (N, H_mask, W_mask)

        if len(bboxes) == 0:
            continue

        # 分数过滤
        keep = scores > score_thr
        if not keep.any():
            continue
        bboxes = bboxes[keep]
        scores = scores[keep]
        masks = masks[keep]

        # SOLOv2 returns valid masks with all-zero placeholder boxes. Derive
        # boxes only for these placeholders; preserve detector boxes otherwise.
        placeholder = np.all(bboxes == 0, axis=1)
        for index in np.flatnonzero(placeholder):
            ys, xs = np.nonzero(masks[index] > 0.5)
            if len(xs):
                bboxes[index] = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]

        # 判断 mask/bbox 坐标系: 通过 mask 分辨率判断
        mask_h, mask_w = masks.shape[1], masks.shape[2]
        # The model receives a patch, never the original large image. In
        # particular, a 512x512 original image must not be mistaken for a
        # full-image Mask2Former prediction.
        is_full_resolution = False  # model input is always a 512x512 patch

        if is_full_resolution:
            # ===== Mask2Former 模式: mask和bbox都在原图坐标系 =====
            # bbox 已在原图坐标系, 只需裁剪到图像边界
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, W)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, H)

            # 过滤空框
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            if (areas <= 0).any():
                valid = areas > 0
                bboxes = bboxes[valid]
                scores = scores[valid]
                masks = masks[valid]
            if len(bboxes) == 0:
                continue

            # mask 已在原图分辨率, 直接二值化并编码
            for i in range(len(masks)):
                binary_mask = (masks[i] > 0.5).astype(np.uint8)
                all_masks_rle.append(encode_mask(binary_mask))
        else:
            # ===== Mask R-CNN 模式: mask和bbox都在patch坐标系 =====
            # bbox 裁剪到 patch 有效区域
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, PATCH_SIZE)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, PATCH_SIZE)

            # 过滤空框
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            if (areas <= 0).any():
                valid = areas > 0
                bboxes = bboxes[valid]
                scores = scores[valid]
                masks = masks[valid]
            if len(bboxes) == 0:
                continue

            # 偏移到原图坐标系
            bboxes[:, 0::2] += x_off
            bboxes[:, 1::2] += y_off

            # mask 在 patch 分辨率, 放入全图画布
            # Keep masks in their compact patch-local RLE form until all
            # boundary fragments have been associated and stitched.
            binary_masks = (masks > 0.5).astype(np.uint8)
            for bbox, score, binary_mask in zip(bboxes, scores, binary_masks):
                local_bbox = bbox.copy()
                local_bbox[0::2] -= x_off
                local_bbox[1::2] -= y_off
                records.append(dict(
                    bbox=bbox.astype(np.float32),
                    score=float(score),
                    patch_rle=encode_mask(binary_mask),
                    tile_x=x_off,
                    tile_y=y_off,
                    col=col,
                    row=row,
                    edges=_edge_flags(local_bbox)))

    if len(records) == 0:
        return dict(bboxes=np.zeros((0, 4)), masks_rle=[], scores=np.array([]))

    # 跨切片策略：根据 stitch 参数决定是否执行边缘拼接。
    if stitch:
        all_bboxes, all_scores, all_masks_rle, stitched_groups = \
            stitch_boundary_instances(records, H, W)
        if verbose and stitched_groups:
            print(f"  Stitched {stitched_groups} cross-tile instance group(s).")
    else:
        all_bboxes, all_scores, all_masks_rle, _ = \
            _merge_records_without_stitching(records, H, W)

    # Mask NMS is preferable for tightly packed PV panels: nearby panels may
    # have overlapping boxes but disjoint masks, whereas tile duplicates have
    # overlapping masks even when their boxes are slightly offset.
    if nms_type == 'mask':
        keep = mask_nms(all_bboxes, all_scores, all_masks_rle,
                        iou_threshold=nms_thr)
    elif nms_type == 'bbox':
        keep = greedy_nms(all_bboxes, all_scores, all_masks_rle,
                          iou_threshold=nms_thr)
    else:
        raise ValueError(f'Unknown NMS type: {nms_type}')
    removed_by_nms = len(all_bboxes) - len(keep)
    if verbose and removed_by_nms:
        print(f"  {nms_type} NMS removed {removed_by_nms} duplicate prediction(s).")

    all_bboxes = all_bboxes[keep]
    all_scores = all_scores[keep]
    all_masks_rle = [all_masks_rle[index] for index in keep]

    # This operates only on final stitched/NMS predictions, never on raw tile
    # candidates. It therefore cannot change cross-tile associations.
    all_bboxes, all_scores, all_masks_rle, removed_contained = \
        filter_contained_predictions(
            all_bboxes, all_scores, all_masks_rle,
            containment_threshold=containment_thr)
    if verbose and removed_contained:
        print(f"  Containment filter removed {removed_contained} prediction(s) "
              f"(coverage >= {containment_thr:.2f}).")

    return dict(
        bboxes=all_bboxes,
        masks_rle=all_masks_rle,
        scores=all_scores,
    )


# ==================== 评估 ====================

def build_merged_gt(test_stations, data_root):
    """Merge station annotations and return their original-to-global ID map."""
    merged_gt = {'images': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'pv'}]}
    img_id_map = {}  # (station_id, original_img_id) -> new_img_id
    new_img_id = 1
    new_ann_id = 1

    for station_id, rel_path in test_stations:
        ann_file = osp.join(data_root, rel_path, 'instances.json')
        with open(ann_file, 'r') as f:
            gt = json.load(f)

        for img in gt['images']:
            old_id = img['id']
            img_id_map[(station_id, old_id)] = new_img_id
            new_img = img.copy()
            new_img['id'] = new_img_id
            # 修正 file_name 路径为相对于 data_root
            new_img['file_name'] = osp.join(rel_path, 'images', img['file_name'])
            merged_gt['images'].append(new_img)
            new_img_id += 1

        for ann in gt['annotations']:
            new_ann = ann.copy()
            new_ann['id'] = new_ann_id
            new_ann['image_id'] = img_id_map[(station_id, ann['image_id'])]
            merged_gt['annotations'].append(new_ann)
            new_ann_id += 1

    return merged_gt, img_id_map


def _build_coco_results_from_predictions(all_predictions, img_id_map, merged_gt):
    """Convert in-memory station predictions to global COCO detections."""
    id_to_filename = {img['id']: img['file_name'] for img in merged_gt['images']}
    coco_results = []
    for (station_id, orig_img_id), preds in all_predictions.items():
        new_id = img_id_map.get((station_id, orig_img_id))
        if new_id is None:
            continue
        file_name = id_to_filename.get(new_id, '')
        for i in range(len(preds['bboxes'])):
            bbox = preds['bboxes'][i].tolist()
            coco_results.append({
                'image_id': new_id,
                'file_name': file_name,
                'category_id': 1,
                'bbox': [round(bbox[0], 2), round(bbox[1], 2),
                         round(bbox[2] - bbox[0], 2), round(bbox[3] - bbox[1], 2)],
                'score': round(float(preds['scores'][i]), 6),
                'segmentation': preds['masks_rle'][i],
            })
    return coco_results


def evaluate_city(all_predictions, test_stations, data_root):
    """Build one COCO evaluation set from in-memory station predictions."""
    merged_gt, img_id_map = build_merged_gt(test_stations, data_root)
    coco_results = _build_coco_results_from_predictions(
        all_predictions, img_id_map, merged_gt)

    if len(coco_results) == 0:
        print("  [WARNING] No predictions to evaluate!")
        return None
    return merged_gt, coco_results


def evaluate_city_from_station_files(test_stations, data_root, result_paths):
    """Merge saved per-station COCO results without rerunning inference."""
    merged_gt, global_id_map = build_merged_gt(test_stations, data_root)
    id_to_filename = {img['id']: img['file_name'] for img in merged_gt['images']}
    coco_results = []

    for station_id, rel_path in test_stations:
        with open(result_paths[station_id], 'r', encoding='utf-8') as f:
            station_results = json.load(f)
        ann_file = osp.join(data_root, rel_path, 'instances.json')
        with open(ann_file, 'r', encoding='utf-8') as f:
            station_gt = json.load(f)

        # Per-station evaluation uses local IDs 1..N in the source image order.
        local_to_global = {
            local_id: global_id_map[(station_id, image['id'])]
            for local_id, image in enumerate(station_gt['images'], start=1)
        }
        for det in station_results:
            global_id = local_to_global.get(det['image_id'])
            if global_id is None:
                raise ValueError(
                    f"Unexpected image_id={det['image_id']} in {result_paths[station_id]}. "
                    "Delete this station cache and rerun it with the current script.")
            merged_det = dict(det)
            merged_det['image_id'] = global_id
            merged_det['file_name'] = id_to_filename[global_id]
            coco_results.append(merged_det)

    if len(coco_results) == 0:
        print("  [WARNING] No saved predictions to evaluate!")
        return None
    return merged_gt, coco_results


def _load_cached_metrics(summary_path, city_name):
    with open(summary_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if city_name not in data:
        raise ValueError(f"Missing key {city_name!r} in cached summary: {summary_path}")
    return data[city_name]


def _atomic_json_dump(path, data, **json_kwargs):
    """Write cache files atomically so interruptions never mark them complete."""
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, **json_kwargs)
    os.replace(tmp_path, path)


def _save_metrics(summary_path, city_name, metrics):
    _atomic_json_dump(
        summary_path, {city_name: metrics}, indent=2, ensure_ascii=False)


def _print_metrics(title, metrics):
    print(f"\n  >>> {title} 汇总指标:")
    print(f"    BBox mAP:   {metrics['bbox_mAP']:.4f}")
    print(f"    BBox mAP50: {metrics['bbox_mAP50']:.4f}")
    print(f"    BBox mAP75: {metrics['bbox_mAP75']:.4f}")
    print(f"    BBox P@0.5: {metrics['bbox_precision_iou50']:.4f} "
          f"(TP={metrics['bbox_tp_iou50']}, FP={metrics['bbox_fp_iou50']})")
    print(f"    BBox R@0.5: {metrics['bbox_recall_iou50']:.4f} "
          f"(FN={metrics['bbox_fn_iou50']})")
    print(f"    Segm mAP:   {metrics['segm_mAP']:.4f}")
    print(f"    Segm mAP50: {metrics['segm_mAP50']:.4f}")
    print(f"    Segm mAP75: {metrics['segm_mAP75']:.4f}")


def _segmentation_to_rle(segmentation, image_height, image_width):
    """Convert either polygon or RLE COCO segmentation to pycocotools RLE."""
    from pycocotools import mask as mask_utils

    if isinstance(segmentation, list):
        polygons = segmentation
        rles = mask_utils.frPyObjects(polygons, image_height, image_width)
        return mask_utils.merge(rles)
    return _rle_for_pycoco(segmentation)


def _union_rle_masks(rles, image_height, image_width):
    """Decode and union RLE masks one at a time to keep memory bounded."""
    from pycocotools import mask as mask_utils

    union = np.zeros((image_height, image_width), dtype=np.uint8)
    for rle in rles:
        mask = mask_utils.decode(_rle_for_pycoco(rle))
        np.maximum(union, mask.astype(np.uint8), out=union)
    return union


def _normalise_geojson_modes(modes):
    """Expand the convenient ``all`` alias and preserve a stable order."""
    all_modes = ['pred', 'tp', 'fp', 'fn', 'tn']
    if 'all' in modes:
        return all_modes
    return [mode for mode in all_modes if mode in modes]


def _binary_mask_to_geojson_geometry(binary_mask, simplify_epsilon=1.0):
    """Convert a binary image mask to Polygon or MultiPolygon GeoJSON.

    Coordinates are image pixels with the requested GIS-compatible transform
    ``(x, y) -> (x, -y)``. The image upper-left is therefore (0, 0), with
    downward image rows represented by negative GeoJSON y values.
    ``simplify_epsilon`` is measured in pixels and keeps large mask GeoJSON
    files practical without changing the underlying prediction logic.
    """
    contours, hierarchy = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return None

    hierarchy = hierarchy[0]

    def contour_to_ring(contour):
        if simplify_epsilon > 0:
            contour = cv2.approxPolyDP(contour, simplify_epsilon, True)
        if len(contour) < 3:
            return None
        ring = [[float(point[0][0]), -float(point[0][1])] for point in contour]
        ring.append(ring[0])
        return ring

    polygons = []
    for index, contour in enumerate(contours):
        # RETR_CCOMP gives exterior contours a parent of -1; its direct
        # children are holes of the same polygon.
        if hierarchy[index][3] != -1:
            continue
        outer_ring = contour_to_ring(contour)
        if outer_ring is None:
            continue
        rings = [outer_ring]
        child = hierarchy[index][2]
        while child != -1:
            hole_ring = contour_to_ring(contours[child])
            if hole_ring is not None:
                rings.append(hole_ring)
            child = hierarchy[child][0]
        polygons.append(rings)

    if not polygons:
        return None
    if len(polygons) == 1:
        return {'type': 'Polygon', 'coordinates': polygons[0]}
    return {'type': 'MultiPolygon', 'coordinates': polygons}


def _rle_to_geojson_feature(rle, properties, simplify_epsilon=1.0):
    """Decode one RLE and make a GeoJSON feature, or None for an empty mask."""
    from pycocotools import mask as mask_utils

    binary_mask = mask_utils.decode(_rle_for_pycoco(rle)).astype(np.uint8)
    geometry = _binary_mask_to_geojson_geometry(binary_mask, simplify_epsilon)
    if geometry is None:
        return None
    return {'type': 'Feature', 'properties': properties, 'geometry': geometry}


def save_mask_geojson_visualizations(merged_gt, coco_results, output_dir,
                                     modes=('all',), mask_iou_thr=0.5,
                                     simplify_epsilon=1.0):
    """Save prediction / TP / FP / FN / TN masks as per-image GeoJSON files.

    A prediction is TP when it is the highest-score unmatched prediction with
    mask IoU >= ``mask_iou_thr`` against a GT instance. All other predictions
    are FP; unmatched GT masks are FN. TN is a *pixel-level* background mask:
    pixels covered by neither any prediction nor any GT mask. Prediction
    features always contain a ``score`` property; GT-only FN and TN features
    use ``score: null`` because they are not model predictions.
    """
    from pycocotools import mask as mask_utils

    modes = _normalise_geojson_modes(modes)
    output_names = {
        'pred': 'all_predictions', 'tp': 'tp', 'fp': 'fp',
        'fn': 'fn', 'tn': 'tn',
    }
    for mode in modes:
        os.makedirs(osp.join(output_dir, output_names[mode]), exist_ok=True)

    gt_by_image = defaultdict(list)
    for ann in merged_gt['annotations']:
        if (ann.get('category_id') == 1 and not ann.get('iscrowd', 0)
                and not ann.get('ignore', 0)):
            gt_by_image[ann['image_id']].append(ann)

    dt_by_image = defaultdict(list)
    for det in coco_results:
        if det.get('category_id') == 1:
            dt_by_image[det['image_id']].append(det)

    print(f"\n  Saving mask GeoJSON files (mask IoU={mask_iou_thr:.2f}, "
          f"simplify={simplify_epsilon}px)...")
    saved_count = 0
    for image in tqdm(merged_gt['images'], desc='  Mask GeoJSON', ncols=80):
        image_id = image['id']
        image_height, image_width = image['height'], image['width']
        gt_items = gt_by_image[image_id]
        dt_items = sorted(dt_by_image[image_id],
                          key=lambda item: item['score'], reverse=True)

        gt_rles = [_segmentation_to_rle(item['segmentation'], image_height,
                                        image_width) for item in gt_items]
        dt_rles = [_rle_for_pycoco(item['segmentation']) for item in dt_items]
        matched_gt = np.zeros(len(gt_rles), dtype=bool)
        matched_pairs = []

        # The matrix is calculated in compressed RLE form; full-size masks are
        # decoded only once later, while saving the requested binary images.
        if dt_rles and gt_rles:
            iou_matrix = mask_utils.iou(dt_rles, gt_rles, [0] * len(gt_rles))
        else:
            iou_matrix = np.zeros((len(dt_rles), len(gt_rles)), dtype=np.float32)

        for det_index in range(len(dt_rles)):
            available_ious = iou_matrix[det_index].copy()
            available_ious[matched_gt] = -1
            best_gt = int(np.argmax(available_ious)) if len(gt_rles) else -1
            if best_gt >= 0 and available_ious[best_gt] >= mask_iou_thr:
                matched_gt[best_gt] = True
                matched_pairs.append(
                    (det_index, best_gt, float(available_ious[best_gt])))
            else:
                pass

        match_info = {
            det_index: (gt_index, matched_iou)
            for det_index, gt_index, matched_iou in matched_pairs
        }

        features_by_mode = {mode: [] for mode in modes}
        for det_index, (det_item, det_rle) in enumerate(zip(dt_items, dt_rles)):
            is_tp = det_index in match_info
            result_type = 'tp' if is_tp else 'fp'
            gt_index, matched_iou = match_info.get(det_index, (None, None))
            det_x, det_y, det_w, det_h = det_item['bbox']
            det_xyxy = np.asarray(
                [det_x, det_y, det_x + det_w, det_y + det_h], dtype=np.float32)
            properties = {
                'image_id': image_id,
                'result_type': result_type,
                'score': float(det_item['score']),
                'mask_iou': matched_iou,
                'aspect_ratio': _mask_oriented_aspect_ratio(det_xyxy, det_rle),
                'matched_gt_id': (gt_items[gt_index]['id'] if gt_index is not None else None),
            }
            feature = _rle_to_geojson_feature(
                det_rle, properties, simplify_epsilon=simplify_epsilon)
            if feature is not None:
                if 'pred' in modes:
                    features_by_mode['pred'].append(feature)
                if result_type in modes:
                    features_by_mode[result_type].append(feature)

        if 'fn' in modes:
            for gt_index in np.flatnonzero(~matched_gt):
                gt_item = gt_items[int(gt_index)]
                gt_x, gt_y, gt_w, gt_h = gt_item['bbox']
                gt_xyxy = np.asarray(
                    [gt_x, gt_y, gt_x + gt_w, gt_y + gt_h], dtype=np.float32)
                feature = _rle_to_geojson_feature(
                    gt_rles[int(gt_index)], {
                        'image_id': image_id,
                        'result_type': 'fn',
                        'score': None,
                        'mask_iou': None,
                        'aspect_ratio': _mask_oriented_aspect_ratio(
                            gt_xyxy, gt_rles[int(gt_index)]),
                        'gt_annotation_id': gt_item['id'],
                    }, simplify_epsilon=simplify_epsilon)
                if feature is not None:
                    features_by_mode['fn'].append(feature)

        if 'tn' in modes:
            pred_union = _union_rle_masks(dt_rles, image_height, image_width)
            gt_union = _union_rle_masks(gt_rles, image_height, image_width)
            tn_mask = np.logical_not(np.logical_or(pred_union, gt_union)).astype(np.uint8)
            tn_geometry = _binary_mask_to_geojson_geometry(
                tn_mask, simplify_epsilon=simplify_epsilon)
            if tn_geometry is not None:
                features_by_mode['tn'].append({
                    'type': 'Feature',
                    'properties': {
                        'image_id': image_id,
                        'result_type': 'tn',
                        'score': None,
                        'mask_iou': None,
                        'aspect_ratio': None,
                    },
                    'geometry': tn_geometry,
                })

        stem = osp.splitext(osp.basename(image['file_name']))[0]
        for mode, features in features_by_mode.items():
            output_path = osp.join(output_dir, output_names[mode],
                                   f'{image_id:04d}_{stem}.geojson')
            feature_collection = {
                'type': 'FeatureCollection',
                'name': f'{image_id:04d}_{stem}_{mode}',
                'properties': {
                    'image_id': image_id,
                    'image_file': image['file_name'],
                    'coordinate_system': 'image_pixel',
                    'pixel_origin': 'upper_left',
                    'pixel_y_transform': 'geojson_y = -image_row',
                    'mask_iou_threshold': mask_iou_thr,
                },
                'features': features,
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(feature_collection, f, ensure_ascii=False)
            saved_count += 1

    print(f"  Saved {saved_count} GeoJSON file(s): {output_dir}")


def _dense_coco_metrics(coco_eval, max_dets):
    """Compute COCO AP/AR for a non-default per-image detection limit."""
    precision = coco_eval.eval['precision']  # [IoU, recall, cls, area, maxDet]
    recall = coco_eval.eval['recall']        # [IoU, cls, area, maxDet]
    iou_thrs = coco_eval.params.iouThrs

    def mean_precision(iou=None):
        values = precision[:, :, :, 0, -1]
        if iou is not None:
            idx = np.where(np.isclose(iou_thrs, iou))[0]
            values = values[idx]
        values = values[values > -1]
        return float(values.mean()) if values.size else float('nan')

    def mean_recall(iou=None):
        values = recall[:, :, 0, -1]
        if iou is not None:
            idx = np.where(np.isclose(iou_thrs, iou))[0]
            values = values[idx]
        values = values[values > -1]
        return float(values.mean()) if values.size else float('nan')

    metrics = {
        'mAP': mean_precision(),
        'mAP50': mean_precision(0.50),
        'mAP75': mean_precision(0.75),
        'AR': mean_recall(),
    }
    print(f"  Dense-image metrics (maxDets={max_dets} per image):")
    print(f"    AP@[.50:.95]: {metrics['mAP']:.4f}")
    print(f"    AP@.50:       {metrics['mAP50']:.4f}")
    print(f"    AP@.75:       {metrics['mAP75']:.4f}")
    print(f"    AR@[.50:.95]: {metrics['AR']:.4f}")
    return metrics


def evaluate_coco_with_progress(coco_eval, stage_name):
    """Run COCOeval.evaluate() with progress bars, preserving its logic.

    The upstream ``COCOeval.evaluate`` performs two opaque, potentially long
    loops: per-image IoU computation and per-image/per-area matching.  This
    is a line-for-line equivalent of that method except that both loops are
    driven by tqdm, which is essential for dense large-image datasets.
    """
    started_at = time.time()
    print(f'Running per image evaluation ({stage_name})...')
    params = coco_eval.params
    if params.useSegm is not None:
        params.iouType = 'segm' if params.useSegm == 1 else 'bbox'
        print(f'useSegm (deprecated) is not None. Running {params.iouType} evaluation')
    print(f'Evaluate annotation type *{params.iouType}*')

    params.imgIds = list(np.unique(params.imgIds))
    if params.useCats:
        params.catIds = list(np.unique(params.catIds))
    params.maxDets = sorted(params.maxDets)
    coco_eval.params = params
    coco_eval._prepare()

    category_ids = params.catIds if params.useCats else [-1]
    if params.iouType in ('segm', 'bbox'):
        compute_iou = coco_eval.computeIoU
    elif params.iouType == 'keypoints':
        compute_iou = coco_eval.computeOks
    else:
        raise ValueError(f'Unsupported COCO IoU type: {params.iouType}')

    pairs = [(image_id, category_id)
             for image_id in params.imgIds for category_id in category_ids]
    coco_eval.ious = {}
    for image_id, category_id in tqdm(
            pairs, desc=f'  {stage_name} IoU', ncols=88, unit='image'):
        coco_eval.ious[(image_id, category_id)] = compute_iou(image_id, category_id)

    evaluate_image = coco_eval.evaluateImg
    max_det = params.maxDets[-1]
    total_evaluations = len(category_ids) * len(params.areaRng) * len(params.imgIds)
    coco_eval.evalImgs = []
    with tqdm(total=total_evaluations, desc=f'  {stage_name} match',
              ncols=88, unit='image') as progress:
        # Keep this order identical to pycocotools.COCOeval.evaluate().
        for category_id in category_ids:
            for area_range in params.areaRng:
                for image_id in params.imgIds:
                    coco_eval.evalImgs.append(
                        evaluate_image(image_id, category_id, area_range, max_det))
                    progress.update(1)

    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
    print(f'DONE (t={time.time() - started_at:0.2f}s).')


def bbox_precision_recall_at_iou(merged_gt, coco_results, iou_thr=0.5,
                                 max_dets=1000):
    """计算固定 IoU、固定 score-thr 下的检测 Precision / Recall。

    这不是 COCO AP：预测已在推理阶段按 ``--score-thr`` 过滤，再按分数
    从高到低与同一张图上的 GT 做一对一匹配。因而能直接看出当前工作阈值下
    的误检（FP）与漏检（FN）数量。
    """
    gt_by_image = {}
    for ann in merged_gt['annotations']:
        # COCO 的 crowd / ignore 标注不计入常规 TP、FP、FN 统计。
        if ann.get('iscrowd', 0) or ann.get('ignore', 0):
            continue
        if ann.get('category_id') != 1:
            continue
        gt_by_image.setdefault(ann['image_id'], []).append(ann['bbox'])

    dt_by_image = {}
    for det in coco_results:
        if det.get('category_id') != 1:
            continue
        dt_by_image.setdefault(det['image_id'], []).append(det)

    per_image = []
    total_tp = total_fp = total_fn = 0
    for image in merged_gt['images']:
        image_id = image['id']
        gt_boxes = np.asarray(gt_by_image.get(image_id, []), dtype=np.float32)
        if gt_boxes.size == 0:
            gt_boxes = gt_boxes.reshape(0, 4)

        # COCO maxDets 的语义是：每张图只评价 score 最高的 max_dets 个预测。
        detections = sorted(dt_by_image.get(image_id, []),
                            key=lambda item: item['score'], reverse=True)[:max_dets]
        matched_gt = np.zeros(len(gt_boxes), dtype=bool)
        tp = fp = 0

        for det in detections:
            dx, dy, dw, dh = det['bbox']
            if len(gt_boxes) == 0 or dw <= 0 or dh <= 0:
                fp += 1
                continue

            gx, gy, gw, gh = gt_boxes.T
            inter_w = np.maximum(0., np.minimum(dx + dw, gx + gw) -
                                 np.maximum(dx, gx))
            inter_h = np.maximum(0., np.minimum(dy + dh, gy + gh) -
                                 np.maximum(dy, gy))
            intersection = inter_w * inter_h
            union = dw * dh + gw * gh - intersection
            ious = np.divide(intersection, union, out=np.zeros_like(intersection),
                             where=union > 0)
            ious[matched_gt] = -1
            best_index = int(np.argmax(ious))
            if ious[best_index] >= iou_thr:
                matched_gt[best_index] = True
                tp += 1
            else:
                fp += 1

        fn = int((~matched_gt).sum())
        total_tp += tp
        total_fp += fp
        total_fn += fn
        per_image.append({
            'image_id': image_id,
            'file_name': image.get('file_name', ''),
            'gt': int(len(gt_boxes)),
            'predictions': int(len(detections)),
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': float(tp / (tp + fp)) if tp + fp else 0.0,
            'recall': float(tp / (tp + fn)) if tp + fn else 0.0,
        })

    metrics = {
        'iou_threshold': iou_thr,
        'max_dets': max_dets,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'precision': float(total_tp / (total_tp + total_fp))
                     if total_tp + total_fp else 0.0,
        'recall': float(total_tp / (total_tp + total_fn))
                  if total_tp + total_fn else 0.0,
        'per_image': per_image,
    }
    return metrics


def _print_bbox_precision_recall(metrics, show_per_image=False):
    """Print global BBox Precision/Recall; per-image details are optional."""
    print(f"\n  ========== BBox Precision / Recall @ IoU={metrics['iou_threshold']:.2f} ==========")
    print(f"    TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    if show_per_image:
        print("    Per-image: image | GT | Pred | TP | FP | FN | Precision | Recall")
        for row in metrics['per_image']:
            print(f"      {osp.basename(row['file_name']):<35} "
                  f"{row['gt']:>4} {row['predictions']:>5} {row['tp']:>4} "
                  f"{row['fp']:>4} {row['fn']:>4} "
                  f"{row['precision']:>9.4f} {row['recall']:>7.4f}")


def run_evaluation(merged_gt, coco_results, save_path=None, max_dets=1000,
                   bbox_pr_iou=0.5, verbose=False):
    """运行 COCO 评估"""
    # 写入临时 GT 文件
    gt_path = save_path + '_gt.json' if save_path else '_tmp_gt.json'
    with open(gt_path, 'w') as f:
        json.dump(merged_gt, f)

    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(coco_results)

    print("\n  ========== BBox AP ==========")
    coco_eval_bbox = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval_bbox.params.maxDets = [1, 10, max_dets]
    evaluate_coco_with_progress(coco_eval_bbox, 'BBox')
    coco_eval_bbox.accumulate()
    bbox_metrics = _dense_coco_metrics(coco_eval_bbox, max_dets)

    print("\n  ========== Segm AP ==========")
    coco_eval_segm = COCOeval(coco_gt, coco_dt, 'segm')
    coco_eval_segm.params.maxDets = [1, 10, max_dets]
    evaluate_coco_with_progress(coco_eval_segm, 'Segm')
    coco_eval_segm.accumulate()
    segm_metrics = _dense_coco_metrics(coco_eval_segm, max_dets)

    bbox_pr_metrics = bbox_precision_recall_at_iou(
        merged_gt, coco_results, iou_thr=bbox_pr_iou, max_dets=max_dets)
    _print_bbox_precision_recall(bbox_pr_metrics, show_per_image=verbose)
    if save_path:
        pr_path = save_path + f'_bbox_pr_iou{bbox_pr_iou:.2f}.json'
        with open(pr_path, 'w', encoding='utf-8') as f:
            json.dump(bbox_pr_metrics, f, indent=2, ensure_ascii=False)
        print(f"  BBox PR 明细已保存: {pr_path}")

    # 清理临时文件
    if osp.exists(gt_path):
        os.remove(gt_path)

    return {
        'bbox_mAP': bbox_metrics['mAP'],
        'bbox_mAP50': bbox_metrics['mAP50'],
        'bbox_mAP75': bbox_metrics['mAP75'],
        'bbox_AR': bbox_metrics['AR'],
        'bbox_precision_iou50': bbox_pr_metrics['precision'],
        'bbox_recall_iou50': bbox_pr_metrics['recall'],
        'bbox_tp_iou50': bbox_pr_metrics['tp'],
        'bbox_fp_iou50': bbox_pr_metrics['fp'],
        'bbox_fn_iou50': bbox_pr_metrics['fn'],
        'segm_mAP': segm_metrics['mAP'],
        'segm_mAP50': segm_metrics['mAP50'],
        'segm_mAP75': segm_metrics['mAP75'],
        'segm_AR': segm_metrics['AR'],
    }


# ==================== 主函数 ====================
def main(args):
    # ---- 初始化模型（仅在需要重新推理时进行）----
    model = None
    test_pipeline = None
    if args.aggregate:
        print('[1/4] 仅汇总模式：跳过模型加载和大图推理。')
    else:
        print(f"[1/4] 加载模型...")
        print(f"  Config: {args.config}")
        print(f"  Checkpoint: {args.checkpoint}")
        model = init_detector(args.config, args.checkpoint, device=args.device)
        print(f"  模型加载完成!")

        # 预构建测试 pipeline (避免每次推理重复构建)
        print(f"  预构建测试 pipeline...")
        test_pipeline = _build_test_pipeline(model)
        print(f"  Batch size: {args.batch_size}")
        print("  Cross-tile merge: boundary-instance stitching")
        print(f"  Final NMS: {args.nms_type}, IoU threshold: {args.nms_thr}")
        print(f"  Containment filter: {args.containment_thr}")
        print(f"  Verbose diagnostics: {args.verbose}")

    os.makedirs(args.out_dir, exist_ok=True)
    all_city_results = {}

    for city_name, test_stations in TEST_SETS.items():
        # With one selected city, --out-dir is its root (the common case). If
        # several cities are enabled, isolate each city under its own folder.
        city_out_dir = (args.out_dir if len(TEST_SETS) == 1
                        else osp.join(args.out_dir, city_name.lower()))
        os.makedirs(city_out_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"[2/4] 城市: {city_name} ({len(test_stations)} 个电站)")
        print(f"  输出目录: {city_out_dir}")
        print(f"{'='*60}")

        if args.aggregate:
            station_result_paths = {
                station_id: osp.join(city_out_dir, station_id,
                                     f'{city_name}_coco_results.json')
                for station_id, _ in test_stations
            }
            missing = [station_id for station_id, result_path in station_result_paths.items()
                       if not osp.exists(result_path)]
            if missing:
                print(f"  [WARNING] 以下电站缺少预测 JSON，无法汇总: {', '.join(missing)}")
                continue

            all_dir = osp.join(city_out_dir, 'all')
            os.makedirs(all_dir, exist_ok=True)
            print(f"\n[3/4] 合并并评估 {city_name}/all（只保存 summary.json）...")
            eval_result = evaluate_city_from_station_files(
                test_stations, args.test_root, station_result_paths)
            if eval_result is None:
                print(f"  [WARNING] {city_name}/all 无预测结果。")
                continue
            merged_gt, coco_results = eval_result
            if args.save_mask_geojson:
                print('  [INFO] --aggregate 模式只保存 summary.json，忽略 --save-mask-geojson。')
            city_metrics = run_evaluation(
                merged_gt, coco_results, save_path=None,
                max_dets=args.max_dets, verbose=False)
            if city_metrics is None:
                print(f"  [WARNING] {city_name}/all 评估失败。")
                continue
            _save_metrics(osp.join(all_dir, 'summary.json'), city_name, city_metrics)
            _print_metrics(f'{city_name}/all', city_metrics)
            all_city_results[city_name] = city_metrics
            continue

        station_metrics = {}
        station_result_paths = {}
        for station_index, (station_id, rel_path) in enumerate(test_stations, start=1):
            station_dir = osp.join(city_out_dir, station_id)
            os.makedirs(station_dir, exist_ok=True)
            station_summary_path = osp.join(station_dir, 'summary.json')
            station_result_path = osp.join(station_dir, f'{city_name}_coco_results.json')
            station_result_paths[station_id] = station_result_path

            if osp.exists(station_summary_path) and osp.exists(station_result_path):
                try:
                    station_metrics[station_id] = _load_cached_metrics(
                        station_summary_path, city_name)
                    print(f"\n  [{station_index}/{len(test_stations)}] 电站 {station_id}: "
                          f"已完成，跳过推理和评估。")
                    continue
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    print(f"\n  [CACHE INVALID] 电站 {station_id}: {exc}; 将重新处理。")

            ann_file = osp.join(args.test_root, rel_path, 'instances.json')
            with open(ann_file, 'r', encoding='utf-8') as f:
                gt = json.load(f)
            print(f"\n  [{station_index}/{len(test_stations)}] 电站 {station_id}: "
                  f"{len(gt['images'])} 张图像, {len(gt['annotations'])} 个标注")

            station_predictions = {}
            total_dets = 0
            t0 = time.time()
            for img_info in tqdm(gt['images'], desc=f'  Station {station_id}', ncols=80):
                img_path = osp.join(args.test_root, rel_path, 'images', img_info['file_name'])
                img_path = _resolve_image_path(img_path)
                if not osp.exists(img_path):
                    tqdm.write(f"    [SKIP] Not found: {img_path}")
                    continue
                preds = inference_single_image(
                    model, img_path, test_pipeline,
                    batch_size=args.batch_size, score_thr=args.score_thr,
                    nms_thr=args.nms_thr, nms_type=args.nms_type,
                    containment_thr=args.containment_thr, verbose=args.verbose)
                station_predictions[(station_id, img_info['id'])] = preds
                total_dets += len(preds['bboxes']) if len(preds['bboxes'].shape) > 1 else 0

            elapsed = time.time() - t0
            print(f"  电站 {station_id} 推理完成: {len(gt['images'])} 张图像, "
                  f"{total_dets} 个检测框, 耗时 {elapsed:.1f}s "
                  f"({elapsed/max(len(gt['images']), 1):.2f}s/img)")

            eval_result = evaluate_city(
                station_predictions, [(station_id, rel_path)], args.test_root)
            if eval_result is None:
                print(f"  [WARNING] 电站 {station_id} 无预测结果，未生成缓存。")
                continue
            station_gt, station_coco_results = eval_result
            if args.save_mask_geojson:
                save_mask_geojson_visualizations(
                    station_gt, station_coco_results,
                    osp.join(station_dir, 'mask_geojson'),
                    modes=args.geojson_mode, mask_iou_thr=args.geojson_iou_thr,
                    simplify_epsilon=args.geojson_simplify_epsilon)
            metrics = run_evaluation(
                station_gt, station_coco_results,
                save_path=osp.join(station_dir, f'{city_name}_preds'),
                max_dets=args.max_dets, verbose=args.verbose)
            if metrics is None:
                print(f"  [WARNING] 电站 {station_id} 评估失败，未生成缓存。")
                continue
            _atomic_json_dump(
                station_result_path, station_coco_results,
                indent=2, ensure_ascii=False)
            _save_metrics(station_summary_path, city_name, metrics)
            station_metrics[station_id] = metrics
            _print_metrics(f'{city_name}/{station_id}', metrics)
            print(f"  电站结果已保存: {station_dir}")

        print(f"\n[3/4] {city_name} 指定电站处理完成；未请求 all 汇总。")

    print(f"\n{'='*60}")
    if all_city_results:
        print("[4/4] all 汇总完成:")
        print(f"{'City':<12} {'BBox_mAP':>10} {'Segm_mAP':>10} {'BBox_mAP50':>12} "
              f"{'BBox_P@.5':>11} {'BBox_R@.5':>11}")
        print("-" * 76)
        for city_name, metrics in all_city_results.items():
            print(f"{city_name:<12} {metrics['bbox_mAP']:>10.4f} {metrics['segm_mAP']:>10.4f} "
                  f"{metrics['bbox_mAP50']:>12.4f} "
                  f"{metrics['bbox_precision_iou50']:>11.4f} "
                  f"{metrics['bbox_recall_iou50']:>11.4f}")
    else:
        print("[4/4] 电站评估完成。")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='大图切片推理 + COCO 评估 (光伏实例分割)')
    # 路径设置
    parser.add_argument('--config', type=str, 
                        default='configs/pv-detect/mask2former_r50_8xb2-lsj-50e_pv_panel.py',
                        help='mmdet 模型配置文件路径')
    parser.add_argument('--checkpoint', type=str, 
                        default='./checkpoints/mask2former.pth',
                        help='训练好的模型权重文件路径 (.pth)')
    parser.add_argument('--test-root', type=str, 
                        default='./data/test',
                        help='测试集根目录 (包含 NingBo/ShaoXing/ShenZhen 子目录)')
    parser.add_argument('--out-dir', type=str, 
                        default='./outputs/mask2former',
                        help='城市结果根目录；每个电站和 all 将创建为其子目录。')
    # 合并参数配置
    parser.add_argument('--score-thr', type=float, default=0.50,
                        help='置信度阈值 (default: 0.50)')
    parser.add_argument('--nms-type', choices=['mask', 'bbox'], default='bbox',
                        help='边缘拼接后的最终去重方式 (default: bbox).')
    parser.add_argument('--nms-thr', type=float, default=0.5,
                        help='最终 NMS 的 IoU 阈值 (default: 0.5)')
    parser.add_argument('--containment-thr', type=float, default=0.5,
                        help='合并/NMS 后删除被更大掩膜覆盖的比例,不比较置信度; 0 关闭。')
    # 推理设置
    parser.add_argument('--max-dets', type=int, default=5000,
                        help='Maximum retained detections per large image for COCO evaluation.')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='推理设备 (default: cuda:0)')
    parser.add_argument('--batch-size', type=int, default=27,
                        help='Batch 推理大小, 每张图同时送多个 patch 进 GPU (default: 27)')
    parser.add_argument('--aggregate', action='store_true',
                        help='不进行推理, 仅汇总生成 all/summary.json')

    # 输出参数
    parser.add_argument('--verbose', action='store_true',
                        help='显示逐图拼接/NMS/包含过滤及逐图 PR 诊断信息。')
    parser.add_argument('--save-mask-geojson', action='store_true',
                        help='保存预测、TP/FP/FN/TN 的掩膜 GeoJSON')
    parser.add_argument('--geojson-mode', nargs='+', default=['pred'],
                        choices=['pred', 'tp', 'fp', 'fn', 'tn','all'],
                        help='要保存的 GeoJSON:pred=全部预测;all=pred、tp、fp、fn、tn 全部保存。')
    parser.add_argument('--geojson-iou-thr', type=float, default=0.5,
                        help='区分 TP/FP/FN 时使用的掩膜 IoU 阈值 (default: 0.5)')
    parser.add_argument('--geojson-simplify-epsilon', type=float, default=1.0,
                        help='掩膜轮廓 GeoJSON 简化容差，单位像素；0 表示不简化 (default: 1.0)')
    args = parser.parse_args()
    main(args)
