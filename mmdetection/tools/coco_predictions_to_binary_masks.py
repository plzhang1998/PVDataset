"""Rasterize COCO instance-segmentation predictions into binary PNG masks.

Each output PNG corresponds to one image in ``--gt-json``.  Pixel value 0 is
background and ``--foreground-value`` (255 by default) is the union of every
retained predicted instance mask.  Empty prediction images are intentionally
written as all-zero masks so that they are included in pixel-level evaluation.

Example
-------
python tools/coco_predictions_to_binary_masks.py \
    --pred-json ./out/detect/mask_rcnn/shaoxing/006/ShaoXing_coco_results.json \
    --gt-json ./Dataset/test/ShaoXing/006/instances.json \
    --out-dir ./out/detect/mask_rcnn/shaoxing/006/pred_binary_masks
"""

from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert COCO instance predictions to union binary masks.')
    parser.add_argument('--pred-json', required=True, help='COCO prediction JSON.')
    parser.add_argument('--gt-json', required=True,help='COCO instances JSON. Its images field supplies output sizes/names.')
    parser.add_argument('--out-dir', required=True, help='Directory for binary PNG masks.')
    parser.add_argument(
        '--score-thr', type=float, default=0.0,
        help='Keep predictions with score >= this value (default: 0.0).')
    parser.add_argument(
        '--category-id', type=int, default=None,
        help='Optional category ID to retain. By default all categories are merged.')
    parser.add_argument(
        '--foreground-value', type=int, default=255, choices=range(1, 256),
        metavar='[1,255]', help='Foreground pixel value (default: 255).')
    parser.add_argument(
        '--limit', type=int, default=None,
        help='Only convert the first N images; useful for a quick sanity check.')
    return parser.parse_args()


def segmentation_to_rle(segmentation: Any, height: int, width: int) -> dict[str, Any]:
    """Convert one COCO polygon/RLE segmentation to one RLE object.

    Keeping masks as RLE until all instances for an image are collected lets
    ``pycocotools`` union them in compiled code. This avoids repeatedly
    scanning a 20-megapixel image once per instance.
    """
    if isinstance(segmentation, dict):
        # Uncompressed RLE has a list of counts and needs conversion first;
        # compressed RLE can be decoded directly.
        rle = (mask_utils.frPyObjects(segmentation, height, width)
               if isinstance(segmentation.get('counts'), list) else segmentation)
    elif isinstance(segmentation, list):
        polygons = segmentation
        if polygons and isinstance(polygons[0], (int, float)):
            polygons = [polygons]
        rle = mask_utils.merge(mask_utils.frPyObjects(polygons, height, width))
    else:
        raise ValueError(f'Unsupported segmentation type: {type(segmentation).__name__}')

    # ``frPyObjects`` may return a list for an uncompressed RLE. Merge it so
    # callers always receive exactly one RLE per prediction.
    return mask_utils.merge(rle) if isinstance(rle, list) else rle


def output_name(file_name: str, image_id: Any, used_names: set[str]) -> str:
    """Create a stable PNG filename and avoid accidental basename collisions."""
    base_name = Path(file_name.replace('\\', '/')).with_suffix('.png').name
    if base_name not in used_names:
        used_names.add(base_name)
        return base_name

    name = f'{image_id}_{base_name}'
    used_names.add(name)
    return name


def main() -> None:
    args = parse_args()
    with Path(args.gt_json).open('r', encoding='utf-8') as file:
        gt_data = json.load(file)
    with Path(args.pred_json).open('r', encoding='utf-8') as file:
        predictions = json.load(file)

    if not isinstance(predictions, list):
        raise TypeError('--pred-json must contain a COCO results list.')

    images = gt_data.get('images')
    if not isinstance(images, list):
        raise KeyError('--gt-json has no valid images list.')

    predictions_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    retained_predictions = 0
    for prediction in predictions:
        if prediction.get('score', 0.0) < args.score_thr:
            continue
        if args.category_id is not None and prediction.get('category_id') != args.category_id:
            continue
        if 'segmentation' not in prediction:
            continue
        predictions_by_image[str(prediction.get('image_id'))].append(prediction)
        retained_predictions += 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()
    decoded_count = 0
    invalid_count = 0
    images_with_foreground = 0
    selected_images = images[:args.limit] if args.limit is not None else images
    total_images = len(selected_images)
    progress_interval = max(1, min(10, total_images // 100 or 1))
    started_at = perf_counter()

    print(f'Converting {total_images} images to binary masks...')

    for image_index, image in enumerate(selected_images, start=1):
        image_id = image['id']
        height, width = int(image['height']), int(image['width'])
        instance_rles = []
        for prediction in predictions_by_image.get(str(image_id), []):
            try:
                instance_rles.append(segmentation_to_rle(
                    prediction['segmentation'], height, width))
            except (TypeError, ValueError, KeyError) as error:
                invalid_count += 1
                print(f'[warning] Skip invalid mask: image_id={image_id}, '
                      f'prediction_id={prediction.get("id")}: {error}')
                continue

        if instance_rles:
            # Union all instances while still compressed, then rasterize the
            # final foreground mask once. This is the main performance path.
            merged_rle = mask_utils.merge(instance_rles)
            binary_mask = mask_utils.decode(merged_rle).astype(np.uint8, copy=False)
            binary_mask *= args.foreground_value
            decoded_count += len(instance_rles)
        else:
            binary_mask = np.zeros((height, width), dtype=np.uint8)

        if binary_mask.any():
            images_with_foreground += 1
        name = output_name(str(image.get('file_name', image_id)), image_id, used_names)
        Image.fromarray(binary_mask).save(out_dir / name, compress_level=9)

        if image_index % progress_interval == 0 or image_index == total_images:
            elapsed = perf_counter() - started_at
            rate = image_index / elapsed if elapsed > 0 else 0.0
            remaining = (total_images - image_index) / rate if rate > 0 else 0.0
            print(
                f'\r  Progress: {image_index}/{total_images} '
                f'({image_index / total_images:.1%}) | '
                f'Masks decoded: {decoded_count} | ETA: {remaining:.0f}s',
                end='', flush=True)

    print()

    print('\nConversion finished')
    print(f'  Output directory: {out_dir}')
    print(f'  Converted images: {len(selected_images)}')
    print(f'  Images with foreground: {images_with_foreground}')
    print(f'  Retained predictions: {retained_predictions}')
    print(f'  Decoded masks: {decoded_count}')
    print(f'  Invalid masks skipped: {invalid_count}')
    if args.limit is not None:
        print('  Note: --limit was used; this is not a complete export.')


if __name__ == '__main__':
    main()
