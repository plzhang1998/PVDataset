"""Strict, resumable COCO union-mask export and foreground pixel evaluation.

Uses the user's converter and evaluator functions; never runs model inference.
Each station is a separate worker. Only owned mask/eval outputs are replaced.
"""
import argparse
import concurrent.futures
import csv
import gc
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
from pycocotools import mask as mu

MODELS = ['mask-rcnn', 'mask2former', 'htc', 'solov2']
VERSION = 'coco-union-pixel-v1'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def atomic_text(path, text):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(text, encoding='utf-8-sig')
    os.replace(tmp, path)


def own_dir(folder, signature):
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / '_job.json'
    if marker.exists():
        if json.loads(marker.read_text()) != signature:
            raise RuntimeError(f'Output provenance differs; refusing overwrite: {folder}')
    elif any(folder.iterdir()):
        raise RuntimeError(f'Nonempty unowned output directory: {folder}')
    else:
        dump(marker, signature)


def images_checked(data):
    images = data['images']
    ids = set()
    names = set()
    for im in images:
        key = str(im['id'])
        name = Path(im['file_name'].replace('\\', '/')).with_suffix('.png').name
        if key in ids or name in names:
            raise ValueError('Duplicate image ID or output basename')
        if int(im['height']) <= 0 or int(im['width']) <= 0:
            raise ValueError('Invalid dimensions')
        ids.add(key)
        names.add(name)
        im['_mask_name'] = name
    return images


def union_mask(annotations, im, converter):
    h, w = int(im['height']), int(im['width'])
    rles = []
    for ann in annotations:
        if ann.get('category_id') != 1:
            raise ValueError('Unexpected category; no annotations silently filtered')
        seg = ann['segmentation']
        if seg == []:
            continue
        rle = converter.segmentation_to_rle(seg, h, w)
        if list(rle['size']) != [h, w]:
            raise ValueError(f'RLE dimension mismatch for {im["id"]}')
        rles.append(rle)
    if rles:
        arr = mu.decode(mu.merge(rles))
        if arr.shape != (h, w) or not np.all((arr == 0) | (arr == 1)):
            raise ValueError('Invalid decoded union mask')
        return arr.astype(np.uint8, copy=False) * 255
    return np.zeros((h, w), dtype=np.uint8)


def save_mask(path, arr):
    tmp = path.with_name(path.name + '.tmp')
    Image.fromarray(arr).save(tmp, format='PNG', compress_level=1)
    os.replace(tmp, path)


def read_binary(path, shape):
    with Image.open(path) as im:
        if im.mode != 'L' or (im.height, im.width) != shape:
            raise ValueError(f'Invalid mask format/size: {path}')
        a = np.asarray(im).copy()
    if not np.all((a == 0) | (a == 255)):
        raise ValueError(f'Nonbinary pixels: {path}')
    return a > 0


def export(source, gt_path, images, annotations, out_dir, converter, evaluator,
           code_hashes, eval_dir=None, gt_mask_dir=None):
    start = time.monotonic()
    source_hash = sha(source)
    signature = dict(version=VERSION, source=str(source), source_sha256=source_hash,
                     gt_json=str(gt_path), gt_sha256=sha(gt_path), code_hashes=code_hashes,
                     category_id=1, extra_score_filter=None, foreground_value=255,
                     semantics='union of all existing instance masks; original resolution; all images included')
    own_dir(out_dir, signature)
    if eval_dir is not None:
        own_dir(eval_dir, signature)
    manifest_path = out_dir / '_conversion_manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        valid = old['signature'] == signature and len(old['images']) == len(images)
        valid = valid and all((out_dir / r['name']).is_file() and sha(out_dir / r['name']) == r['png_sha256'] for r in old['images'])
        if not valid:
            raise RuntimeError(f'Completed output modified: {out_dir}')
        if eval_dir is None:
            print(f'REUSE_VALIDATED {out_dir}', flush=True)
            return old
        if (eval_dir / 'summary.json').exists():
            summary = json.loads((eval_dir / 'summary.json').read_text())
            if summary['signature'] != signature or summary['evaluated_images'] != len(images):
                raise RuntimeError('Invalid existing evaluation')
            if sha(eval_dir / 'per_image.csv') != summary['per_image_csv_sha256']:
                raise RuntimeError('Evaluation CSV modified')
            print(f'REUSE_VALIDATED {eval_dir}', flush=True)
            return summary
    by_image = {str(im['id']): [] for im in images}
    for ann in annotations:
        key = str(ann['image_id'])
        if key not in by_image:
            raise ValueError(f'Unknown annotation image_id={key} in {source}')
        if 'segmentation' not in ann:
            raise ValueError('Missing segmentation; refusing to skip')
        if 'score' in ann and not math.isfinite(float(ann['score'])):
            raise ValueError('Nonfinite score')
        by_image[key].append(ann)
    n_ann = len(annotations)
    annotations.clear()
    total = dict(tp=0, fp=0, tn=0, fn=0)
    rows, records = [], []
    empty = 0
    for i, im in enumerate(images, 1):
        shape = int(im['height']), int(im['width'])
        name = im['_mask_name']
        anns = by_image.pop(str(im['id']))
        arr = union_mask(anns, im, converter)
        foreground = int(np.count_nonzero(arr))
        empty += foreground == 0
        path = out_dir / name
        save_mask(path, arr)
        del arr, anns
        records.append(dict(image_id=im['id'], file_name=im['file_name'], name=name,
                            height=shape[0], width=shape[1], foreground_pixels=foreground,
                            png_sha256=sha(path)))
        if eval_dir is not None:
            pred = read_binary(path, shape)
            label = read_binary(gt_mask_dir / name, shape)
            counts = evaluator.region_counts(label, pred)
            if sum(counts.values()) != shape[0] * shape[1]:
                raise AssertionError('Pixel count mismatch')
            for k in total:
                total[k] += counts[k]
            rows.append(dict(image_id=im['id'], image=name, height=shape[0], width=shape[1],
                             **counts, **evaluator.region_metrics(counts)))
            del pred, label
        if i % 25 == 0 or i == len(images):
            print(f'PROGRESS {source.parent} {i}/{len(images)} seconds={time.monotonic()-start:.1f}', flush=True)
    expected_names = {x['_mask_name'] for x in images}
    if {p.name for p in out_dir.glob('*.png')} != expected_names:
        raise ValueError('Mask filename set mismatch')
    manifest = dict(signature=signature, image_count=len(images), instance_count=n_ann,
                    empty_masks=int(empty), elapsed_seconds=time.monotonic()-start, images=records)
    dump(manifest_path, manifest)
    if eval_dir is None:
        return manifest
    metrics = evaluator.region_metrics(total)
    assert all(math.isfinite(v) for v in metrics.values())
    assert abs(metrics['f1'] - metrics['dice']) < 1e-12
    expected_pixels = sum(int(x['height']) * int(x['width']) for x in images)
    assert sum(total.values()) == expected_pixels
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(eval_dir / 'per_image.csv', buf.getvalue())
    summary = dict(signature=signature, evaluated_images=len(images), total_pixels=expected_pixels,
                   region={**total, **metrics}, pixel_aggregation='sum counts over all images, then calculate metrics',
                   foreground='pv_panel; binary union; no instance matching',
                   boundary_evaluated=False, visualization_generated=False, unmatched_predictions=[], missing_predictions=[],
                   all_empty_convention='IoU=F1=1 only when both foreground masks are empty',
                   conversion_and_evaluation_seconds=time.monotonic()-start,
                   per_image_csv_sha256=sha(eval_dir / 'per_image.csv'))
    evaluator.save_statistics_txt(total, metrics, eval_dir / 'statistics.txt')
    dump(eval_dir / 'summary.json', summary)
    print(f'EVAL_COMPLETE {source.parent} IoU={metrics["iou"]:.8f} F1={metrics["f1"]:.8f}', flush=True)
    return summary


def station_job(gt_string, args, code_hashes):
    gt_path = Path(gt_string)
    city, station = gt_path.parent.parent.name, gt_path.parent.name
    converter = load_module('converter', Path(args.tools) / 'coco_predictions_to_binary_masks.py')
    evaluator = load_module('evaluator', Path(args.tools) / 'segmentation_evaluator.py')
    data = json.loads(gt_path.read_text())
    images = images_checked(data)
    if any(a.get('ignore', 0) or a.get('iscrowd', 0) for a in data['annotations']):
        raise ValueError('Crowd/ignore policy requires review')
    if data['categories'] != [{'id': 1, 'name': 'pv_panel', 'supercategory': 'object'}]:
        raise ValueError('Category metadata changed')
    export(gt_path, gt_path, images, data['annotations'], gt_path.parent / 'mask',
           converter, evaluator, code_hashes)
    del data
    gc.collect()
    completed = []
    for model in MODELS:
        pred = Path(args.out_root) / model / city.lower() / station / f'{city}_coco_results.json'
        predictions = json.loads(pred.read_text())
        if not isinstance(predictions, list):
            raise ValueError('Expected COCO prediction list')
        summary = export(pred, gt_path, images, predictions, pred.parent / 'mask', converter,
                         evaluator, code_hashes, pred.parent / 'eval', gt_path.parent / 'mask')
        completed.append(dict(model=model, city=city, station=station,
                              eval_path=str(pred.parent / 'eval'), **summary))
        del predictions
        gc.collect()
    return completed


def self_test(args):
    converter = load_module('converter', Path(args.tools) / 'coco_predictions_to_binary_masks.py')
    evaluator = load_module('evaluator', Path(args.tools) / 'segmentation_evaluator.py')
    im = dict(id=1, height=6, width=7)
    masks = []
    for start in (0, 2):
        a = np.zeros((6, 7), np.uint8)
        a[start:start+3, 1:5] = 1
        masks.append(a)
    anns = [dict(category_id=1, segmentation=mu.encode(np.asfortranarray(a))) for a in masks]
    assert np.array_equal(union_mask(anns, im, converter) > 0, masks[0] | masks[1])
    assert not union_mask([], im, converter).any()
    polygon = dict(category_id=1, segmentation=[[1, 1, 4, 1, 4, 4, 1, 4]])
    assert np.count_nonzero(union_mask([polygon], im, converter)) == 9
    c = evaluator.region_counts(np.array([[1,1],[0,0]], bool), np.array([[1,0],[1,0]], bool))
    assert c == dict(tp=1, fp=1, fn=1, tn=1)
    m = evaluator.region_metrics(c)
    assert m['iou'] == 1/3 and m['f1'] == .5
    for c in [dict(tp=0,fp=0,fn=0,tn=42), dict(tp=0,fp=0,fn=10,tn=32), dict(tp=0,fp=10,fn=0,tn=32)]:
        m = evaluator.region_metrics(c)
        assert m['f1'] == m['dice'] and all(math.isfinite(v) for v in m.values())
    print('SELF_TEST_PASSED union/RLE/polygon/overlap/empty/counts/IoU/F1', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tools', default=str(Path(__file__).resolve().parent))
    parser.add_argument('--test-root', default='./data/test')
    parser.add_argument('--out-root', default='./data/out')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--self-test-only', action='store_true')
    args = parser.parse_args()
    self_test(args)
    if args.self_test_only:
        return
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    code_hashes = {n: sha(Path(args.tools)/n) for n in ['coco_predictions_to_binary_masks.py', 'segmentation_evaluator.py']}
    code_hashes['pixel_eval_batch.py'] = sha(__file__)
    gts = sorted(Path(args.test_root).glob('*/*/instances.json'))
    if len(gts) != 10:
        raise ValueError('Expected ten station GT files')
    for g in gts:
        for model in MODELS:
            p = Path(args.out_root)/model/g.parent.parent.name.lower()/g.parent.name/f'{g.parent.parent.name}_coco_results.json'
            if not p.is_file():
                raise FileNotFoundError(p)
    dump(run_dir / 'run_manifest.json', dict(version=VERSION, args=vars(args), code_hashes=code_hashes,
                                           ground_truth_jsons=list(map(str,gts)), models=MODELS,
                                           expected_gt_masks=4799, expected_prediction_masks=19196,
                                           png_compression=1, memory_max_gib=64, memory_high_gib=48))
    all_results = []
    failures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(station_job, str(g), args, code_hashes):str(g) for g in gts}
        for future in concurrent.futures.as_completed(futures):
            try:
                all_results.extend(future.result())
                dump(run_dir / 'progress.json', dict(completed_evaluations=len(all_results), results=all_results, failures=failures))
            except Exception as exc:
                import traceback
                traceback.print_exc()
                failures.append(dict(gt=futures[future], error=repr(exc)))
                dump(run_dir / 'failures.json', failures)
    if failures:
        raise RuntimeError(f'{len(failures)} station jobs failed; completed outputs retained')
    assert len(all_results) == 40
    evaluator = load_module('evaluator', Path(args.tools) / 'segmentation_evaluator.py')
    pooled = {}
    for model in MODELS:
        subset = [r for r in all_results if r['model'] == model]
        assert sum(r['evaluated_images'] for r in subset) == 4799
        total = {k: sum(r['region'][k] for r in subset) for k in ['tp','fp','tn','fn']}
        pooled[model] = dict(evaluated_images=4799, **total, **evaluator.region_metrics(total))
    dump(run_dir / 'evaluation_summary.json', dict(results=all_results, pooled_by_model=pooled,
         aggregation='pixel micro aggregation: sum TP/FP/TN/FN; not mean station IoU'))
    print('ALL_40_PIXEL_EVALUATIONS_VALIDATED', flush=True)


if __name__ == '__main__':
    main()
