# Benchmark usage and current defaults

The author confirms that the code defaults are the settings used for the benchmark;
the manuscript will also describe them.
Explicit command-line overrides take precedence over defaults and must be recorded.

## Supervised models (from mmdetection/)

```bash
python tools/train.py configs/pv-detect/mask-rcnn_r50_fpn_1x_pv_panel.py --work-dir ../outputs/train_mask_rcnn
# Linux, 4 GPUs; the current model configs use batch_size=8 per GPU:
bash tools/dist_train.sh configs/pv-detect/mask-rcnn_r50_fpn_1x_pv_panel.py 4 --work-dir ../outputs/train_mask_rcnn
# Note: this local test.py uses named options, not upstream positional arguments:
python tools/test.py --config configs/pv-detect/mask-rcnn_r50_fpn_1x_pv_panel.py --checkpoint ../checkpoints/mask_rcnn.pth --work-dir ../outputs/tile_test
python tools/big_image_inference_eval.py --config configs/pv-detect/mask-rcnn_r50_fpn_1x_pv_panel.py --checkpoint ../checkpoints/mask_rcnn.pth --test-root ../data/test --out-dir ../outputs/mask_rcnn
python tools/big_image_inference_eval.py --aggregate --test-root ../data/test --out-dir ../outputs/mask_rcnn
```

Select stations by editing `TEST_SETS` in the large-image evaluation script, matching
the author's existing workflow. `--aggregate` only recomputes city/all metrics from
existing station predictions and saves summary.json; it is not a mean of station AP.
Never aggregate partial/different-setting runs as if they were one experiment.

| Setting | Current large-image evaluation default |
|---|---|
| Tile / stride | 512 / 448 |
| Score threshold | 0.50 |
| Stitch shared-region mask IoU | 0.20 |
| Final NMS type / threshold | bbox / 0.50 |
| Containment threshold | 0.50 |
| COCO maxDets | 5000 |
| MMDetection patch batch | 27 (adjust for available memory, document changes) |

Other stitching safeguards remain in source. Filtering at score=0.5 restricts the
candidate predictions available to AP computation; it is not merely a visualization
threshold. Keep settings explicit when comparing models.

The single-image GeoJSON tool defaults to **mask NMS**, unlike the evaluator's bbox
NMS. For matching the evaluator, explicitly select bbox and leave TTA disabled:

```bash
python tools/big_image_inference_geojson.py --image ../data/example.tif --config configs/pv-detect/mask-rcnn_r50_fpn_1x_pv_panel.py --checkpoint ../checkpoints/mask_rcnn.pth --out ../outputs/example.geojson --nms-type bbox
```

TTA options are exploratory and not automatically part of the main benchmark.

## Pixel coverage from instance results

```bash
python tools/coco_predictions_to_binary_masks.py --pred-json ../outputs/mask_rcnn/shaoxing/006/ShaoXing_coco_results.json --gt-json ../data/test/ShaoXing/006/instances.json --out-dir ../outputs/pred_masks
python tools/segmentation_evaluator.py --pred-dir ../outputs/pred_masks --label-dir ../outputs/gt_masks --out-dir ../outputs/pixel_metrics --no-vis
```

Supply correctly rasterized GT masks of the same dimensions. `pixel_eval_batch.py`
can generate both unions and run batches; inspect `--help` and its expected model
folder names before use. Boundary metrics depend on the tolerance settings and must
be reported explicitly. These are instance-result coverage metrics, not independent
semantic-model experiments.

## SAM 3 (from sam3/, separate Linux environment)

```bash
python big_image_inference_eval_sam3.py --checkpoint ../checkpoints/sam3.pt --test-root ../data/test --out-dir ../outputs/sam3_solar_panel --prompt "solar panel"
```

The SAM 3 runner uses a fixed text prompt, BF16 CUDA inference, tile batch=1,
native processor resolution 1008, default score=0.5, max_per_patch=100 and
maxDets=5000. It does not train or update model weights. `--stations` selects sites;
`--limit-images 2` is a smoke run, not full evaluation. `--resume` and `--eval-only`
require matching run manifests. Changing scripts/settings requires a new output
directory. Linux cgroup/systemd memory protection remains enabled.

The fixed text prompt is `solar panel`. It was selected through preliminary
comparison on 10 training images, then held fixed for test-set inference. No model
training or fine-tuning is performed. This is zero-shot inference with respect to
model weights, with task-specific prompt selection on training data; it is not a
claim of prompt selection without access to any dataset examples.
The main evaluation script is not a prompt-search tool. The image identifiers,
candidate prompts and comparison criterion are not included in this release yet.
Publication of the supervised models' final checkpoints is still under consideration.
