# PVDataset: UAV Photovoltaic Module Instance Segmentation

[English](README.md) | [简体中文](README_zh-CN.md)

PVDataset provides data-processing tools and benchmark code for **individual photovoltaic (PV) module instance segmentation in UAV imagery**. It supports annotation preparation, supervised and text-prompt zero-shot evaluation, and inference on large images.

The dataset covers 22 PV sites in Ningbo, Shaoxing and Shenzhen, China. UAV oblique imagery captures variations in module scale, orientation, installation height, occlusion, illumination and dense arrangement. Each instance represents one physical PV module—not an internal cell, an internal subdivision or an entire array.

## Dataset

| Region | Training sites | Test sites | Evaluation role |
|---|---:|---:|---|
| Ningbo | 12 | 6 | Within-city evaluation |
| Shaoxing | 0 | 3 | Cross-city generalization |
| Shenzhen | 0 | 1 | Cross-city generalization |

Data are partitioned by site. The planned data release consists of undistorted images and a COCO-format `instances.json` for each site. **Download links and a dataset DOI are not yet available.** Images, 3D models, camera parameters and model weights are not included in this code repository.

Annotations are prepared through 3D module delineation, reprojection to individual images and manual review. Partially occluded modules retain their projected outlines; fully occluded modules are removed. COCO coordinates use an upper-left origin, with x increasing rightward and y downward.

## Benchmarks

- **Supervised instance segmentation:** Mask R-CNN, HTC without a semantic branch, SOLOv2 and Mask2Former, using **MMDetection 3.2.0**.
- **Text-prompt zero-shot instance segmentation:** the **official SAM 3 implementation**, without training or fine-tuning. The fixed prompt `solar panel` was selected through preliminary comparison on 10 training images and is held fixed during testing.
- **Large-image evaluation:** tiled inference, boundary-instance merging, duplicate suppression and COCO bounding-box/mask AP evaluation, with station-level and aggregate results.
- **Pixel-level evaluation:** foreground and boundary metrics computed from the union of predicted instance masks, rather than separately trained semantic segmentation models.

Code defaults are the author's benchmark settings. Command-line overrides should be recorded when reporting results. The exact SAM 3 upstream commit and a locked server environment have not yet been documented; final supervised-checkpoint distribution is under consideration.

## Repository structure

```text
preprocess/                  Reprojection, annotation conversion and image/label cropping
mmdetection/
  configs/pv-detect/         Four supervised-model configurations
  configs/_base_/           Required base configurations
  tools/                    Training, inference, merging, export and evaluation
sam3/                       Text-prompt zero-shot inference and evaluation
tests/                      Synthetic preprocessing checks
docs/                       Installation and usage documentation
third_party/                Upstream license texts
provenance/                 File inventory, source mapping and validation records
```

The `mmdetection/` and `sam3/` directories contain project-specific tools/configurations, **not complete framework distributions**. Install the upstream frameworks separately.

## Getting started

1. Follow [installation and path setup](docs/SETUP.md). Use separate environments for preprocessing, MMDetection and SAM 3.
2. Prepare images and COCO annotations using the [preprocessing guide](docs/PREPROCESSING.md). The review workflow is **SHP → ISAT → manual review → COCO**; image and annotation cropping are provided separately.
3. Set dataset paths in the model configurations, including HTC's own `data_root` setting.
4. Follow the [benchmark guide](docs/BENCHMARK.md) for training, large-image inference, aggregation and pixel-level evaluation.

Example layout used by the evaluation tools:

```text
data/
  train-crop/009/{images/, instances.json}
  test-crop/013/{images/, instances.json}
  test/
    NingBo/013/{images/, instances.json}
    ShaoXing/004/{images/, instances.json}
    ShenZhen/031/{images/, instances.json}
```

This layout is illustrative, not a complete split manifest. The current validation configuration uses `test-crop/013`; see [setup notes](docs/SETUP.md) for the checkpoint-selection protocol requiring confirmation against the final experiments.

### Usage notes

- ISAT-to-COCO treats each annotation object as an instance; matching group IDs do not merge objects.
- Reprojection requires external 3D annotations and compatible camera parameters. Users of the released 2D data need not repeat this step. The undistortion script is not currently included.
- Inspection GeoJSON uses pixel coordinates with y negated, not geographic longitude/latitude.
- The single-image GeoJSON tool defaults to mask NMS; the large-image evaluator defaults to bounding-box NMS. Select matching settings when comparing outputs.
- The SAM 3 runner requires Linux cgroup/systemd support for its memory guard.
- Use a new output directory when changing weights, data, prompts or postprocessing settings to avoid reusing incompatible cached results.

Preprocessing checks can be run from the repository root after installing the required dependencies:

```bash
python -m unittest discover -s tests -v
```

Validation records in `provenance/validation.json` distinguish current checks from historical checks; they do not constitute a full GPU experiment reproduction.

## File integrity

`SHA256SUMS.txt` provides file checksums to detect changes or corruption. It is not required to run the code. `provenance/release_inventory.json` lists payload files; `source_manifest.json` records script provenance. Checksums exclude the checksum file itself, Git metadata and Python caches.

## License and citation

**A license for the author-written code has not yet been assigned.** Public availability does not by itself grant an open-source license. Third-party components remain subject to their respective licenses; see [third-party notices](THIRD_PARTY_NOTICES.md) and `third_party/`.

Dataset licensing will be specified with the data release. Publication details and a recommended citation will be added when available.

## Questions and feedback

Please use [GitHub Issues](https://github.com/plzhang1998/PVDataset/issues) to report problems or discuss the dataset and tools. Include the relevant command, framework versions and error message when reporting a technical issue.
