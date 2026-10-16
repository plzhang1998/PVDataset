# UAV photovoltaic module instance-segmentation benchmark

**Release candidate — author review required before public release.**

[中文说明](README_zh-CN.md)

This repository contains selected data-processing scripts and benchmark tools for
UAV photovoltaic **module-level instance segmentation**. It is not a complete copy
of MMDetection or SAM 3. Install those frameworks separately; see [setup](docs/SETUP.md).

Supervised baselines use **MMDetection 3.2.0**. Zero-shot evaluation uses the
**official SAM 3 implementation**; its exact upstream commit is not recorded here
yet. The framework directories contain project-specific configs/tools, not full
framework distributions. Code defaults are the author's benchmark settings.
SAM 3 uses the fixed text prompt `solar panel`, selected through preliminary
comparison on 10 training images, with no model training or fine-tuning.

本目录为独立整理的开源候选版，原始工作目录未修改。尚未确定自编代码许可证，
尚未补入去畸变脚本、最终数据 DOI、模型权重链接和服务器环境锁定文件。
请先阅读 [发布前检查清单](docs/RELEASE_CHECKLIST.md)，不要直接视为已完成授权的正式开源版本。

## Scope

- `preprocess/`: 3D polygon reprojection, SHP–ISAT–COCO conversion, direct SHP-to-COCO conversion, and image/COCO cropping. Reprojection requires private inputs not included here.
- `mmdetection/configs/pv-detect/`: Mask R-CNN, HTC (without semantic branch), SOLOv2 and Mask2Former configurations.
- `mmdetection/tools/`: training/testing entry points, large-image inference, station/city evaluation,
  GeoJSON export and foreground/boundary pixel evaluation from instance predictions.
- `sam3/`: the project's text-prompt zero-shot large-image evaluation script.
- `third_party/`: retained upstream licence texts.
- `provenance/`: selected source-file hashes, selection decisions and validation reports.

MMSegmentation, neural-network weights, image data, caches, build outputs, backup
scripts, standalone debug runs and server-specific service launchers are not included.

## Data

Data download: **not yet published — add the final repository record/DOI here.**
The planned release includes undistorted images and COCO `instances.json` per station.
3D models, camera parameters and photogrammetry project files are not bundled.
For the current evaluation scripts use:

```text
data/
  train-crop/009/{images/, instances.json}
  ...
  test-crop/013/{images/, instances.json}  # current local validation configuration
  test/
    NingBo/013/{images/, instances.json}
    ...
    ShaoXing/004/{images/, instances.json}
    ...
    ShenZhen/031/{images/, instances.json}
```

The planned station-level split is 12 Ningbo training sites, 6 Ningbo test sites,
3 Shaoxing and 1 Shenzhen cross-city test sites. The final downloadable station
manifest still needs confirmation against the published data. Do not infer split
membership from a filename alone.

One annotated object corresponds to one physical PV module, not an internal cell
or an entire array. Record the partial-occlusion annotation convention in the data
README. COCO coordinates are pixels, x rightward/y downward; inspection GeoJSON
uses y=-y and is **not georeferenced longitude/latitude**.

## Quick links

- [Installation and paths](docs/SETUP.md)
- [Data preparation commands](docs/PREPROCESSING.md)
- [Benchmark commands and current defaults](docs/BENCHMARK.md)
- [Release notes / exclusions](docs/RELEASE_NOTES.md)
- [Items requiring author confirmation](docs/RELEASE_CHECKLIST.md)
- [Third-party attribution](THIRD_PARTY_NOTICES.md)

## File integrity

`SHA256SUMS.txt` records each release file's SHA-256 checksum (excluding itself,
Git metadata and Python caches). It is optional for running the tools. Use it to
detect changed or corrupted files; it is not proof of scientific correctness or
publisher identity. Regenerate it after final edits. `provenance/release_inventory.json`
lists the current payload files; validation records explicitly distinguish checks
performed on this revision from historical results.

## Licence and citation

Author-written additions: **licence pending author approval**. Public visibility
alone does not grant an open-source licence. Do not replace upstream copyright
notices with a project-wide licence.

MMDetection-derived files retain the upstream Apache-2.0 notices. SAM materials
are governed by the separate SAM licence; no SAM model weights are redistributed.
Data licensing is separate from code licensing. Manuscript citation and DOI are
pending; no placeholder DOI should be treated as real.
