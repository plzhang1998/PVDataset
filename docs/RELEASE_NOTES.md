# Repository notes and update history

## Scope

PVDataset provides project-specific preprocessing, model configurations and
evaluation tools. MMDetection and SAM 3 are installed separately. Datasets,
checkpoints, full framework source, 3D models and camera exports are not bundled.
See [project status](RELEASE_CHECKLIST.md) for available resources and known
reproducibility limitations.

## 2026-10-16 — Public documentation

- Published code and bilingual project introductions on GitHub.
- Documented MMDetection 3.2.0 and the official SAM 3 implementation.
- Described the fixed `solar panel` prompt and its selection using 10 training images.
- Organized documentation around installation, preprocessing, evaluation and
  current resource availability.
- Synchronized the file inventory and SHA-256 checksums with documentation updates.

These documentation updates do not change model configurations, inference
defaults, annotation rules, merging algorithms or test source.

## Code organization

- `preprocess/reproject.py` projects 3D polygons using externally supplied camera
  parameters; focal length and principal point are explicit command-line inputs.
- `preprocess/shp2coco.py` provides direct conversion of reviewed SHP annotations.
- `preprocess/shp2isat.py` and `preprocess/isat2coco.py` support the manual review workflow.
- Four supervised-model configurations and their required base configurations
  are included under `mmdetection/configs/`.
- Training and evaluation entry points use portable example paths. The distributed
  test wrapper passes named `--config` and `--checkpoint` options to this repository's
  `test.py` interface.
- Camera-export conversion is external to this repository. The undistortion script
  is not currently included.

## Traceability

`provenance/source_manifest.json` maps included source files to their repository
paths. `selection.json` records scope decisions, and `validation.json` separates
historical checks from current checks. Historical paths in these records describe
earlier source layouts, not commands for the current repository.

`SHA256SUMS.txt` and `provenance/release_inventory.json` describe the current file
contents. An archive made from an older commit remains an older snapshot; obtain
the desired Git commit when comparing versions or reporting experiments.
