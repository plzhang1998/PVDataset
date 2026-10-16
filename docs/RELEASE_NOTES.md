# Candidate packaging notes — 2026-10-16

## What changed

Only a separate copy was edited. Personal paths were replaced by portable example
paths; preprocessing entry points gained explicit CLI arguments and new-output
guards. Projection focal length/principal point must now be supplied explicitly,
instead of silently using one station's calibration. Existing geometry, filtering,
stitching, NMS and evaluation function implementations were retained.

The four models come only from the author-designated `configs/pv-detect` directory.
Their recursive `_base_` dependencies are included. CONFIG_MAP fallback paths now
point to these configs. Original numeric training settings were not optimized.
The distributed test shell wrapper now passes named --config/--checkpoint options
to match the local test.py interface (instead of upstream positional arguments).

No `.git` history, images, weights, binary caches, build trees, old backups,
MMSegmentation or full third-party frameworks were copied. Old documentation was
not reused. Removed from this candidate only: the legacy `merge_coco_datasets.py`
which offsets IDs by counts and is unsafe for arbitrary non-contiguous IDs.
Original scripts remain available in the author's source directory.

## Scope limitations

This is a source/CLI packaging pass, not an algorithm correction or an experiment
rerun. Legacy functions not imported by selected tools were omitted, not deleted
from the original folder. See `provenance/selection.json` for decisions.
The missing undistortion script will be added by the author. Camera-pose CSV
preparation is external to the current candidate.

The source hash manifest uses paths relative to the original code directory and
does not embed the author's workstation root. Local MMDetection/SAM3 cores may have
changes beyond these entry points; audit against the training environment before
claiming an official upstream installation will reproduce all results.

## Author cleanup and documentation synchronization

The author consolidated projection into `preprocess/reproject.py`, renamed the
direct SHP converter to `preprocess/shp2coco.py`, and removed unused conversion
helpers and the separate data-generation directory. Documentation and release
manifests now follow the current tree. Test source is maintained by the author
and was not edited during this synchronization. Historical validation is retained
as historical evidence only; full functional tests were not rerun in this pass.
Any ZIP created before this cleanup is an older snapshot, not the current release.
