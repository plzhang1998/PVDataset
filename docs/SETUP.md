# Installation and path setup

## Separate environments

Use separate environments for GIS preprocessing, MMDetection and SAM 3. The included
dependency list is a convenience list, not an exact reconstruction of the server.
Do not install every project's newest dependencies into one environment.

Preprocessing requires Python 3.9+ (BooleanOptionalAction), GDAL/osgeo for SHP and
image cropping, and packages listed in `requirements-preprocess.txt`. Install a
GDAL build compatible with Python and the native GIS libraries. The tests do not
install packages or modify the user's environments.
On Windows, activate the Conda environment before running scripts so its
`Library/bin` DLL search paths are present; invoking python.exe alone may fail
when NumPy/GDAL native dependencies load.

The supervised experiments use MMDetection **3.2.0**. Tools require MMEngine, MMCV with
compiled ops, PyTorch/torchvision, CUDA (for GPU inference), OpenCV and pycocotools.
Start from the author's known working environment or the upstream installation
guide: <https://github.com/open-mmlab/mmdetection/tree/v3.2.0>.
This repository is an overlay of project files, not a standalone `mmdet` package.
Framework-core differences have not been audited against that tag. Compare
against the actual training server before claiming exact reproducibility.

SAM 3 requires its own installed upstream framework, its compatible Python/PyTorch
environment and an officially obtained checkpoint. See <https://github.com/facebookresearch/sam3>.
The experiments use the official SAM 3 implementation. The exact upstream commit
has not yet been recorded; 'official' identifies the source, not an immutable version.
Do not substitute a new upstream commit and assume identical benchmark results.

## Working directories and data roots

For supervised-model commands, change into `mmdetection/`. Relative config paths
then resolve as expected. Edit `data_root` in
`configs/_base_/datasets/pv_concat_instance.py` to your absolute dataset directory,
with a trailing slash. The HTC config also has a `data_root` assignment: set it to
the same directory. Merely overriding a top-level `data_root` on the CLI does not
necessarily rewrite already-created nested dataset dictionaries.

The current dataset base preserves the source configuration: training uses the
12 station subdatasets and validation uses `test-crop/013`. This choice has not
been corrected or endorsed by packaging. Check `docs/RELEASE_CHECKLIST.md` before
describing held-out test results in a manuscript.

Always pass checkpoint, input and output paths explicitly in evaluation commands.
Use a NEW output folder when any data, weights, prompt or postprocessing setting
changes. MMDetection's station-cache logic does not comprehensively fingerprint
these inputs; an old output directory can silently reuse previous metrics.

## SAM 3 platform restriction

The preserved SAM runner uses Linux `/proc`, cgroup v2 and `systemd-run --user` to
enforce a RAM limit (default 64 GiB). It is not a Windows-native runner and may not
run in containers without this support. Packaging did not remove this safety guard.
Full GPU inference must be validated in the author's server environment.
