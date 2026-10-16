# Data-processing commands

Run from repository root in the GIS/preprocessing environment. Paths below are
examples. Input images/annotations are not included. Output to new locations.

## Crop images, then crop annotations using the existing tiles

```bash
python preprocess/crop.py --input-dir data/train/009 --output-dir data/train-crop/009 --extension .tif
python preprocess/crop_coco.py --input-json data/train/009/instances.json --patch-image-dir data/train-crop/009/images --output-json data/train-crop/009/instances.json
```

Default window: 512x512, stride 448, last window aligned to the image edge, scale=1.
Names are `source_col_row.tif` with zero-padded indices. Image cropping preserves
the original rule excluding windows with >=60% zero-valued channel samples;
`--crop-all` disables that rule. It is not a test for whether PV annotations exist.
Use the actual extension (`.JPG` if applicable); annotation cropping supports the
existing patch files. Annotation clipping uses true geometric intersection and
filters rasterized bbox width OR height <5 pixels. Check missing/extra tile reports.

## Review conversions

```bash
python preprocess/shp2isat.py --shp-dir private_inputs/station/shp --image-dir data/station/images --output-dir outputs/station/ISAT
# After manual review in ISAT:
python preprocess/isat2coco.py --isat-dir outputs/station/ISAT --image-dir data/station/images --out-json outputs/station/instances.json
# Alternative: final SHP directly to COCO:
python preprocess/shp2coco.py --shp-dir private_inputs/station/shp --image-dir data/station/images --out-json outputs/station/instances_from_shp.json
```

SHP inputs here must already be image-plane pixel coordinates, not geographic
coordinates. SHP conversion defaults to y=-y; use `--no-flip-y` for y-down input.
ISAT conversion uses each object as one instance and does NOT merge objects merely
because their group numbers match. Invalid/degenerate objects are skipped and
reported by ISAT-to-COCO. Rasterized mask area/bbox and 5px filtering are retained.
Do not apply the old group-merging instructions to these scripts.

The primary review workflow is projected SHP → ISAT → manual review → COCO.
Direct SHP-to-COCO conversion is an alternative for already-reviewed SHP labels;
it does not replace manual quality control.

## Private-input data generation

```bash
python preprocess/reproject.py --ground-shp private_inputs/pv.shp --pose-csv private_inputs/pose.csv --image-dir private_inputs/images --output-dir outputs/projected_shp --width 5280 --height 3956 --focal 3700 --pixel-size 1 --cx 2640 --cy 1978
```

**The numeric camera parameters in this command are illustrative, not dataset
calibration values.** Supply the real values for the specific undistorted images.
Pose CSV uses `Image Name,Xs,Ys,Zs,Phi,Omega,Kappa`; angle units default to radians.
Prepare this CSV externally; camera-export conversion is not included. The first
column identifies the image, followed by camera-center coordinates and the three
orientation angles in the stated order. Use `--angle-unit degree` for degrees.
Coordinate systems, height datums and the camera orientation convention must
match the projection implementation. Focal length and pixel size must use the
same unit; cx/cy are pixel coordinates for the supplied undistorted images.

Projection does not perform automatic occlusion reasoning. Manual review remains
necessary. Full 3D data/camera exports are not distributed, so the published 2D
dataset alone cannot replay the complete generation process.
Undistortion code is pending a separate author-supplied file; no replacement was
invented during packaging.
