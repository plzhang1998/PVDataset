# Data-processing commands

Run from repository root in the GIS/preprocessing environment. Paths below are
examples. Input images/annotations are not included. Output to new locations.

## Crop images, then crop annotations using the existing tiles

```bash
python preprocess/crop.py --input-dir data/train/009 --output-dir data/train-crop/009 --extension .tif
python preprocess/crop_coco.py --input-json data/train/009/instances.json --patch-image-dir data/train-crop/009/images --output-json data/train-crop/009/instances.json
```

Default window: 512x512, stride 448, last window aligned to the image edge, scale=1.
Names are `source_col_row.tif` with zero-padded indices. Image cropping excludes
windows with >=60% zero-valued channel samples;
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
reported by ISAT-to-COCO. Area and bbox are computed from rasterized masks;
instances with bbox width or height below 5 pixels are filtered.

The primary review workflow is projected SHP → ISAT → manual review → COCO.
Direct SHP-to-COCO conversion is an alternative for already-reviewed SHP labels;
it does not replace manual quality control.

## Reprojection with external 3D annotations and camera parameters

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
necessary. The planned data release contains only 2D images and annotations;
3D data and camera exports are not included. Reprojection is therefore optional
for users of the 2D dataset and requires separately supplied inputs.
The undistortion script is not currently included. Supply undistorted images and
their corresponding camera parameters when using the projection tool.
