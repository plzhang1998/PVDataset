"""Synthetic-only regression checks. Run: python -m unittest discover -s tests -v"""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
from pycocotools import mask as mu
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'preprocess'))
from isat2coco import isat2coco
from coco2isat import coco2isat
from crop_coco import crop_coco_annotations, window_offsets
from convert_coco import geometry_to_coco
from reproject import project_points


class ReleaseTests(unittest.TestCase):
    def test_same_group_is_not_merged_and_small_invalid_filtered(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            (p/'images').mkdir()
            (p/'isat').mkdir()
            Image.new('RGB', (64,64)).save(p/'images/a.tif')
            def obj(points):
                return dict(category='pv_panel', group=118, segmentation=points)
            data = dict(info=dict(description='ISAT', name='a.tif', width=64, height=64), objects=[
                obj([[5,5],[15,5],[15,15],[5,15]]),
                obj([[25,5],[35,5],[35,15],[25,15]]),
                obj([[40,5],[44,5],[44,15],[40,15]]),
                obj([[5,25],[15,35],[5,35],[15,25]])])
            (p/'isat/a.json').write_text(json.dumps(data), encoding='utf-8')
            report=isat2coco(p/'isat',p/'images',p/'out.json',categories=['pv_panel'])
            coco=json.loads((p/'out.json').read_text(encoding='utf-8'))
            self.assertEqual(len(coco['annotations']),2)
            self.assertEqual(report['skipped_invalid_polygon_count'],1)
            self.assertEqual(report['skipped_small_bbox_count'],1)
            for ann in coco['annotations']:
                rle=mu.merge(mu.frPyObjects(ann['segmentation'],64,64))
                self.assertEqual(ann['area'],float(mu.area(rle)))
                np.testing.assert_array_equal(ann['bbox'],mu.toBbox(rle))
            coco2isat(p/'out.json',p/'images',p/'isat_roundtrip')
            isat2coco(p/'isat_roundtrip',p/'images',p/'roundtrip.json',categories=['pv_panel'])
            self.assertEqual(len(json.loads((p/'roundtrip.json').read_text())['annotations']),2)

    def test_shp_geometry_conversion_flip_and_threshold(self):
        ann,_=geometry_to_coco(Polygon([(5,-5),(15,-5),(15,-15),(5,-15)]),64,64,True,5,'error')
        self.assertEqual(ann['bbox'],[5.0,5.0,10.0,10.0])
        tiny,_=geometry_to_coco(Polygon([(5,5),(9,5),(9,15),(5,15)]),64,64,False,5,'error')
        self.assertIsNone(tiny)

    def test_image_and_annotation_windows(self):
        self.assertEqual(window_offsets(1024),[0,448,512])
        self.assertEqual(window_offsets(512),[0])
        self.assertEqual(window_offsets(511),[])
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            (p/'patches').mkdir()
            for col,x in enumerate(window_offsets(600)):
                for row,y in enumerate(window_offsets(600)):
                    Image.new('RGB',(512,512)).save(p/'patches'/f'a_{col:04d}_{row:04d}.tif')
            coco=dict(images=[dict(id=7,file_name='a.tif',width=600,height=600)],
                      categories=[dict(id=1,name='pv_panel')],
                      annotations=[dict(id=93,image_id=7,category_id=1,iscrowd=0,
                          segmentation=[[100,100,120,100,120,120,100,120]],area=400,bbox=[100,100,20,20])])
            (p/'input.json').write_text(json.dumps(coco),encoding='utf-8')
            crop_coco_annotations(p/'input.json',p/'output.json',patch_image_dir=p/'patches')
            out=json.loads((p/'output.json').read_text())
            self.assertEqual(len(out['images']),4)
            self.assertEqual(len(out['annotations']),4)
            self.assertEqual({a['area'] for a in out['annotations']},{400})

    def test_projection_center_and_y_flip(self):
        xyz=np.array([[0,0,0],[1,1,0]],dtype=float)
        pose=[0,0,10,0,0,0]
        pixels,valid=project_points(xyz,pose,100,1,50,50,flip_y=False)
        np.testing.assert_allclose(pixels,[[50,50],[60,40]])
        self.assertTrue(valid.all())
        flipped,_=project_points(xyz,pose,100,1,50,50,flip_y=True)
        np.testing.assert_allclose(flipped[:,1],-pixels[:,1])


if __name__=='__main__':
    unittest.main()
