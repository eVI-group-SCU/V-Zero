import json
from pathlib import Path
import tempfile
import unittest

from annotation.geometric_nearby import choose, generate, intersection


class GeometryTest(unittest.TestCase):
    def test_exhaustive_small_images(self):
        for width in range(1, 7):
            for height in range(1, 7):
                for w in range(1, width + 1):
                    for h in range(1, height + 1):
                        boxes = [[x,y,x+w,y+h] for x in range(width-w+1) for y in range(height-h+1)]
                        for positive in boxes:
                            disjoint = [b for b in boxes if intersection(b, positive) == 0]
                            feasible = disjoint or [b for b in boxes if 21*intersection(b,positive) <= 2*w*h]
                            if not feasible:
                                with self.assertRaises(ValueError):
                                    choose(width,height,positive)
                            else:
                                actual = choose(width,height,positive)
                                distance = lambda b: (b[0]-positive[0])**2+(b[1]-positive[1])**2
                                self.assertIn(actual, feasible)
                                self.assertEqual(distance(actual), min(map(distance,feasible)))

    def test_partial_overlap_and_invalid_bbox(self):
        b = choose(39,20,[0,0,20,20])
        self.assertGreater(intersection(b,[0,0,20,20]),0)
        self.assertLessEqual(21*intersection(b,[0,0,20,20]),800)
        for box in ([0,0,0,1], [-1,0,2,2], [0,0,10,11], [0.0,0,1,1]):
            with self.assertRaises(ValueError): choose(10,10,box)

    def test_export_and_failure(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            Image.new('RGB',(39,20),'red').save(root/'image.png')
            Image.new('RGB',(20,20),'red').save(root/'positive.png')
            rows=[{'images':[{'image':'image.png'}], 'teacher_images':[{'image':'positive.png'}],
                   'teacher_neg_images':[{'image':'old.png'}],
                   'extra_info':json.dumps({'bbox_pixel':[0,0,20,20]}), 'question':'untouched'}]
            source=root/'source.parquet';table=pa.Table.from_pylist(rows);pq.write_table(table,source)
            summary=generate(source,root,root/'ok',2)
            self.assertEqual(summary['ok'],1)
            result=pq.read_table(root/'ok/train.parquet')
            for col in table.column_names:
                if col not in ('teacher_neg_images','extra_info'):self.assertTrue(table[col].equals(result[col]))
            with Image.open(result['teacher_neg_images'][0].as_py()[0]['image']) as image:
                self.assertEqual(image.size,(20,20))
            with self.assertRaises(FileExistsError):generate(source,root,root/'ok')
            Image.new('RGB',(21,20)).save(root/'positive.png')
            with self.assertRaises(ValueError):generate(source,root,root/'failed')
            self.assertFalse((root/'failed/train.parquet').exists())
            self.assertEqual(json.loads((root/'failed/summary.json').read_text())['failed'],1)


if __name__=='__main__':unittest.main()
