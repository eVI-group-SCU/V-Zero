"""Generate nearest same-size negative crops without API calls."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from numbers import Integral
from pathlib import Path
import random


def intersection(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))


def choose(width, height, positive, index=0, seed=42):
    """Prefer disjoint boxes, then IoU <= .05; minimize squared center distance."""
    if len(positive) != 4 or not all(isinstance(v, Integral) and not isinstance(v, bool) for v in positive):
        raise ValueError('bbox_pixel must contain four integer pixel coordinates')
    px, py, x2, y2 = positive
    if not (0 <= px < x2 <= width and 0 <= py < y2 <= height):
        raise ValueError('Positive bbox must be nonempty and inside the original image')
    w, h = x2 - px, y2 - py
    # Equal-area boxes: I / (2*A - I) <= 1/20 iff 21*I <= 2*A.
    for cap in (0, (2 * w * h) // 21):
        best = best_key = None
        rng = random.Random(seed + index)
        for x in range(width - w + 1):
            ix = max(0, min(x + w, x2) - max(x, px))
            allowed = h if ix == 0 else min(h, cap // ix)
            if allowed >= h:
                ys = [py]
            else:
                upper = min(height - h, py - h + allowed)
                lower = max(0, py + h - allowed)
                ys = []
                if upper >= 0:
                    ys.append(min(py, upper))
                if lower <= height - h:
                    ys.append(max(py, lower))
            for y in ys:
                box = [x, y, x + w, y + h]
                if intersection(positive, box) > cap:
                    continue
                key = ((x - px)**2 + (y - py)**2, rng.random())
                if best_key is None or key < best_key:
                    best, best_key = box, key
        if best is not None:
            return best
    raise ValueError('No same-size negative box fits with IoU <= 0.05')


def image_path(value, image_root):
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError('Expected exactly one image path per image column')
    item = value[0]
    path = item.get('image') if isinstance(item, dict) else item
    if not isinstance(path, str) or not path:
        raise ValueError('Expected a nonempty image path; embedded images are unsupported')
    path = Path(path)
    return path if path.is_absolute() else image_root / path


def generate(dataset, image_root, output, workers=4, seed=42):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image, ImageChops

    if workers < 1:
        raise ValueError('workers must be positive')
    table = pq.read_table(dataset)
    required = {'images', 'teacher_images', 'extra_info', 'teacher_neg_images'}
    if not required.issubset(table.column_names) or table.num_rows == 0:
        raise ValueError('Nonempty Parquet requires ' + ', '.join(sorted(required)))
    # Refuse to overwrite successful or partial outputs.
    output.mkdir(parents=True, exist_ok=False)
    crops = output / 'crops'
    crops.mkdir()

    def process(pair):
        index, row = pair
        try:
            extra = row['extra_info']
            if isinstance(extra, str):
                extra = json.loads(extra)
            positive = extra['bbox_pixel']
            with Image.open(image_path(row['images'], image_root)) as original:
                original = original.convert('RGB')
                width, height = original.size
                box = choose(width, height, positive, index, seed)
                size = (positive[2] - positive[0], positive[3] - positive[1])
                with Image.open(image_path(row['teacher_images'], image_root)) as pos:
                    pos.load()
                    if pos.size != size:
                        raise ValueError('Positive bbox size differs from positive crop image')
                crop = original.crop(box)
                path = crops / f'nearby_{index:06d}.png'
                crop.save(path)
                with Image.open(path) as saved:
                    saved.load()
                    if saved.size != size or ImageChops.difference(saved.convert('RGB'), crop).getbbox():
                        raise ValueError('Saved crop failed pixel verification')
            inter = intersection(positive, box)
            return {'global_index': index, 'ok': True, 'positive_bbox': positive,
                    'negative_bbox': box, 'point_pixel': [(box[0]+box[2])//2, (box[1]+box[3])//2],
                    'iou': inter / (2 * size[0] * size[1] - inter),
                    'crop_size': list(size), 'image_size': [width, height], 'crop_path': str(path.resolve())}
        except Exception as exc:
            return {'global_index': index, 'ok': False, 'error': str(exc)}

    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool, (output / 'grounding.jsonl').open('w') as stream:
        for record in pool.map(process, enumerate(table.to_pylist())):
            records.append(record)
            stream.write(json.dumps(record) + '\n')
            stream.flush()
    successes = [r for r in records if r['ok']]
    summary = {'rows': table.num_rows, 'ok': len(successes), 'failed': len(records)-len(successes),
               'api_calls': 0, 'seed': seed, 'source_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(),
               'zero_iou': sum(r['iou'] == 0 for r in successes),
               'max_iou': max((r['iou'] for r in successes), default=None),
               'semantic_validation': 'Geometry only; answer evidence is not checked',
               'changed_columns': ['teacher_neg_images', 'extra_info (negative crop metadata only)']}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    if summary['failed']:
        raise ValueError('Failed rows recorded in grounding.jsonl; no training Parquet exported')
    # Preserve other image fields/metadata in each negative-image entry, if present.
    negatives = []
    for old, record in zip(table['teacher_neg_images'].to_pylist(), records):
        if not isinstance(old, list) or len(old) != 1 or not isinstance(old[0], dict) or 'image' not in old[0]:
            raise ValueError('teacher_neg_images must be a one-element list of image dictionaries')
        negatives.append([dict(old[0], image=record['crop_path'])])
    negative_field = table.schema.field('teacher_neg_images')
    result = table.set_column(table.schema.get_field_index(negative_field.name), negative_field, pa.array(negatives, type=negative_field.type))
    # Keep negative metadata consistent instead of retaining annotations for the old crop.
    metadata = []
    updates = []
    for old, record in zip(table['extra_info'].to_pylist(), records):
        extra = json.loads(old) if isinstance(old, str) else dict(old)
        patch = {'teacher_neg_crop_image': record['crop_path'],
                 'teacher_neg_bbox_pixel': record['negative_bbox'],
                 'teacher_neg_bbox_2d': [min(999, round(v * 1000 / record['image_size'][i % 2]))
                                         for i, v in enumerate(record['negative_bbox'])],
                 'teacher_neg_crop_status': 'geometric_nearby',
                 'teacher_neg_crop_error': None, 'teacher_neg_target_iou': 0.05}
        # Preserve unrelated source metadata, including the positive bbox.
        extra.update(patch)
        metadata.append(extra)
        updates.append(patch)
    extra_type = table.schema.field('extra_info').type
    if pa.types.is_string(extra_type) or pa.types.is_large_string(extra_type):
        extra_array = pa.array([json.dumps(e) for e in metadata], type=extra_type)
    elif pa.types.is_struct(extra_type):
        inferred = pa.array(metadata).type
        fields = list(extra_type)
        names = {f.name for f in fields}
        for field in inferred:
            if field.name not in names:
                fields.append(field)
        # Previously null fields must acquire the type of the new metadata.
        fields = [pa.field(f.name, inferred.field(f.name).type) if pa.types.is_null(f.type)
                  else f for f in fields]
        extra_array = pa.array(metadata, type=pa.struct(fields))
    else:
        raise ValueError('extra_info must be a struct or JSON string')
    result = result.set_column(result.schema.get_field_index('extra_info'), 'extra_info', extra_array)
    temporary = output / 'train.parquet.tmp'
    pq.write_table(result, temporary)
    checked = pq.read_table(temporary)
    if checked.schema != result.schema or checked.num_rows != table.num_rows:
        raise ValueError('Export schema or row count changed')
    for name in table.column_names:
        if name not in (negative_field.name, 'extra_info') and not table[name].equals(checked[name]):
            raise ValueError('Unexpected change in source column: ' + name)
    if not checked[negative_field.name].equals(result[negative_field.name]):
        raise ValueError('Negative paths changed during export')
    for old, new, patch in zip(table['extra_info'].to_pylist(), checked['extra_info'].to_pylist(), updates):
        old = json.loads(old) if isinstance(old, str) else old
        new = json.loads(new) if isinstance(new, str) else new
        if any(new.get(k) != v for k, v in old.items() if k not in patch):
            raise ValueError('Unrelated metadata changed')
        if any(new.get(k) != v for k, v in patch.items()):
            raise ValueError('Negative metadata changed during export')
    temporary.rename(output / 'train.parquet')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--image-root', type=Path, help='Base for source image paths; defaults to dataset directory')
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory; existing directories are refused')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    print(json.dumps(generate(dataset, (args.image_root or dataset.parent).resolve(),
                              args.output_dir.resolve(), args.workers, args.seed), indent=2))


if __name__ == '__main__':
    main()
