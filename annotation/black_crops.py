"""Replace negative evidence with same-size black crops, preserving all rows."""
import argparse
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    table = pq.read_table(args.dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    crops = args.output_dir / 'crops'
    crops.mkdir(exist_ok=True)
    negatives = []
    sizes = {}
    for row in table.to_pylist():
        positives = row['teacher_images']
        if len(positives) != 1:
            raise ValueError('Expected exactly one positive crop')
        for column in ('images', 'teacher_images'):
            for item in row[column]:
                if not Path(item['image']).is_file():
                    raise FileNotFoundError(item['image'])
        with Image.open(positives[0]['image']) as positive:
            size = positive.size
        if size not in sizes:
            path = (crops / f'{size[0]}x{size[1]}.png').resolve()
            Image.new('RGB', size, (0, 0, 0)).save(path)
            with Image.open(path) as black:
                if black.size != size or black.getextrema() != ((0, 0),) * 3:
                    raise ValueError('Black crop verification failed')
            sizes[size] = str(path)
        negatives.append([{'image': sizes[size]}])
    index = table.schema.get_field_index('teacher_neg_images')
    if index < 0:
        raise ValueError('Source is missing teacher_neg_images')
    result = table.set_column(index, table.schema.field(index),
                              pa.array(negatives, type=table.schema.field(index).type))
    output = args.output_dir / 'train.parquet'
    pq.write_table(result, output)
    checked = pq.read_table(output)
    for name in table.column_names:
        if name != 'teacher_neg_images' and not checked[name].equals(table[name]):
            raise ValueError(f'Unexpected change to {name}')
    manifest = {'source': str(args.dataset.resolve()),
                'source_sha256': hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                'rows': table.num_rows, 'unique_crop_sizes': len(sizes),
                'negative': 'pure black RGB, same dimensions as positive crop',
                'changed_columns': ['teacher_neg_images']}
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
