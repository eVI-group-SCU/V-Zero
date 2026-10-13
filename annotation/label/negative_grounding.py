"""Direct negative grounding using the positive annotation's bbox/point protocol."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import re
import time
from pathlib import Path

from .ark import call_ark
from .config import parse_args, validate_args
from .data import flatten_images, image_from_value, image_to_data_url, iter_examples, resolve_image_value
from .grounding import bbox_to_pixel, parse_grounding_response, point_to_pixel
from .output import base_record


def build_prompt(question, strategy, image_size=None, positive_size=None):
    if strategy == 'random':
        rule = (
            'Select an arbitrary unrelated region anywhere in Image 1, independently of the '
            'positive target location. Do not deliberately select its immediate surroundings. '
            'The region must be unrelated to the question and must not contain the positive '
            'target, another instance that answers the question, or any answer-bearing detail. '
            'Random means a model-chosen unrelated region, not uniform statistical sampling.'
        )
    elif strategy == 'nearby':
        rule = (
            'Locate the positive crop (Image 2) within Image 1. Select the closest suitable '
            'region immediately beside it, showing local surrounding context. Keep the box '
            'outside the positive region without overlapping it. Do not select a distant '
            'unrelated region. Exclude the target, other answer-bearing instances, and all '
            'details that would help answer the question.'
        )
    else:
        raise ValueError(strategy)
    size_rule = (
        f'Image 1 size: {image_size[0]} x {image_size[1]} pixels. '
        f'Positive crop size: {positive_size[0]} x {positive_size[1]} pixels. '
        f'The negative box MUST have exactly this width and height in original-image pixels. '
        f'Its normalized width is approximately {positive_size[0] * 1000 / image_size[0]:.3f} '
        f'and normalized height {positive_size[1] * 1000 / image_size[1]:.3f}. '
        'Only coordinate quantization rounding is allowed; do not choose a smaller or larger box.\n'
        if image_size and positive_size else
        'The negative box MUST have the same original-image pixel width and height as Image 2.\n'
    )
    return (
        'Annotate one negative visual region for the question.\n'
        'Image 1 is the ORIGINAL IMAGE. Image 2 is the POSITIVE CROP containing relevant '
        'evidence. Use both images and the question to distinguish positive from negative evidence.\n'
        f'Negative type: {strategy}. {rule}\n'
        + size_rule +
        'Requirements:\n'
        '1. Output one bounding box meeting the negative-region and fixed-size requirements. '
        'Select the location directly; no candidate IDs.\n'
        '2. Output one point at the center of the selected negative target or region, inside the box.\n'
        '3. Use normalized integer coordinates on a 1000 x 1000 canvas. The valid range is '
        '[0, 999], with the image top-left corner as the origin. BOTH bbox and point must refer '
        'to Image 1 (the original image), never Image 2.\n'
        '4. If the question depends on text, signs, numbers, or fine-grained local details, '
        'exclude those answer-bearing regions from the negative box.\n'
        '5. Strictly output only the following two lines. Do not explain. Do not output JSON:\n'
        '<bbox>x_min y_min x_max y_max</bbox>\n'
        '<point>x y</point>\n'
        '6. If no valid negative region exists, output <bbox>0 0 0 0</bbox> and '
        '<point>0 0</point> on those two lines; this will be recorded as a failed annotation.\n\n'
        f'Question: {question}'
    )


def strict_parse(raw, width, height):
    match = re.fullmatch(r'\s*<bbox>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)</bbox>\s*\n\s*<point>(\d+)\s+(\d+)</point>\s*', raw)
    if not match:
        raise ValueError('Expected exactly two bbox/point lines')
    x1, y1, x2, y2, x, y = map(int, match.groups())
    if not (0 <= x1 < x2 <= 999 and 0 <= y1 < y2 <= 999 and x1 <= x <= x2 and y1 <= y <= y2):
        raise ValueError('Invalid/empty negative box or point outside box')
    parsed = parse_grounding_response(raw)
    parsed['bbox_pixel'] = bbox_to_pixel(parsed['bbox_1000'], width, height)
    parsed['point_pixel'] = point_to_pixel(parsed['point_1000'], width, height)
    a, b, c, d = parsed['bbox_pixel']
    if a >= c or b >= d:
        raise ValueError('Negative box has no pixels')
    return parsed


def enforce_crop_size(parsed, image_size, positive_size):
    """Correct only 1000-grid quantization; never resize a semantic proposal."""
    width, height = image_size
    cw, ch = positive_size
    if cw > width or ch > height:
        raise ValueError('Positive crop is larger than original image')
    x1, y1, x2, y2 = parsed['bbox_pixel']
    if abs((x2-x1)-cw) > math.ceil(width/1000)+1 or abs((y2-y1)-ch) > math.ceil(height/1000)+1:
        raise ValueError('Proposed negative box does not match positive crop size')
    box = [x1, y1, x1+cw, y1+ch]
    if box[2] > width or box[3] > height:
        raise ValueError('Same-size negative box exceeds original image')
    x,y = parsed['point_pixel']
    if not (box[0] <= x < box[2] and box[1] <= y < box[3]):
        raise ValueError('Point outside final same-size crop')
    return parsed | {'proposed_bbox_pixel': parsed['bbox_pixel'], 'bbox_pixel': box,
                     'positive_crop_size': list(positive_size),
                     'bbox_1000': [min(999, round(v*1000/size)) for v,size in
                                   zip(box, [width,height,width,height])]}


def process(args, example, strategy):
    started = time.perf_counter()
    prompt = build_prompt(example.question, strategy)
    record = base_record(example, args, prompt, False, 0) | {
        'strategy': strategy, 'raw_response': None, 'crop_path': None,
        'bbox_1000': None, 'point_1000': None, 'bbox_pixel': None, 'point_pixel': None}
    try:
        if example.row.get('_prepare_error'):
            raise ValueError(example.row['_prepare_error'])
        positives = flatten_images(example.row.get('teacher_images'))
        if len(positives) != 1:
            raise ValueError('Expected exactly one positive crop in teacher_images')
        positive = image_from_value(resolve_image_value(positives[0], args), args)
        prompt = build_prompt(example.question, strategy,
                              (example.image_width, example.image_height), positive.size)
        record['prompt'] = prompt
        raw = call_ark(args, [example.image_data_url, image_to_data_url(positive)], prompt)
        record['raw_response'] = raw
        record.update(enforce_crop_size(
            strict_parse(raw, example.image_width, example.image_height),
            (example.image_width, example.image_height), positive.size))
        extra = example.row.get('extra_info') or {}
        if isinstance(extra, str):
            extra = json.loads(extra)
        positive_box = extra.get('bbox_pixel')
        if positive_box is not None:
            a,b,c,d = record['bbox_pixel']; x,y,z,w = positive_box
            if min(c,z) > max(a,x) and min(d,w) > max(b,y):
                raise ValueError('Negative bbox overlaps positive bbox')
        if args.save_crops:
            image = image_from_value(example.image_value, args)
            path = (args.crop_output_dir / f'{strategy}_crop_{example.global_index:08d}.png').resolve()
            image.crop(record['bbox_pixel']).save(path)
            record['crop_path'] = str(path)
        record.update(ok=True, error=None)
    except Exception as exc:
        record.update(ok=False, error=str(exc))
    record['elapsed_sec'] = time.perf_counter() - started
    return record


def main(strategy):
    args = parse_args()
    validate_args(args)
    if not args.dataset.is_file() or args.dataset.suffix != '.parquet' or args.random:
        raise ValueError('Use one Parquet file; --random row shuffling is not supported by this entry point')
    if args.start_index < 0:
        raise ValueError('start-index must be nonnegative')
    if str(args.output_dir) == 'results/annotation':
        args.output_dir /= strategy + '_negative'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.save_crops = args.save_crops or args.write_train_parquet
    args.crop_output_dir = args.crop_output_dir or args.output_dir/'crops'
    args.crop_output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'strategy':strategy, 'source_sha256':hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                'config':{k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items() if k != 'resume'},
                'prompt':build_prompt('{question}', strategy), 'version':2}
    mp = args.output_dir/'manifest.json'
    if mp.exists() and json.loads(mp.read_text()) != manifest:
        raise ValueError('Output configuration differs; use a new output directory')
    mp.write_text(json.dumps(manifest, indent=2))
    results = args.output_dir/((args.output_name or 'grounding')+'.jsonl')
    records = {}
    if args.resume and results.exists():
        for line in results.read_text().splitlines():
            r = json.loads(line)
            if r.get('ok') and (not args.save_crops or Path(r['crop_path']).is_file()):
                records[r['global_index']] = r
    # Apply limit before skipping completed rows, preserving the exact selected set on resume.
    selected = []
    with results.open('a' if args.resume else 'w') as out, concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        pending = []
        def flush():
            for future in pending:
                r = future.result(); out.write(json.dumps(r, ensure_ascii=False)+'\n');out.flush()
                if r['ok']: records[r['global_index']] = r
            pending.clear()
        for example in iter_examples(args, set()):
            selected.append(example.global_index)
            if example.global_index not in records:
                pending.append(pool.submit(process, args, example, strategy))
            if len(pending) >= args.workers: flush()
        flush()
    ok = sum(i in records for i in selected)
    summary = {'strategy':strategy,'total':len(selected),'ok':ok,'failed':len(selected)-ok}
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
    if not selected or ok != len(selected):
        raise SystemExit('Incomplete annotations; no training Parquet exported. Retry failed rows.')
    if args.write_train_parquet:
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pq.read_table(args.dataset).take(pa.array(selected))
        field = 'teacher_neg_images'
        values = [[{'image':records[i]['crop_path']}] for i in selected]
        if field in table.column_names:
            f=table.schema.field(field);result=table.set_column(table.schema.get_field_index(field),f,pa.array(values,type=f.type))
        else: result=table.append_column(field,pa.array(values))
        path=args.output_dir/'train.parquet';pq.write_table(result,path)
        checked=pq.read_table(path)
        for col in table.column_names:
            if col != field and not checked[col].equals(table[col]):raise ValueError(f'Changed source column {col}')
