"""Prepare aligned black, Seed-labelled irrelevant and nearby negative crops."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import random
from pathlib import Path

from PIL import Image

from annotation.label.ark import call_ark
from annotation.label.data import (
    answer_text, first_image_value, flatten_images, image_from_value,
    image_to_data_url, question_text, resolve_image_value,
)


def normalize(value):
    if hasattr(value, "tolist"):
        return normalize(value.tolist())
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    return value


def intersection(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))


def grounding_fields(box, image_size):
    """Keep pixel boxes authoritative: 1000-grid rounding loses precision on large images."""
    width, height = image_size
    x1, y1, x2, y2 = box
    point = [(x1 + x2) // 2, (y1 + y2) // 2]
    scale = lambda v, size: max(0, min(999, round(v * 1000 / size)))
    bbox = [scale(x1, width), scale(y1, height), scale(x2, width), scale(y2, height)]
    center = [scale(point[0], width), scale(point[1], height)]
    return {"bbox_pixel": list(box), "point_pixel": point,
            "bbox_1000": bbox, "point_1000": center,
            "annotation_text": "<bbox>" + " ".join(map(str, bbox)) + "</bbox>\n"
                               + "<point>" + " ".join(map(str, center)) + "</point>"}


def candidates(image_size, crop_size, positive, strategy, seed):
    width, height = image_size
    cw, ch = crop_size
    if cw > width or ch > height:
        raise ValueError("Positive crop is larger than the source image")
    x1, y1, x2, y2 = positive
    if strategy == "nearby":
        # Touch the positive box at an edge; never move a distant crop into this group.
        centers = [(x1 - cw / 2, (y1 + y2) / 2),
                   (x2 + cw / 2, (y1 + y2) / 2),
                   ((x1 + x2) / 2, y1 - ch / 2),
                   ((x1 + x2) / 2, y2 + ch / 2)]
        boxes = []
        for cx, cy in centers:
            left, top = round(cx - cw / 2), round(cy - ch / 2)
            # Clamp only along the edge, not towards/into the positive box.
            if cx < x1 or cx > x2:
                top = max(0, min(height - ch, top))
            else:
                left = max(0, min(width - cw, left))
            boxes.append((left, top, left + cw, top + ch))
    else:
        rng = random.Random(seed)
        boxes = []
        for _ in range(256):
            left = rng.randint(0, width - cw)
            top = rng.randint(0, height - ch)
            boxes.append((left, top, left + cw, top + ch))
        # A spread of proposals for Seed to select semantically, not random final labels.
        boxes = sorted(boxes, key=lambda b: -math.hypot(
            (b[0] + b[2] - x1 - x2) / 2, (b[1] + b[3] - y1 - y2) / 2))
    valid = []
    for box in boxes:
        if (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height
                and not intersection(box, positive) and box not in valid):
            valid.append(box)
    return valid[:16]


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(text)


def recover_positive_bbox(image, positive_image):
    """Recover an unresized crop only when the pixel match is unambiguous."""
    import cv2
    import numpy as np
    if positive_image.width > image.width or positive_image.height > image.height:
        raise ValueError("Missing positive bbox and positive crop is larger than image")
    scores = cv2.matchTemplate(np.asarray(image), np.asarray(positive_image), cv2.TM_SQDIFF_NORMED)
    minimum, _, location, _ = cv2.minMaxLoc(scores)
    if minimum > 0.02:
        raise ValueError("Missing positive bbox: original crop could not be matched reliably")
    x, y = location
    cw, ch = positive_image.size
    # Reject competing locations outside the local JPEG matching basin.
    other = scores.copy()
    other[max(0, y - max(2, ch // 4)):y + max(2, ch // 4) + 1,
          max(0, x - max(2, cw // 4)):x + max(2, cw // 4) + 1] = 1
    if float(other.min()) < max(0.005, minimum * 2):
        raise ValueError("Missing positive bbox: crop position is ambiguous")
    return [x, y, x + cw, y + ch]


def select_crop(args, image, positive, boxes, question, answer, strategy):
    if not boxes:
        raise ValueError(f"No valid same-size {strategy} crop outside positive evidence")
    prompt = (
        "Select a negative evidence crop for a visual reasoning ablation. Coordinates "
        "are ORIGINAL IMAGE PIXELS [left, top, right, bottom], not normalized.\n"
        f"Image size: {image.size}. Positive evidence box: {positive}.\n"
        f"Question: {question}\nReference answer (annotation only): {answer}\n"
        f"Strategy: {strategy}. Candidates: {dict(enumerate(boxes))}\n"
        "For irrelevant select a region unrelated to the question. For nearby select "
        "an adjacent region with local surrounding context. In BOTH cases the crop "
        "must exclude every detail that provides or reveals the answer, including "
        "other instances of the target. If none qualifies, return candidate_id null. "
        'Return ONLY JSON: {"candidate_id": integer_or_null, "reason": "..."}.'
    )
    raw = call_ark(args, image_to_data_url(image), prompt)
    result = parse_json(raw)
    index = result.get("candidate_id")
    if type(index) is not int or not 0 <= index < len(boxes):
        raise ValueError(f"Seed found no valid {strategy} candidate")
    box = boxes[index]
    crop = image.crop(box)
    verification = call_ark(args, image_to_data_url(crop), (
        "Verify this proposed negative crop, shown without the original image.\n"
        f"Question: {question}\nReference answer: {answer}\n"
        "Does this crop contain any visual evidence that reveals or supports the "
        "answer? Reject if uncertain. "
        'Return ONLY JSON: {"contains_answer_evidence": true_or_false, "reason": "..."}.'
    ))
    if parse_json(verification).get("contains_answer_evidence") is not False:
        raise ValueError(f"Seed verification rejected {strategy} crop")
    return crop, {"bbox_pixel": box, "selection": raw, "verification": verification}


def process_row(args, index, raw_row):
    row = normalize(raw_row)
    extra = row.get("extra_info") or {}
    if isinstance(extra, str):
        extra = json.loads(extra)
    image = image_from_value(first_image_value(row, args), args)
    positives = flatten_images(row.get("teacher_images"))
    if len(positives) != 1:
        raise ValueError("Expected exactly one positive crop")
    pos_image = image_from_value(resolve_image_value(positives[0], args), args)
    positive = extra.get(args.bbox_key)
    bbox_source = "metadata"
    if positive is None:
        positive = recover_positive_bbox(image, pos_image)
        bbox_source = "unambiguous_pixel_match"
    if not isinstance(positive, list) or len(positive) != 4:
        raise ValueError(f"Missing positive bbox extra_info.{args.bbox_key}")
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in positive):
        raise ValueError("Invalid positive bbox coordinates")
    positive = [math.floor(positive[0]), math.floor(positive[1]),
                math.ceil(positive[2]), math.ceil(positive[3])]
    if not (0 <= positive[0] < positive[2] <= image.width
            and 0 <= positive[1] < positive[3] <= image.height):
        raise ValueError("Positive bbox outside original image")
    question = question_text(row)
    if not question:
        raise ValueError("Question missing")
    record = {"row_index": index, "model": args.model, "crops": {},
              "positive_bbox_pixel": positive, "positive_bbox_source": bbox_source}
    strategies = getattr(args, "strategies", ("black", "irrelevant", "nearby"))
    crops = {}
    if "black" in strategies:
        crops["black"] = (Image.new("RGB", pos_image.size, (0, 0, 0)), {"bbox_pixel": None})
    for strategy in (s for s in strategies if s != "black"):
        boxes = candidates(image.size, pos_image.size, positive, strategy, args.seed + index)
        crops[strategy] = select_crop(args, image, positive, boxes, question, answer_text(row), strategy)
    for strategy, (crop, metadata) in crops.items():
        if metadata.get("bbox_pixel") is not None:
            metadata.update(grounding_fields(metadata["bbox_pixel"], image.size))
        path = (args.output_dir / strategy / "crops" / f"{index:08d}.png").resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(path)
        record["crops"][strategy] = {"image": str(path), "size": crop.size, **metadata}
    record["ok"] = True
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--bbox-key", default="bbox_pixel")
    parser.add_argument("--strategies", nargs="+", choices=("black", "irrelevant", "nearby"),
                        default=["black", "irrelevant", "nearby"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", default="ARK_API_KEY")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.strategies = list(dict.fromkeys(args.strategies))
    args.timeout, args.max_retries, args.retry_sleep = 120, 3, 2
    args.temperature, args.top_p, args.max_tokens = 0, 1, 512
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    import pyarrow as pa
    import pyarrow.parquet as pq
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256()
    with args.dataset.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            source_hash.update(chunk)
    digest = source_hash.hexdigest()
    manifest = {"source": str(args.dataset.resolve()), "sha256": digest,
                "model": args.model, "base_url": args.base_url, "seed": args.seed,
                "limit": args.limit, "bbox_key": args.bbox_key,
                "image_root": str(args.image_root) if args.image_root else None,
                "strategies": args.strategies, "version": 2}
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Output contains another annotation configuration; use a new output directory")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    table = pq.read_table(args.dataset)
    if args.limit is not None:
        table = table.slice(0, args.limit)
    rows = table.to_pylist()
    results_path = args.output_dir / "annotations.jsonl"
    records = {}
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            record = json.loads(line)
            if (record.get("ok") and set(record["crops"]) == set(args.strategies)
                    and all(Path(c["image"]).exists() for c in record["crops"].values())):
                records[record["row_index"]] = record
    def worker(index):
        try:
            return process_row(args, index, rows[index])
        except Exception as error:
            return {"row_index": index, "ok": False, "error": str(error)}
    with results_path.open("a") as output, concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        # Bound pending requests and image memory to one worker batch.
        pending = [i for i in range(len(rows)) if i not in records]
        for start in range(0, len(pending), args.workers):
            for record in pool.map(worker, pending[start:start + args.workers]):
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                if record["ok"]:
                    records[record["row_index"]] = record
                print(json.dumps({"row": record["row_index"], "ok": record["ok"]}), flush=True)
    summary = {"total": len(rows), "aligned_valid": len(records), "failed": len(rows) - len(records)}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if len(records) != len(rows):
        raise SystemExit("Incomplete annotations: no training parquet exported. Inspect errors and retry.")
    for strategy in args.strategies:
        labels = args.output_dir / strategy / "grounding.jsonl"
        with labels.open("w", encoding="utf-8") as handle:
            for i in range(len(rows)):
                label = {"ok": True, "global_index": i, "row_index": i,
                         "source_file": str(args.dataset.resolve()), "strategy": strategy,
                         "positive_bbox_pixel": records[i]["positive_bbox_pixel"],
                         **records[i]["crops"][strategy]}
                label["crop_path"] = label["image"]
                handle.write(json.dumps(label, ensure_ascii=False) + "\n")
        field = "teacher_neg_images"
        values = [[{"image": records[i]["crops"][strategy]["image"]}] for i in range(len(rows))]
        if field in table.column_names:
            schema_field = table.schema.field(field)
            result = table.set_column(table.schema.get_field_index(field), schema_field,
                                      pa.array(values, type=schema_field.type))
        else:
            result = table.append_column(field, pa.array(values))
        # RLHFDataset rebuilds teacher_neg_raw_prompt from these images at load time.
        path = args.output_dir / strategy / "train.parquet"
        pq.write_table(result, path)
        checked = pq.read_table(path)
        assert checked.num_rows == table.num_rows
        for name in table.column_names:
            if name != field and not checked[name].equals(table[name]):
                raise ValueError(f"Unexpected change to source column {name}")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
