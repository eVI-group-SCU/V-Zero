from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .schema import Example


def base_record(
    example: Example,
    args: argparse.Namespace,
    prompt: str,
    ok: bool,
    elapsed_sec: float,
) -> dict[str, Any]:
    prepare_error = example.row.get("_prepare_error")
    return {
        "ok": ok and prepare_error is None,
        "global_index": example.global_index,
        "row_index": example.row_index,
        "source_file": example.source_file,
        "id": sample_id(example.row, example.global_index),
        "image": example.image_source,
        "image_width": example.image_width,
        "image_height": example.image_height,
        "question": example.question,
        "answer": example.answer,
        "model": args.model,
        "prompt": prompt,
        "elapsed_sec": elapsed_sec,
    }


def sample_id(row: dict[str, Any], global_index: int) -> str:
    for key in ("id", "sample_id", "uid", "uuid"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return str(global_index)


def load_completed_indices(path: Path) -> set[int]:
    completed: set[int] = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                completed.add(int(record["global_index"]))
            except Exception:
                continue
    return completed


def summarize_results(path: Path) -> dict[str, Any]:
    total = ok = failed = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            total += 1
            record = json.loads(line)
            if record.get("ok"):
                ok += 1
            else:
                failed += 1
    return {"total": total, "ok": ok, "failed": failed}


def write_summary(path: Path, results_path: Path, args: argparse.Namespace, started: float) -> None:
    if not results_path.exists():
        return
    summary = {
        "dataset": str(args.dataset),
        "model": args.model,
        "base_url": args.base_url,
        "elapsed_sec": time.perf_counter() - started,
        "workers": args.workers,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "random": args.random,
        "random_seed": args.random_seed,
        **summarize_results(results_path),
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def write_train_parquet(results_path: Path, output_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    with results_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if not record.get("ok"):
                continue
            crop_path = record.get("crop_path")
            if not crop_path:
                continue
            row = record.get("source_row")
            if not isinstance(row, dict):
                continue
            training_row = training_row_from_record(row, record)
            training_row["_label_global_index"] = int(record.get("global_index") or 0)
            rows.append(training_row)
    rows.sort(key=lambda item: int(item.pop("_label_global_index")))
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output_path)


def training_row_from_record(row: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    updated = dict(row)
    image = record.get("image")
    crop_path = record.get("crop_path")
    question = str(record.get("question") or "").strip()
    updated["images"] = image_list(image) if image else row.get("images", [])
    updated["teacher_images"] = image_list(crop_path) if crop_path else []
    updated["teacher_raw_prompt"] = teacher_raw_prompt(question)

    extra_info = dict(updated.get("extra_info") or {})
    extra_info.update(
        {
            "crop_image": crop_path,
            "teacher_crop_status": "crop",
            "teacher_crop_error": None,
            "bbox": as_float_list(record.get("bbox_pixel")),
            "bbox_pixel": as_int_list(record.get("bbox_pixel")),
            "bbox_2d": as_int_list(record.get("bbox_1000")),
            "point": as_int_list(record.get("point_pixel")),
            "point_2d": as_int_list(record.get("point_1000")),
        }
    )
    updated["extra_info"] = extra_info
    return updated


def image_list(path: str | None) -> list[dict[str, str]]:
    return [{"image": path}] if path else []


def as_int_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    return [int(item) for item in value]


def as_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    return [float(item) for item in value]


def teacher_raw_prompt(question: str) -> list[dict[str, Any]]:
    content = (
        f"<image>\n{question}\n\n"
        "The next image is a cropped view of the target region from the original image: <crop>\n"
        "Use the crop as focused visual evidence for the relevant object or region, "
        "while keeping the original image as context."
    )
    return [{"role": "user", "content": content}]
