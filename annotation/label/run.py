from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .ark import call_ark
from .config import parse_args, validate_args
from .data import count_selected_rows, image_from_value, iter_examples
from .grounding import bbox_to_pixel, build_grounding_prompt, parse_grounding_response, point_to_pixel
from .output import (
    base_record,
    load_completed_indices,
    summarize_results,
    write_summary,
    write_train_parquet,
)
from .schema import Example


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.write_train_parquet:
        args.save_crops = True
    if args.save_crops:
        crop_dir = args.crop_output_dir or (args.output_dir / "crops")
        crop_dir.mkdir(parents=True, exist_ok=True)
        args.crop_output_dir = crop_dir

    output_name = args.output_name or "grounding"
    results_path = args.output_dir / f"{output_name}.jsonl"
    summary_path = args.output_dir / "summary.json"
    train_parquet_path = args.output_dir / "train.parquet"

    run_batch(args, results_path, summary_path, train_parquet_path)


def run_batch(
    args: argparse.Namespace,
    results_path: Path,
    summary_path: Path,
    train_parquet_path: Path,
) -> None:
    started = time.perf_counter()

    completed = load_completed_indices(results_path) if args.resume else set()
    progress_total = count_selected_rows(args, completed)
    mode = "a" if args.resume else "w"
    new_records = 0

    with results_path.open(mode, encoding="utf-8") as output:
        iterator = iter_examples(args, completed)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures: set[concurrent.futures.Future[dict[str, Any]]] = set()
            with tqdm(total=progress_total, desc="labeling", unit="sample", dynamic_ncols=True) as progress:
                for example in iterator:
                    futures.add(executor.submit(process_example, args, example))
                    if len(futures) >= args.workers:
                        finished = drain_one(futures, output)
                        new_records += finished
                        progress.update(finished)
                        write_summary(summary_path, results_path, args, started)

                while futures:
                    finished = drain_one(futures, output)
                    new_records += finished
                    progress.update(finished)
                    write_summary(summary_path, results_path, args, started)
                if progress.total is not None and progress.n < progress.total:
                    progress.total = progress.n
                    progress.refresh()

    write_summary(summary_path, results_path, args, started)
    if args.write_train_parquet:
        write_train_parquet(results_path, train_parquet_path)
    print(
        json.dumps(
            {
                "new_records": new_records,
                "mode": args.mode,
                "results": str(results_path),
                "summary": str(summary_path),
                "train_parquet": str(train_parquet_path) if args.write_train_parquet else None,
                **summarize_results(results_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def drain_one(futures: set[concurrent.futures.Future[dict[str, Any]]], output: Any) -> int:
    done, pending = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
    futures.clear()
    futures.update(pending)
    for future in done:
        record = future.result()
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
        output.flush()
    return len(done)


def process_example(args: argparse.Namespace, example: Example) -> dict[str, Any]:
    started = time.perf_counter()
    prompt = build_grounding_prompt(example.question)
    prepare_error = example.row.get("_prepare_error")
    if prepare_error:
        return failed_record(args, example, prompt, str(prepare_error), started)
    try:
        raw_response = call_ark(args, example.image_data_url, prompt)
        parsed = parse_grounding_response(raw_response)
        bbox_1000 = parsed["bbox_1000"]
        point_1000 = parsed["point_1000"]
        bbox_pixel = bbox_to_pixel(bbox_1000, example.image_width, example.image_height)
        point_pixel = point_to_pixel(point_1000, example.image_width, example.image_height)
        crop_path = save_crop(args, example, bbox_pixel) if args.save_crops else None
        return base_record(example, args, prompt, True, time.perf_counter() - started) | {
            "raw_response": raw_response,
            "bbox_1000": bbox_1000,
            "point_1000": point_1000,
            "bbox_pixel": bbox_pixel,
            "point_pixel": point_pixel,
            "crop_path": crop_path,
            "source_row": example.row,
            "error": None,
        }
    except Exception as exc:
        return failed_record(args, example, prompt, str(exc), started)


def failed_record(
    args: argparse.Namespace,
    example: Example,
    prompt: str,
    error: str,
    started: float,
) -> dict[str, Any]:
    return base_record(example, args, prompt, False, time.perf_counter() - started) | {
        "raw_response": None,
        "bbox_1000": None,
        "point_1000": None,
        "bbox_pixel": None,
        "point_pixel": None,
        "crop_path": None,
        "source_row": example.row,
        "error": error,
    }


def save_crop(args: argparse.Namespace, example: Example, bbox_pixel: list[int]) -> str:
    assert args.crop_output_dir is not None
    image = image_from_value(example.image_value, args).convert("RGB")
    x1, y1, x2, y2 = bbox_pixel
    crop = image.crop((x1, y1, x2, y2))
    path = Path(args.crop_output_dir) / f"seed_crop_{example.global_index:06d}.jpg"
    crop.save(path, quality=95)
    return str(path)


if __name__ == "__main__":
    main()
