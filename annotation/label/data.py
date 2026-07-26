from __future__ import annotations

import argparse
import base64
import io
import json
import random
import re
from pathlib import Path
from typing import Any

from PIL import Image

from .schema import Example


TEXT_KEYS = ("question", "problem", "query", "text", "instruction")
ANSWER_KEYS = ("answer", "ground_truth", "gt_answer", "gts", "GTS", "target", "label")
IMAGE_KEYS = ("images", "image", "original_images", "original_image", "image_path", "image_paths")
NESTED_TEXT_KEYS = TEXT_KEYS + ("raw_question",)
NESTED_ANSWER_KEYS = ANSWER_KEYS + ("final_answer", "response")


def iter_examples(args: argparse.Namespace, completed: set[int]) -> Any:
    for global_index, row_index, source_file, row in iter_selected_rows(args, completed):
        try:
            yield prepare_example(args, global_index, row_index, source_file, row)
        except Exception as exc:
            yield failed_prepare_example(global_index, row_index, source_file, row, str(exc))


def iter_selected_rows(args: argparse.Namespace, completed: set[int]) -> Any:
    rows = (
        item
        for item in iter_rows(args.dataset, args.read_batch_size)
        if item[0] >= args.start_index and item[0] not in completed
    )
    if not args.random:
        for emitted, item in enumerate(rows):
            if args.limit is not None and emitted >= args.limit:
                break
            yield item
        return

    selected = list(rows)
    rng = random.Random(args.random_seed)
    rng.shuffle(selected)
    if args.limit is not None:
        selected = selected[: args.limit]
    yield from selected


def iter_rows(dataset: Path, read_batch_size: int) -> Any:
    files = dataset_files(dataset)
    global_index = 0
    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            import pyarrow.parquet as pq

            parquet_file = pq.ParquetFile(path)
            row_index = 0
            for batch in parquet_file.iter_batches(batch_size=read_batch_size):
                for row in batch.to_pylist():
                    yield global_index, row_index, str(path), row
                    global_index += 1
                    row_index += 1
        elif suffix == ".jsonl":
            with path.open(encoding="utf-8") as handle:
                for row_index, line in enumerate(handle):
                    if not line.strip():
                        continue
                    yield global_index, row_index, str(path), json.loads(line)
                    global_index += 1
        elif suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload.get("data", payload) if isinstance(payload, dict) else payload
            for row_index, row in enumerate(rows):
                yield global_index, row_index, str(path), row
                global_index += 1
        else:
            raise ValueError(f"Unsupported dataset file type: {path}")


def dataset_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files: list[Path] = []
    for pattern in ("*.parquet", "*.jsonl", "*.json"):
        files.extend(sorted(path.rglob(pattern)))
    if not files:
        raise FileNotFoundError(f"No parquet/jsonl/json files found under: {path}")
    return files


def count_selected_rows(args: argparse.Namespace, completed: set[int]) -> int:
    total_rows = count_rows(args.dataset)
    remaining = max(total_rows - args.start_index, 0)
    completed_after_start = sum(
        1 for index in completed if args.start_index <= index < total_rows
    )
    remaining = max(remaining - completed_after_start, 0)
    if args.limit is not None:
        return min(args.limit, remaining)
    return remaining


def count_rows(dataset: Path) -> int:
    total = 0
    for path in dataset_files(dataset):
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            import pyarrow.parquet as pq

            total += pq.ParquetFile(path).metadata.num_rows
        elif suffix == ".jsonl":
            with path.open(encoding="utf-8") as handle:
                total += sum(1 for line in handle if line.strip())
        elif suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload.get("data", payload) if isinstance(payload, dict) else payload
            total += len(rows)
        else:
            raise ValueError(f"Unsupported dataset file type: {path}")
    return total


def prepare_example(
    args: argparse.Namespace,
    global_index: int,
    row_index: int,
    source_file: str,
    row: dict[str, Any],
) -> Example:
    image_value = first_image_value(row, args)
    image_source = image_source_text(image_value)
    question = question_text(row)
    if not question:
        raise ValueError("Could not find question text.")
    image = image_from_value(image_value, args)
    width, height = image.size
    image_data_url = image_to_data_url(image)
    return Example(
        global_index=global_index,
        row_index=row_index,
        source_file=source_file,
        row=row,
        image_value=image_value,
        image_source=image_source,
        question=question,
        answer=answer_text(row),
        image_width=width,
        image_height=height,
        image_data_url=image_data_url,
    )


def failed_prepare_example(
    global_index: int,
    row_index: int,
    source_file: str,
    row: dict[str, Any],
    error: str,
) -> Example:
    return Example(
        global_index=global_index,
        row_index=row_index,
        source_file=source_file,
        row=row | {"_prepare_error": error},
        image_value=None,
        image_source="",
        question=question_text(row) or "",
        answer=answer_text(row),
        image_width=0,
        image_height=0,
        image_data_url="",
    )


def first_image_value(row: dict[str, Any], args: argparse.Namespace) -> Any:
    for key in IMAGE_KEYS:
        if key not in row:
            continue
        values = flatten_images(row[key])
        if values:
            return resolve_image_value(values[0], args)
    extra = row.get("extra_info")
    if isinstance(extra, dict):
        for key in IMAGE_KEYS:
            values = flatten_images(extra.get(key))
            if values:
                return resolve_image_value(values[0], args)
    raise ValueError("Could not find original image.")


def flatten_images(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, dict):
        if any(key in value for key in ("path", "bytes", "image")):
            return [value]
        images: list[Any] = []
        for nested in value.values():
            images.extend(flatten_images(nested))
        return images
    if isinstance(value, (list, tuple)):
        images = []
        for item in value:
            images.extend(flatten_images(item))
        return images
    return [value]


def resolve_image_value(value: Any, args: argparse.Namespace) -> Any:
    if isinstance(value, dict):
        item = dict(value)
        if item.get("path") is not None:
            item["path"] = resolve_path(str(item["path"]), args)
        if item.get("image") is not None and isinstance(item["image"], str):
            item["image"] = resolve_path(str(item["image"]), args)
        return item
    if isinstance(value, str):
        return resolve_path(value, args)
    return value


def resolve_path(value: str, args: argparse.Namespace) -> str:
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", value):
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    candidates = []
    if args.image_root is not None:
        candidates.append(args.image_root / path)
    if args.dataset.is_dir():
        candidates.append(args.dataset / path)
    else:
        candidates.append(args.dataset.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0] if candidates else path)


def image_from_value(value: Any, args: argparse.Namespace) -> Image.Image:
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path") is not None:
            return Image.open(value["path"]).convert("RGB")
        if value.get("image") is not None:
            return image_from_value(value["image"], args)
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(value))).convert("RGB")
    if isinstance(value, str):
        if re.match(r"^data:image/[^;]+;base64,", value):
            payload = value.split(",", 1)[1]
            return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
        return Image.open(value).convert("RGB")
    raise ValueError(f"Unsupported image value: {type(value).__name__}")


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=95)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def question_text(row: dict[str, Any]) -> str | None:
    value = first_string(row, TEXT_KEYS)
    if value:
        return value
    extra = row.get("extra_info")
    if isinstance(extra, dict):
        value = first_string(extra, NESTED_TEXT_KEYS)
        if value:
            return value
    return text_from_messages(row.get("prompt")) or text_from_messages(row.get("messages"))


def answer_text(row: dict[str, Any]) -> str | None:
    value = first_string(row, ANSWER_KEYS)
    if value:
        return value
    for key in ("extra_info", "metadata", "reward_model"):
        nested = row.get(key)
        if isinstance(nested, dict):
            value = first_string(nested, NESTED_ANSWER_KEYS)
            if value:
                return value
    return None


def first_string(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is not None and not isinstance(value, (dict, list, tuple)):
            text = str(value).strip()
            if text:
                return text
    return None


def text_from_messages(messages: Any) -> str | None:
    if not isinstance(messages, list):
        return None
    pieces: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            pieces.append(strip_image_placeholders(content))
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    pieces.append(strip_image_placeholders(str(item.get("text", ""))))
                elif isinstance(item, str):
                    pieces.append(strip_image_placeholders(item))
    text = "\n".join(piece.strip() for piece in pieces if piece.strip()).strip()
    return text or None


def strip_image_placeholders(text: str) -> str:
    return text.replace("<image>", "").replace("<crop>", "").strip()


def image_source_text(value: Any) -> str:
    if isinstance(value, dict):
        if value.get("path") is not None:
            return str(value["path"])
        if value.get("image") is not None:
            return image_source_text(value["image"])
        if value.get("bytes") is not None:
            return "<image-bytes>"
    return str(value)
