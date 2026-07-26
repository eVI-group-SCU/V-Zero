from __future__ import annotations

import base64
import csv
import ast
import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .schema import EvalExample, EvalImage, base64_image, bytes_image, path_image, pil_image
from .scoring import choice_map_from_prompt, expected_mcq_letter


DATASET_ALIASES = {
    "vstar": "vstar",
    "vstar_bench": "vstar",
    "hr-4k": "hrbench_4k",
    "hrbench_4k": "hrbench_4k",
    "hr_bench_4k": "hrbench_4k",
    "hr-8k": "hrbench_8k",
    "hrbench_8k": "hrbench_8k",
    "hr_bench_8k": "hrbench_8k",
    "zoombench": "zoombench",
    "zoom_bench": "zoombench",
    "mmstar": "mmstar",
    "mm_star": "mmstar",
    "mmstar_bench": "mmstar",
    "mme_realworld": "mme_realworld",
    "mme_real_world": "mme_realworld",
    "mme-realworld": "mme_realworld",
    "mme-real-world": "mme_realworld",
    "mme_realworld_lite": "mme_realworld",
    "mme-realworld-lite": "mme_realworld",
    "mme_realworld_en": "mme_realworld_en",
    "mme-realworld-en": "mme_realworld_en",
    "mme_rw_en": "mme_realworld_en",
    "mme-rw-en": "mme_realworld_en",
    "mme_realworld_cn": "mme_realworld_cn",
    "mme-realworld-cn": "mme_realworld_cn",
    "mme_rw_cn": "mme_realworld_cn",
    "mme-rw-cn": "mme_realworld_cn",
    "cvbench": "cvbench",
    "cv_bench": "cvbench",
    "cv-bench": "cvbench",
    "colorbench": "colorbench",
    "color_bench": "colorbench",
    "color-bench": "colorbench",
    "babyvision": "babyvision",
    "baby_vision": "babyvision",
    "baby-vision": "babyvision",
    "pope": "pope",
    "pope_coco": "pope",
    "pope-coco": "pope",
    "mmvp": "mmvp",
    "mmvp_benchmark": "mmvp",
    "mmvp-benchmark": "mmvp",
}

DEFAULT_DATASET_ORDER = (
    "vstar",
    "hrbench_4k",
    "hrbench_8k",
    "zoombench",
)

DATASET_ORDER = (
    *DEFAULT_DATASET_ORDER,
    "mme_realworld_en",
    "mme_realworld_cn",
    "mmstar",
    "cvbench",
    "colorbench",
    "babyvision",
)

SUPPORTED_DATASET_ORDER = (
    *DATASET_ORDER,
    "pope",
    "mmvp",
)

DATASET_GROUPS = {
    "all": DATASET_ORDER,
    "default": DEFAULT_DATASET_ORDER,
    "core": DEFAULT_DATASET_ORDER,
}

HF_REPOS = {
    "vstar": ("craigwu", "vstar_bench"),
    "hrbench_4k": ("DreamMr", "HR-Bench"),
    "hrbench_8k": ("DreamMr", "HR-Bench"),
    "zoombench": ("inclusionAI", "ZoomBench"),
    "mmstar": ("Lin-Chen", "MMStar"),
    "mme_realworld": ("yifanzhang114", "MME-RealWorld-Lite"),
    "mme_realworld_en": ("MME-RealWorld", "MME-RW-en"),
    "mme_realworld_cn": ("MME-RealWorld", "MME-RW-cn"),
    "cvbench": ("nyu-visionx", "CV-Bench"),
    "colorbench": ("umd-zhou-lab", "ColorBench"),
    "babyvision": ("UnipatAI", "BabyVision"),
    "pope": ("lmms-lab", "POPE"),
    "mmvp": ("MMVP", "MMVP"),
}

INLINE_OPTION_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z])\s*[:.)]\s*")
LABELED_OPTION_RE = re.compile(r"^\s*[\(\[]?([A-Z])[\)\].:：]\s*(.+?)\s*$")
MMVP_OPTION_RE = re.compile(r"\(([A-Za-z])\)\s*")
MMVP_OPTION_LABELS = ("A", "B")


def normalize_dataset_name(name: str) -> str:
    if name in DATASET_GROUPS:
        return name
    try:
        return DATASET_ALIASES[name.lower()]
    except KeyError as exc:
        allowed = ", ".join((*DATASET_GROUPS, *SUPPORTED_DATASET_ORDER))
        raise ValueError(f"Unsupported dataset '{name}'. Choose one of: {allowed}.") from exc


def iter_dataset(
    name: str,
    data_root: Path,
    start_index: int = 0,
    limit: int | None = None,
    zoombench_view: str = "full",
) -> Iterator[tuple[int, EvalExample]]:
    dataset = normalize_dataset_name(name)
    if dataset in DATASET_GROUPS:
        emitted = 0
        for item in DATASET_GROUPS[dataset]:
            for row_index, example in iter_dataset(
                item,
                data_root=data_root,
                start_index=0,
                limit=None,
                zoombench_view=zoombench_view,
            ):
                if emitted < start_index:
                    emitted += 1
                    continue
                if limit is not None and emitted - start_index >= limit:
                    return
                emitted += 1
                yield emitted - 1, example
        return

    adapter = {
        "vstar": iter_vstar,
        "hrbench_4k": lambda root: iter_hrbench(root, "hrbench_4k"),
        "hrbench_8k": lambda root: iter_hrbench(root, "hrbench_8k"),
        "zoombench": lambda root: iter_zoombench(root, zoombench_view),
        "mmstar": iter_mmstar,
        "mme_realworld": lambda root: iter_mme_realworld(root, "mme_realworld"),
        "mme_realworld_en": lambda root: iter_mme_realworld(root, "mme_realworld_en"),
        "mme_realworld_cn": lambda root: iter_mme_realworld(root, "mme_realworld_cn"),
        "cvbench": iter_cvbench,
        "colorbench": iter_colorbench,
        "babyvision": iter_babyvision,
        "pope": iter_pope,
        "mmvp": iter_mmvp,
    }[dataset]
    emitted = 0
    for row_index, example in enumerate(adapter(data_root)):
        if row_index < start_index:
            continue
        if limit is not None and emitted >= limit:
            break
        emitted += 1
        yield row_index, example


def iter_vstar(data_root: Path) -> Iterator[EvalExample]:
    root = resolve_dataset_root(data_root, "vstar")
    questions_path = root / "test_questions.jsonl"
    if not questions_path.is_file():
        raise FileNotFoundError(f"VStar questions not found: {questions_path}")
    with questions_path.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            record = json.loads(line)
            prompt = str(record["text"])
            choices = choice_map_from_prompt(prompt)
            answer = expected_mcq_letter(str(record["label"]), choices) if choices else str(record["label"])
            yield EvalExample(
                id=str(record.get("question_id", row_index)),
                dataset="vstar",
                split="test",
                question=prompt,
                images=[path_image(root / str(record["image"]))],
                choices=choices,
                answer=answer,
                question_type="mcq",
                category=str(record.get("category", "")) or None,
                metadata={"source": "craigwu/vstar_bench"},
            )


def iter_hrbench(data_root: Path, split: str) -> Iterator[EvalExample]:
    root = resolve_dataset_root(data_root, split)
    suffix = "4k" if split.endswith("4k") else "8k"
    tsv_path = root / f"hr_bench_{suffix}.tsv"
    if tsv_path.is_file():
        yield from _iter_hrbench_tsv(tsv_path, split)
        return

    parquet_path = root / f"hr_bench_{suffix}.parquet"
    if parquet_path.is_file():
        yield from _iter_hrbench_parquet(parquet_path, split)
        return
    raise FileNotFoundError(f"HR-Bench {suffix} TSV/parquet not found under: {root}")


def _iter_hrbench_tsv(path: Path, split: str) -> Iterator[EvalExample]:
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            yield _hrbench_example(row, split)


def _iter_hrbench_parquet(path: Path, split: str) -> Iterator[EvalExample]:
    for row in _iter_parquet_records(path):
        yield _hrbench_example(row, split)


def _hrbench_example(row: dict[str, Any], split: str) -> EvalExample:
    choices = {label: str(row[label]) for label in ("A", "B", "C", "D")}
    question = format_mcq_question(str(row["question"]), choices)
    answer = expected_mcq_letter(str(row["answer"]), choices)
    return EvalExample(
        id=str(row["index"]),
        dataset="hrbench",
        split=split,
        question=question,
        images=[base64_image(str(row["image"]))],
        choices=choices,
        answer=answer,
        question_type="mcq",
        category=str(row.get("category", "")) or None,
        metadata={
            "cycle_category": str(row.get("cycle_category", "")),
            "source": "DreamMr/HR-Bench",
        },
    )


def iter_zoombench(data_root: Path, view: str) -> Iterator[EvalExample]:
    if view not in {"full", "crop", "both"}:
        raise ValueError("--zoombench-view must be one of: full, crop, both.")
    root = resolve_dataset_root(data_root, "zoombench")
    parquet_path = root / "data" / "test.parquet"
    if not parquet_path.is_file():
        raise FileNotFoundError(f"ZoomBench parquet not found: {parquet_path}")

    for row in _iter_parquet_records(parquet_path):
        views = ("full", "crop") if view == "both" else (view,)
        for item_view in views:
            yield _zoombench_example(row, item_view)


def _zoombench_example(row: dict[str, Any], view: str) -> EvalExample:
    query = str(row["query"])
    choices = choice_map_from_prompt(query)
    question_type = str(row.get("question_type") or ("mcq" if choices else "open"))
    answer = str(row["response"])
    if choices:
        answer = expected_mcq_letter(answer, choices)
    image_field = row["crop_image"] if view == "crop" else row["image"]
    row_id = str(row.get("id") or "")
    return EvalExample(
        id=f"{row_id}:{view}" if view != "full" else row_id,
        dataset="zoombench",
        split=f"test_{view}",
        question=query,
        images=[image_from_hf_image_feature(image_field)],
        choices=choices,
        answer=answer,
        question_type=question_type,
        category=question_type,
        metadata={
            "bbox": row.get("bbox"),
            "source": "inclusionAI/ZoomBench",
            "view": view,
        },
    )


def iter_mmstar(data_root: Path) -> Iterator[EvalExample]:
    root = resolve_dataset_root(data_root, "mmstar")
    parquet_path = root / "mmstar.parquet"
    if parquet_path.is_file():
        for row in _iter_parquet_records(parquet_path):
            yield _mmstar_example(row)
        return

    tsv_path = root / "MMStar.tsv"
    if tsv_path.is_file():
        csv.field_size_limit(sys.maxsize)
        with tsv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                yield _mmstar_example(row)
        return
    raise FileNotFoundError(f"MMStar parquet/TSV not found under: {root}")


def _mmstar_example(row: dict[str, Any]) -> EvalExample:
    prompt = str(row["question"])
    question, choices = parse_inline_options_question(prompt)
    answer = expected_mcq_letter(str(row["answer"]), choices)
    meta_info = row.get("meta_info") or {}
    metadata = {
        "source": "Lin-Chen/MMStar",
        "l2_category": str(row.get("l2_category", "")),
    }
    if isinstance(meta_info, dict):
        metadata.update(
            {
                key: value
                for key, value in meta_info.items()
                if value not in (None, "")
            }
        )
    return EvalExample(
        id=str(row["index"]),
        dataset="mmstar",
        split=str(metadata.get("split") or "val"),
        question=format_mcq_question(question, choices),
        images=[image_from_hf_image_feature(row["image"])],
        choices=choices,
        answer=answer,
        question_type="mcq",
        category=str(row.get("category", "")) or None,
        metadata=metadata,
    )


def iter_mme_realworld(data_root: Path, dataset: str = "mme_realworld") -> Iterator[EvalExample]:
    root = resolve_dataset_root(data_root, dataset)
    if dataset in {"mme_realworld_en", "mme_realworld_cn"}:
        json_paths = _mme_realworld_json_paths(root, dataset)
        if json_paths:
            for path in json_paths:
                for row in _iter_json_records(path):
                    yield _mme_realworld_example(row, dataset, root)
            return

    paths = sorted((root / "data").glob("train-*.parquet"))
    if not paths:
        paths = sorted(root.glob("train-*.parquet"))
    if not paths:
        paths = sorted(root.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"MME-RealWorld parquet shards not found under: {root}")

    for path in paths:
        for row in _iter_parquet_records(path):
            yield _mme_realworld_example(row, dataset, root)


def _mme_realworld_example(
    row: dict[str, Any],
    dataset: str = "mme_realworld",
    root: Path | None = None,
) -> EvalExample:
    choices = parse_labeled_options(row.get("multi-choice options") or row.get("Answer choices") or [])
    answer_value = row.get("answer", row.get("Ground truth"))
    answer = expected_mcq_letter(str(answer_value), choices)
    image_value = row.get("bytes")
    if image_value in (None, ""):
        image_value = row.get("image") or row.get("Image") or row.get("path")
    if root is not None:
        image = _mme_realworld_image(root, dataset, image_value)
    else:
        image = image_from_hf_image_feature(image_value)
    return EvalExample(
        id=str(row.get("index", row.get("Question_id"))),
        dataset=dataset,
        split="train",
        question=format_mme_realworld_question(
            str(row.get("question", row.get("Text"))),
            choices,
            dataset,
        ),
        images=[image],
        choices=choices,
        answer=answer,
        question_type="mcq",
        category=str(row.get("category", row.get("Category", ""))) or None,
        metadata={
            "l2_category": str(row.get("l2-category", row.get("Subtask", ""))),
            "task": str(row.get("Task", "")),
            "subtask": str(row.get("Subtask", "")),
            "question_type": str(row.get("Question Type", row.get("Question type", ""))),
            "path": str(row.get("path", row.get("Image", ""))),
            "source": _mme_realworld_source(dataset),
        },
    )


def format_mme_realworld_question(question: str, choices: dict[str, str], dataset: str) -> str:
    if dataset != "mme_realworld_cn":
        return format_mcq_question(question, choices)

    labels = ("A", "B", "C", "D", "E", "F")
    option_lines = [f"{label}. {choices[label]}" for label in labels if label in choices]
    return "\n".join(
        [
            question.strip(),
            *option_lines,
            "请从给定选项中回答选项字母。",
        ]
    )


def _mme_realworld_source(dataset: str) -> str:
    return {
        "mme_realworld": "yifanzhang114/MME-RealWorld-Lite",
        "mme_realworld_en": "MME-RW-en",
        "mme_realworld_cn": "MME-RW-cn",
    }.get(dataset, dataset)


def _mme_realworld_json_paths(root: Path, dataset: str) -> list[Path]:
    names = {
        "mme_realworld_en": ("MME_RealWorld.json", "MME-RealWorld.json"),
        "mme_realworld_cn": ("MME_RealWorld_CN.json", "MME-RealWorld-CN.json"),
    }.get(dataset, ())
    paths = [root / name for name in names if (root / name).is_file()]
    if paths:
        return paths

    folder_names = {
        "mme_realworld_en": ("MME-RealWorld",),
        "mme_realworld_cn": ("MME-RealWorld-CN", "MME-HD-CN"),
    }.get(dataset, ())
    for folder_name in folder_names:
        folder = root / folder_name
        if folder.is_dir():
            paths.extend(sorted(path for path in folder.glob("*.json") if path.is_file()))
    return paths


def _iter_json_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                for row in value:
                    yield dict(row)
        return
    for row in data:
        yield dict(row)


def _mme_realworld_image(root: Path, dataset: str, value: Any) -> EvalImage:
    if isinstance(value, str):
        try:
            value_path = Path(value)
        except (OSError, ValueError):
            return image_from_hf_image_feature(value)
        if not value_path.is_absolute():
            candidates = [root / value_path]
            if dataset == "mme_realworld_cn":
                candidates.append(root / "MME-HD-CN" / value_path)
            for candidate in candidates:
                if candidate.exists():
                    return path_image(str(candidate))
        return image_from_hf_image_feature(value)
    return image_from_hf_image_feature(value)


def iter_cvbench(data_root: Path) -> Iterator[EvalExample]:
    root = resolve_dataset_root(data_root, "cvbench")
    paths = _cvbench_parquet_paths(root)
    if not paths:
        raise FileNotFoundError(f"CV-Bench parquet files not found under: {root}")
    for path in paths:
        for row in _iter_parquet_records(path):
            yield _cvbench_example(row, root)


def _cvbench_parquet_paths(root: Path) -> list[Path]:
    preferred_patterns = (
        "data/default/test-*.parquet",
        "data/default/test_*.parquet",
        "default/test-*.parquet",
        "default/test_*.parquet",
        "data/test-*.parquet",
        "data/test_*.parquet",
        "test-*.parquet",
        "test_*.parquet",
    )
    for pattern in preferred_patterns:
        paths = sorted(path for path in root.glob(pattern) if path.is_file())
        if paths:
            return paths

    split_paths = []
    for pattern in (
        "data/2D/test-*.parquet",
        "2D/test-*.parquet",
        "data/3D/test-*.parquet",
        "3D/test-*.parquet",
    ):
        split_paths.extend(path for path in root.glob(pattern) if path.is_file())
    if split_paths:
        return sorted(split_paths)

    return sorted(path for path in root.rglob("*.parquet") if path.is_file())


def _cvbench_example(row: dict[str, Any], root: Path) -> EvalExample:
    raw_choices = row.get("choices") or []
    choices = {
        chr(ord("A") + index): str(choice)
        for index, choice in enumerate(raw_choices)
    }
    answer = expected_mcq_letter(str(row["answer"]), choices)
    image = _cvbench_image(root, row.get("image") or row.get("filename"))
    task_type = str(row.get("type", ""))
    task = str(row.get("task", ""))
    category = "/".join(part for part in (task_type, task) if part) or None
    return EvalExample(
        id=str(row.get("idx", row.get("filename", ""))),
        dataset="cvbench",
        split="test",
        question=format_mcq_question(str(row["question"]), choices),
        images=[image],
        choices=choices,
        answer=answer,
        question_type="mcq",
        category=category,
        metadata={
            "type": task_type,
            "task": task,
            "filename": str(row.get("filename", "")),
            "source": "nyu-visionx/CV-Bench",
            "source_dataset": str(row.get("source_dataset", "")),
            "source_filename": str(row.get("source_filename", "")),
            "target_class": str(row.get("target_class", "")),
            "target_size": row.get("target_size"),
            "bbox": row.get("bbox"),
        },
    )


def _cvbench_image(root: Path, value: Any) -> EvalImage:
    if isinstance(value, str):
        try:
            value_path = Path(value)
        except (OSError, ValueError):
            return image_from_hf_image_feature(value)
        if not value_path.is_absolute():
            candidate = root / value_path
            if candidate.exists():
                return path_image(candidate)
    return image_from_hf_image_feature(value)


def iter_colorbench(data_root: Path) -> Iterator[EvalExample]:
    root = resolve_dataset_root(data_root, "colorbench")
    paths = sorted((root / "data").glob("test-*.parquet"))
    if not paths:
        paths = sorted(root.glob("test-*.parquet"))
    if not paths:
        paths = sorted(root.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"ColorBench parquet files not found under: {root}")
    for path in paths:
        for row in _iter_parquet_records(path):
            yield _colorbench_example(row, root)


def _colorbench_example(row: dict[str, Any], root: Path) -> EvalExample:
    raw_choices = row.get("choices") or []
    choices = {
        chr(ord("A") + index): str(choice)
        for index, choice in enumerate(raw_choices)
    }
    answer = expected_mcq_letter(str(row["answer"]), choices)
    task_type = str(row.get("type", ""))
    task = str(row.get("task", ""))
    category = "/".join(part for part in (task_type, task) if part) or None
    return EvalExample(
        id=str(row.get("idx", row.get("id", ""))),
        dataset="colorbench",
        split="test",
        question=str(row.get("prompt") or format_mcq_question(str(row["question"]), choices)),
        images=[_colorbench_image(root, row.get("image") or row.get("filename"))],
        choices=choices,
        answer=answer,
        question_type="mcq",
        category=category,
        metadata={
            "id": row.get("id"),
            "type": task_type,
            "task": task,
            "filename": str(row.get("filename", "")),
            "image_url": str(row.get("image_url", "")),
            "image_type": str(row.get("image_type", "")),
            "source": "umd-zhou-lab/ColorBench",
        },
    )


def _colorbench_image(root: Path, value: Any) -> EvalImage:
    if isinstance(value, str):
        try:
            value_path = Path(value)
        except (OSError, ValueError):
            return image_from_hf_image_feature(value)
        if not value_path.is_absolute():
            for candidate in (root / value_path, root / "images" / value_path):
                if candidate.exists():
                    return path_image(candidate)
    return image_from_hf_image_feature(value)


def iter_babyvision(data_root: Path) -> Iterator[EvalExample]:
    root = resolve_dataset_root(data_root, "babyvision")
    paths = sorted((root / "data").glob("train-*.parquet"))
    if not paths:
        paths = sorted(root.glob("train-*.parquet"))
    if not paths:
        paths = sorted(root.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"BabyVision parquet files not found under: {root}")
    for path in paths:
        for row in _iter_parquet_records(path):
            yield _babyvision_example(row)


def _babyvision_example(row: dict[str, Any]) -> EvalExample:
    ans_type = str(row.get("ansType", "")).strip().lower()
    task_type = str(row.get("type", ""))
    subtype = str(row.get("subtype", ""))
    category = "/".join(part for part in (task_type, subtype) if part) or None
    raw_choices = row.get("options") or []
    choices = {
        chr(ord("A") + index): str(choice)
        for index, choice in enumerate(raw_choices)
    }

    if ans_type == "choice":
        choice_index = row.get("choiceAns")
        if choice_index is None:
            answer = expected_mcq_letter(str(row.get("blankAns", "")), choices)
        else:
            answer = chr(ord("A") + int(choice_index))
        question = format_mcq_question(str(row["question"]), choices)
        question_type = "mcq"
    else:
        answer = str(row.get("blankAns", ""))
        question = str(row["question"])
        question_type = "open"

    return EvalExample(
        id=str(row.get("taskId", "")),
        dataset="babyvision",
        split="train",
        question=question,
        images=[image_from_hf_image_feature(row["image"])],
        choices=choices if ans_type == "choice" else {},
        answer=answer,
        question_type=question_type,
        category=category,
        metadata={
            "status": str(row.get("status", "")),
            "type": task_type,
            "subtype": subtype,
            "ans_type": ans_type,
            "choice_ans": row.get("choiceAns"),
            "cot": str(row.get("coT", "")),
            "source": "UnipatAI/BabyVision",
        },
    )


def iter_pope(data_root: Path) -> Iterator[EvalExample]:
    """Yield the three official COCO POPE evaluation splits."""
    root = resolve_dataset_root(data_root, "pope")
    question_paths = _pope_question_paths(root)
    if not question_paths:
        raise FileNotFoundError(f"POPE question files not found under: {root}")

    for path in question_paths:
        split = path.stem.rsplit("_", 1)[-1].lower()
        with path.open(encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                if not line.strip():
                    continue
                row = json.loads(line)
                image_name = str(row["image"])
                image_path = _pope_image_path(root, image_name)
                answer = str(row["label"]).strip().lower()
                if answer not in {"yes", "no"}:
                    raise ValueError(
                        f"Unsupported POPE label {answer!r} in {path}:{row_index + 1}"
                    )
                question_id = str(row.get("question_id", row_index))
                yield EvalExample(
                    id=f"{split}:{question_id}",
                    dataset="pope",
                    split=split,
                    question=(
                        f"{str(row['text']).strip()}\n"
                        "Answer the question using a single word: yes or no."
                    ),
                    images=[path_image(image_path)],
                    answer=answer,
                    question_type="binary",
                    category=split,
                    metadata={
                        "question_id": row.get("question_id"),
                        "image": image_name,
                        "source": "RUCAIBox/POPE",
                        "sampling_strategy": split,
                    },
                )


def _pope_question_paths(root: Path) -> list[Path]:
    patterns = (
        "questions/*_pope_random.json",
        "questions/*_pope_popular.json",
        "questions/*_pope_adversarial.json",
        "output/*/*_pope_random.json",
        "output/*/*_pope_popular.json",
        "output/*/*_pope_adversarial.json",
        "*_pope_random.json",
        "*_pope_popular.json",
        "*_pope_adversarial.json",
    )
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(path for path in root.glob(pattern) if path.is_file())
    return sorted(set(paths))


def _pope_image_path(root: Path, image_name: str) -> Path:
    image_path = Path(image_name)
    candidates = (
        image_path if image_path.is_absolute() else root / image_path,
        root / "val2014" / image_path.name,
        root / "images" / image_path.name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"POPE image {image_name!r} not found; tried: {tried}")


def iter_mmvp(data_root: Path) -> Iterator[EvalExample]:
    """Yield MMVP multiple-choice samples while preserving official pairs."""
    root = resolve_dataset_root(data_root, "mmvp")
    questions_path = root / "Questions.csv"
    if not questions_path.is_file():
        raise FileNotFoundError(f"MMVP Questions.csv not found: {questions_path}")

    pair_questions: dict[int, str] = {}
    with questions_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            index = int(str(row["Index"]).strip())
            question = str(row["Question"]).strip()
            choices = parse_mmvp_options(str(row["Options"]))
            answer = parse_mmvp_answer(str(row["Correct Answer"]), choices)
            image_path = _mmvp_image_path(root, index)
            pair_id = (index - 1) // 2 + 1
            paired_question = pair_questions.setdefault(pair_id, question)
            if paired_question != question:
                raise ValueError(
                    f"MMVP pair {pair_id} has mismatched questions in "
                    f"{questions_path}:{row_index + 2}"
                )
            yield EvalExample(
                id=str(index),
                dataset="mmvp",
                split="test",
                question=format_mcq_question(question, choices),
                images=[path_image(image_path)],
                choices=choices,
                answer=answer,
                question_type="mcq",
                metadata={
                    "pair_id": pair_id,
                    "pair_position": (index - 1) % 2,
                    "source": "MMVP/MMVP",
                    "source_row": row_index,
                },
            )


def parse_mmvp_options(options_text: str) -> dict[str, str]:
    """Parse the two official MMVP options, labelled ``(a)`` and ``(b)``."""
    choices: dict[str, str] = {}
    matches = list(MMVP_OPTION_RE.finditer(options_text))
    labels = tuple(match.group(1).upper() for match in matches)
    if labels != MMVP_OPTION_LABELS:
        raise ValueError(
            "MMVP options must contain exactly '(a)' and '(b)' in that order: "
            f"{options_text!r}"
        )
    for index, match in enumerate(matches):
        label = match.group(1).upper()
        next_start = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(options_text)
        )
        option = options_text[match.end() : next_start].strip(" ,;\t\r\n")
        if option and label not in choices:
            choices[label] = option
    if tuple(choices) != MMVP_OPTION_LABELS or any(
        not option for option in choices.values()
    ):
        raise ValueError(f"Could not parse the two MMVP options: {options_text!r}")
    return choices


def parse_mmvp_answer(answer_text: str, choices: dict[str, str]) -> str:
    """Normalize an MMVP answer such as ``(a)`` to its option letter."""
    match = MMVP_OPTION_RE.search(answer_text)
    answer = match.group(1).upper() if match else answer_text.strip().upper()
    if answer not in MMVP_OPTION_LABELS or answer not in choices:
        raise ValueError(f"MMVP answer {answer_text!r} must be '(a)' or '(b)'")
    return answer


def _mmvp_image_path(root: Path, index: int) -> Path:
    image_names = (f"{index}.jpg", f"{index}.jpeg", f"{index}.png")
    image_dirs = (root / "MMVP Images", root / "images", root)
    for image_dir in image_dirs:
        for image_name in image_names:
            candidate = image_dir / image_name
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"MMVP image for Index={index} not found under: {root}")


def parse_inline_options_question(prompt: str) -> tuple[str, dict[str, str]]:
    if "Options:" not in prompt:
        return prompt, choice_map_from_prompt(prompt)

    question, options_text = prompt.split("Options:", 1)
    choices: dict[str, str] = {}
    matches = list(INLINE_OPTION_RE.finditer(options_text))
    for index, match in enumerate(matches):
        label = match.group(1).upper()
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(options_text)
        option = options_text[match.end() : next_start].strip(" ,;\t\r\n")
        if option and label not in choices:
            choices[label] = option
    return question.strip(), choices


def parse_labeled_options(options: list[Any] | str) -> dict[str, str]:
    if isinstance(options, str):
        try:
            parsed_options = ast.literal_eval(options)
        except (SyntaxError, ValueError):
            parsed_options = options.splitlines()
        options = parsed_options if isinstance(parsed_options, list) else [parsed_options]
    choices: dict[str, str] = {}
    for option in options:
        match = LABELED_OPTION_RE.match(str(option))
        if not match:
            continue
        label = match.group(1).upper()
        if label not in choices:
            choices[label] = match.group(2).strip()
    return choices


def image_from_hf_image_feature(value: Any) -> EvalImage:
    if isinstance(value, dict):
        path = value.get("path")
        data = value.get("bytes")
        if data:
            return bytes_image(data)
        if path:
            return path_image(path)
    if isinstance(value, (bytes, bytearray)):
        return bytes_image(bytes(value))
    if isinstance(value, str):
        try:
            if Path(value).exists():
                return path_image(value)
        except OSError:
            pass
        try:
            base64.b64decode(value, validate=True)
        except Exception:
            return path_image(value)
        return base64_image(value)
    return pil_image(value)


def format_mcq_question(question: str, choices: dict[str, str]) -> str:
    labels = ("A", "B", "C", "D", "E", "F")
    option_lines = [f"{label}. {choices[label]}" for label in labels if label in choices]
    return "\n".join(
        [
            question.strip(),
            *option_lines,
            "Answer with the option's letter from the given choices.",
        ]
    )


def resolve_dataset_root(data_root: Path, dataset: str) -> Path:
    data_root = data_root.expanduser()
    candidates = _dataset_candidates(data_root, dataset)
    for candidate in candidates:
        if candidate.exists() and _candidate_matches_dataset(candidate, dataset):
            return candidate
    owner, repo = HF_REPOS[dataset]
    hint = "\n".join(f"  - {candidate}" for candidate in candidates[:8])
    raise FileNotFoundError(
        f"Could not locate dataset {owner}/{repo} under {data_root}. Tried:\n{hint}"
    )


def _candidate_matches_dataset(candidate: Path, dataset: str) -> bool:
    if dataset == "cvbench":
        return bool(_cvbench_candidate_parquet_paths(candidate))
    if dataset == "colorbench":
        return bool(_colorbench_candidate_parquet_paths(candidate))
    if dataset == "babyvision":
        return bool(_babyvision_candidate_parquet_paths(candidate))
    if dataset == "pope":
        return bool(_pope_question_paths(candidate)) and any(
            (candidate / image_dir).is_dir()
            for image_dir in ("val2014", "images")
        )
    if dataset == "mmvp":
        return (candidate / "Questions.csv").is_file() and any(
            (candidate / image_dir).is_dir()
            for image_dir in ("MMVP Images", "images")
        )
    if dataset in {"mme_realworld_en", "mme_realworld_cn"}:
        if _mme_realworld_json_paths(candidate, dataset):
            return True
        return bool(
            list((candidate / "data").glob("train-*.parquet"))
            or list(candidate.glob("train-*.parquet"))
            or list(candidate.rglob("*.parquet"))
        )
    return True


def _cvbench_candidate_parquet_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in (
        "data/default/test-*.parquet",
        "data/default/test_*.parquet",
        "default/test-*.parquet",
        "default/test_*.parquet",
        "data/test-*.parquet",
        "data/test_*.parquet",
        "test-*.parquet",
        "test_*.parquet",
        "data/2D/test-*.parquet",
        "2D/test-*.parquet",
        "data/3D/test-*.parquet",
        "3D/test-*.parquet",
    ):
        paths.extend(path for path in root.glob(pattern) if path.is_file())
    return sorted(paths)


def _colorbench_candidate_parquet_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in ("data/test-*.parquet", "test-*.parquet"):
        paths.extend(path for path in root.glob(pattern) if path.is_file())
    return sorted(paths)


def _babyvision_candidate_parquet_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in ("data/train-*.parquet", "train-*.parquet"):
        paths.extend(path for path in root.glob(pattern) if path.is_file())
    return sorted(paths)


def _dataset_candidates(data_root: Path, dataset: str) -> list[Path]:
    owner, repo = HF_REPOS[dataset]
    simple_dirs = {
        "vstar": ["vstar_bench", "vstar-bench", repo],
        "hrbench_4k": ["hr_bench", "hr-bench", "HR-Bench", repo],
        "hrbench_8k": ["hr_bench", "hr-bench", "HR-Bench", repo],
        "zoombench": ["zoom_bench", "zoombench", "ZoomBench", repo],
        "mmstar": ["MMStar", "mmstar", "mm_star", repo],
        "cvbench": ["CV-Bench", "cvbench", "cv_bench", "cv-bench", repo],
        "colorbench": ["ColorBench", "colorbench", "color_bench", "color-bench", repo],
        "babyvision": ["BabyVision", "babyvision", "baby_vision", "baby-vision", repo],
        "pope": ["POPE", "pope", "pope_coco", "pope-coco", repo],
        "mmvp": ["MMVP", "mmvp", "mmvp_benchmark", "mmvp-benchmark", repo],
        "mme_realworld": [
            "MME-RealWorld-lite-lmms-eval",
            "MME-RealWorld-Lite",
            "mme_realworld",
            "mme-realworld",
            repo,
        ],
        "mme_realworld_en": [
            "MME-RealWorld",
            "MME-RW-en",
            "MME-RW-EN",
            "MME-RealWorld-en",
            "MME-RealWorld-EN",
            "MME-RealWorld/MME-RW-en",
            "MME-RealWorld/MME-RW-EN",
            "MME-RealWorld/MME-HD-EN",
            repo,
        ],
        "mme_realworld_cn": [
            "MME-RealWorld",
            "MME-RW-cn",
            "MME-RW-CN",
            "MME-RealWorld-cn",
            "MME-RealWorld-CN",
            "MME-RealWorld/MME-RW-cn",
            "MME-RealWorld/MME-RW-CN",
            "MME-RealWorld/MME-HD-CN",
            repo,
        ],
    }[dataset]
    candidates = [data_root / name for name in simple_dirs]
    if dataset not in {"mme_realworld_en", "mme_realworld_cn"}:
        candidates.append(data_root)
    hub_rel = Path(f"datasets--{owner}--{repo}") / "snapshots"
    for base in (data_root / hub_rel, data_root / "hub" / hub_rel):
        if base.is_dir():
            candidates.extend(path for path in base.iterdir() if path.is_dir())
    return candidates


def _iter_parquet_records(path: Path, batch_size: int = 8) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Reading parquet datasets requires the optional dependency "
            "'pyarrow'. Install project dependencies first."
        ) from exc

    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            yield dict(row)
