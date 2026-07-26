from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


DEFAULT_DATASET = "data/annotation/train.parquet"
DEFAULT_OUTPUT_DIR = "results/annotation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use Doubao Seed / Volcengine Ark visual grounding to relabel each sample "
            "with a question-relevant bbox and point."
        )
    )
    parser.add_argument("--dataset", type=Path, default=Path(os.environ.get("SEED_DATASET", DEFAULT_DATASET)))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("SEED_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)),
    )
    parser.add_argument("--output-name", default=os.environ.get("SEED_OUTPUT_NAME"))
    parser.add_argument("--image-root", type=Path, default=optional_path_env("SEED_IMAGE_ROOT"))
    parser.add_argument("--model", default=os.environ.get("ARK_MODEL", "doubao-seed-2-0-pro-260215"))
    parser.add_argument("--base-url", default=os.environ.get("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"))
    parser.add_argument("--api-key-env", default=os.environ.get("ARK_API_KEY_ENV", "ARK_API_KEY"))
    parser.add_argument(
        "--mode",
        choices=("batch", "online"),
        default=os.environ.get("SEED_MODE", "batch"),
        help="batch sends concurrent requests and collects results in one run; online is an alias for debugging.",
    )
    parser.add_argument("--limit", type=int, default=optional_int_env("SEED_LIMIT"))
    parser.add_argument("--start-index", type=int, default=int(os.environ.get("SEED_START_INDEX", "0")))
    parser.add_argument(
        "--random",
        action=argparse.BooleanOptionalAction,
        default=bool_env("SEED_RANDOM", False),
        help="shuffle eligible samples before applying --limit",
    )
    parser.add_argument("--random-seed", type=int, default=optional_int_env("SEED_RANDOM_SEED"))
    parser.add_argument("--read-batch-size", type=int, default=int(os.environ.get("SEED_READ_BATCH_SIZE", "1024")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SEED_WORKERS", "4")))
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("SEED_TIMEOUT", "120")))
    parser.add_argument("--max-retries", type=int, default=int(os.environ.get("SEED_MAX_RETRIES", "2")))
    parser.add_argument("--retry-sleep", type=float, default=float(os.environ.get("SEED_RETRY_SLEEP", "2")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("SEED_TEMPERATURE", "0")))
    parser.add_argument("--top-p", type=float, default=float(os.environ.get("SEED_TOP_P", "1")))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("SEED_MAX_TOKENS", "128")))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=bool_env("SEED_RESUME", True))
    parser.add_argument("--write-train-parquet", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-crops", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--crop-output-dir", type=Path, default=optional_path_env("SEED_CROP_OUTPUT_DIR"))
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {args.dataset}")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be at least 1.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1 when set.")


def optional_int_env(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return int(value)


def optional_path_env(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    return Path(value)


def bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "label"
