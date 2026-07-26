from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


ImageKind = Literal["path", "base64", "bytes", "pil"]


@dataclass(slots=True)
class EvalImage:
    kind: ImageKind
    value: Any


@dataclass(slots=True)
class EvalExample:
    id: str
    dataset: str
    split: str
    question: str
    images: list[EvalImage]
    answer: str
    choices: dict[str, str] = field(default_factory=dict)
    question_type: str | None = None
    category: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalResult:
    ok: bool
    row_index: int
    id: str
    dataset: str
    split: str
    model: str
    engine: str
    answer: str | None = None
    prediction: str | None = None
    correct: bool | None = None
    accuracy: float | None = None
    strict_prediction: str | None = None
    strict_correct: bool | None = None
    response: str | None = None
    category: str | None = None
    question_type: str | None = None
    generation_elapsed_sec: float | None = None
    batch_generation_elapsed_sec: float | None = None
    judge_enabled: bool = False
    judge_provider: str | None = None
    judge_task: str | None = None
    judge_model: str | None = None
    judge_prediction: str | None = None
    judge_correct: bool | None = None
    judge_used_for_score: bool = False
    judge_response: str | None = None
    judge_error: str | None = None
    judge_elapsed_sec: float | None = None
    sample_eval_elapsed_sec: float | None = None
    total_eval_elapsed_sec: float | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def path_image(path: str | Path) -> EvalImage:
    return EvalImage(kind="path", value=str(path))


def base64_image(value: str) -> EvalImage:
    return EvalImage(kind="base64", value=value)


def bytes_image(value: bytes) -> EvalImage:
    return EvalImage(kind="bytes", value=value)


def pil_image(value: Any) -> EvalImage:
    return EvalImage(kind="pil", value=value)
