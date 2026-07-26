from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Example:
    global_index: int
    row_index: int
    source_file: str
    row: dict[str, Any]
    image_value: Any
    image_source: str
    question: str
    answer: str | None
    image_width: int
    image_height: int
    image_data_url: str
