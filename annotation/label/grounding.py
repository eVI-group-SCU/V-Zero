from __future__ import annotations

import re


def build_grounding_prompt(question: str) -> str:
    return (
        "Locate the visual region that is most useful for answering the question.\n"
        "Requirements:\n"
        "1. Output one bounding box around the smallest key region needed to answer the question.\n"
        "2. Output one point at the center of the key target or region.\n"
        "3. Use normalized integer coordinates on a 1000 x 1000 canvas. The valid range is [0, 999], "
        "with the image top-left corner as the origin.\n"
        "4. If the question depends on text, signs, numbers, or fine-grained local details, "
        "box the corresponding text or detail region.\n"
        "5. Strictly output only the following two lines. Do not explain. Do not output JSON:\n"
        "<bbox>x_min y_min x_max y_max</bbox>\n"
        "<point>x y</point>\n\n"
        f"Question: {question}"
    )


def parse_grounding_response(text: str) -> dict[str, list[int]]:
    bbox_match = re.search(r"<bbox>\s*([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s*</bbox>", text)
    point_match = re.search(r"<point>\s*([0-9]+)\s+([0-9]+)\s*</point>", text)
    if bbox_match is None:
        raise ValueError(f"Missing <bbox> in response: {text!r}")
    bbox = [clamp_1000(int(item)) for item in bbox_match.groups()]
    if bbox[2] < bbox[0]:
        bbox[0], bbox[2] = bbox[2], bbox[0]
    if bbox[3] < bbox[1]:
        bbox[1], bbox[3] = bbox[3], bbox[1]
    if bbox[2] == bbox[0]:
        bbox[2] = min(999, bbox[0] + 1)
    if bbox[3] == bbox[1]:
        bbox[3] = min(999, bbox[1] + 1)
    if point_match is None:
        point = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]
    else:
        point = [clamp_1000(int(item)) for item in point_match.groups()]
    return {"bbox_1000": bbox, "point_1000": point}


def bbox_to_pixel(bbox: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = bbox
    return [
        clamp(int(round(x1 * width / 1000)), 0, max(width - 1, 0)),
        clamp(int(round(y1 * height / 1000)), 0, max(height - 1, 0)),
        clamp(int(round(x2 * width / 1000)), 0, width),
        clamp(int(round(y2 * height / 1000)), 0, height),
    ]


def point_to_pixel(point: list[int], width: int, height: int) -> list[int]:
    x, y = point
    return [
        clamp(int(round(x * width / 1000)), 0, max(width - 1, 0)),
        clamp(int(round(y * height / 1000)), 0, max(height - 1, 0)),
    ]


def clamp_1000(value: int) -> int:
    return clamp(value, 0, 999)


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))
