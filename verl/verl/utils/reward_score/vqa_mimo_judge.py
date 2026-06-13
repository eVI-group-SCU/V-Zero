# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""OpenAI-compatible MiMo judge scorer for VQA validation data."""

from __future__ import annotations

import os
import re
import time
import warnings
from typing import Any

JUDGE_PROMPT = """Question:
{question}

Ground truth answer:
{ground_truth}

Model answer:
{response}

Decide whether the model answer matches the ground truth answer for this VQA item.

Rules:
1. If the question is multiple choice and the ground truth is an option label such as A, B, C, D, or E, judge by the selected option label, not by exact string equality.
2. For multiple choice questions, answer Yes when the model clearly selects the same option label as the ground truth, even if it also includes the option text, punctuation, formatting, or explanation.
3. For multiple choice questions, answer No when the model clearly selects a different option label, gives conflicting final option labels, or never clearly selects the ground-truth option.
4. For non-multiple-choice questions, answer Yes only when the model answer is semantically equivalent to the ground truth; minor wording or formatting differences are allowed.

Examples:
- Ground truth answer: C
  Model answer: C. Ely Diocese
  Decision: Yes
  Reason: the model selects option C; the extra option text does not make it wrong.
- Ground truth answer: C
  Model answer: D. Ely Cathedral
  Decision: No
  Reason: the model selects option D, not option C.
- Ground truth answer: A
  Model answer: The answer is (A) crossing the crosswalk.
  Decision: Yes
  Reason: the model selects option A.
- Ground truth answer: A
  Model answer: C. walking on the sidewalk
  Decision: No
  Reason: the model selects option C, not option A.

Reply with exactly one word on the first line: Yes or No."""

_CLIENT_CACHE: dict[tuple[str, str], Any] = {}


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _falsey_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"0", "false", "no", "off"}


def _api_config() -> tuple[str | None, str | None]:
    api_key = _env_first("VQA_JUDGE_API_KEY", "JUDGE_API_KEY", "JUDGE_OPENAI_API_KEY")
    base_url = _env_first(
        "VQA_JUDGE_BASE_URL",
        "JUDGE_BASE_OPENAI_URL",
        "JUDGE_OPENAI_BASE_URL",
        "JUDGE_BASE_URL",
    )
    return api_key, base_url


def _judge_model() -> str:
    return _env_first("VQA_JUDGE_MODEL", "JUDGE_MODEL", "API_JUDGE_MODEL") or "mimo-v2.5"


def _client(api_key: str, base_url: str) -> Any:
    cache_key = (api_key, base_url)
    if cache_key not in _CLIENT_CACHE:
        from openai import OpenAI

        _CLIENT_CACHE[cache_key] = OpenAI(api_key=api_key, base_url=base_url)
    return _CLIENT_CACHE[cache_key]


def _completion_message_text(message: Any) -> str:
    parts: list[str] = []
    content = getattr(message, "content", None)
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                text = part.get("text") or part.get("content")
            else:
                text = getattr(part, "text", None) or getattr(part, "content", None)
            if isinstance(text, str):
                parts.append(text)
    for attr in ("reasoning_content", "reasoning", "refusal"):
        value = getattr(message, attr, None)
        if isinstance(value, str):
            parts.append(value)
    extra = getattr(message, "model_extra", None)
    if isinstance(extra, dict):
        for key in ("reasoning_content", "reasoning", "reasoning_text", "content"):
            value = extra.get(key)
            if isinstance(value, str):
                parts.append(value)
    return "\n".join(part.strip() for part in parts if part and part.strip())


def _parse_judge_decision(text: str) -> bool | None:
    cleaned = text.strip()
    if not cleaned:
        return None
    match = re.search(r"\b(yes|no)\b", cleaned, re.IGNORECASE)
    if match:
        return match.group(1).lower() == "yes"
    if re.search(r"(^|[^\w])是($|[^\w])", cleaned):
        return True
    if re.search(r"(^|[^\w])否($|[^\w])", cleaned):
        return False
    if re.search(r"\b(true|correct)\b", cleaned, re.IGNORECASE):
        return True
    if re.search(r"\b(false|incorrect)\b", cleaned, re.IGNORECASE):
        return False
    return None


def _question_from_extra(extra_info: dict[str, Any] | None) -> str:
    if not extra_info:
        return ""
    question = extra_info.get("question")
    if question is None:
        question = extra_info.get("raw_question") or extra_info.get("prompt")
    if isinstance(question, list):
        question = "\n".join(str(item) for item in question)
    question_text = str(question or "").replace("<image>\n", "").replace("<image>", "").strip()

    choices = extra_info.get("choices")
    if choices is not None:
        choices_text = str(choices).strip()
        if choices_text and choices_text not in question_text:
            question_text = f"{question_text}\n\nOptions:\n{choices_text}" if question_text else f"Options:\n{choices_text}"

    return question_text


def _request_judge(
    *,
    question: str,
    response: str,
    ground_truth: str,
    model: str,
    api_key: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> str:
    user_prompt = JUDGE_PROMPT.format(question=question, ground_truth=ground_truth, response=response)
    completion = _client(api_key, base_url).chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a strict VQA answer evaluator."},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return _completion_message_text(completion.choices[0].message)


def _zero_score_result(
    *,
    model: str | None,
    judge_used: bool,
    error: str | None,
    data_source: str | None,
    response: str = "",
    latency_sec: float | None = None,
) -> dict[str, Any]:
    if error:
        warnings.warn(f"MiMo judge failed; assigning score 0.0. {error}", RuntimeWarning, stacklevel=2)
    return {
        "score": 0.0,
        "acc": 0.0,
        "judge_used": judge_used,
        "judge_model": model,
        "judge_correct": None,
        "judge_response": response,
        "judge_latency_sec": latency_sec,
        "judge_error": error,
        "judge_data_source": data_source,
    }


def compute_score(
    predict_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    data_source: str | None = None,
) -> dict[str, Any]:
    if _falsey_env("VQA_MIMO_JUDGE"):
        return _zero_score_result(
            model=None,
            judge_used=False,
            error="VQA_MIMO_JUDGE is disabled.",
            data_source=data_source,
        )

    model = _judge_model()
    api_key, base_url = _api_config()
    if not api_key or not base_url:
        message = (
            "Missing MiMo judge API config. Set VQA_JUDGE_API_KEY/JUDGE_API_KEY and "
            "VQA_JUDGE_BASE_URL/JUDGE_BASE_OPENAI_URL."
        )
        return _zero_score_result(model=model, judge_used=False, error=message, data_source=data_source)

    question = _question_from_extra(extra_info)
    temperature = float(os.environ.get("VQA_JUDGE_TEMPERATURE", os.environ.get("JUDGE_TEMPERATURE", "0")))
    max_tokens = int(os.environ.get("VQA_JUDGE_MAX_TOKENS", os.environ.get("JUDGE_MAX_TOKENS", "2048")))
    timeout = float(os.environ.get("VQA_JUDGE_TIMEOUT", os.environ.get("JUDGE_TIMEOUT", "30")))
    retries = int(os.environ.get("VQA_JUDGE_RETRIES", os.environ.get("JUDGE_RETRIES", "2")))
    retry_sleep = float(os.environ.get("VQA_JUDGE_RETRY_SLEEP", os.environ.get("JUDGE_RETRY_SLEEP", "5")))

    started = time.time()
    last_error: str | None = None
    last_response = ""
    for attempt in range(retries + 1):
        try:
            judge_text = _request_judge(
                question=question,
                response=predict_str,
                ground_truth=str(ground_truth),
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            last_response = judge_text
            decision = _parse_judge_decision(judge_text)
            if decision is None:
                last_error = "Could not parse Yes/No judge response."
                if attempt >= retries:
                    return _zero_score_result(
                        model=model,
                        judge_used=True,
                        error=last_error,
                        data_source=data_source,
                        response=last_response,
                        latency_sec=round(time.time() - started, 3),
                    )
                time.sleep(retry_sleep)
                continue
            score = 1.0 if decision else 0.0
            return {
                "score": score,
                "acc": score,
                "judge_used": True,
                "judge_model": model,
                "judge_correct": decision,
                "judge_response": judge_text,
                "judge_latency_sec": round(time.time() - started, 3),
                "judge_error": None,
                "judge_data_source": data_source,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt >= retries:
                return _zero_score_result(
                    model=model,
                    judge_used=True,
                    error=last_error,
                    data_source=data_source,
                    response=last_response,
                    latency_sec=round(time.time() - started, 3),
                )
            time.sleep(retry_sleep)

    raise AssertionError("unreachable")
